from typing import Tuple

import warp as wp

from .ogc import Ogc, OgcData
from .constraints import ConstraintSet, ConstraintSetData
from .multimesh import MultiMeshData
from . import halfedges

# TODO: Review/refactor/improve the dual update implementation.


# ---------------------------------------------------------------------------
# Gap vector functions — one per contact type
# ---------------------------------------------------------------------------


@wp.func
def _gap_vv(
    x: wp.array[wp.vec3f],
    V: wp.array[wp.int32],
    u: wp.int32,
    v: wp.int32,
) -> wp.vec3f:
    """Gap vector (x_u - x_v) for a vertex-vertex contact pair."""
    return x[V[u]] - x[V[v]]  # type: ignore


@wp.func
def _gap_ve(
    x: wp.array[wp.vec3f],
    V: wp.array[wp.int32],
    F: wp.array[wp.vec3i],
    t: wp.float32,
    u: wp.int32,
    v: wp.int32,
) -> wp.vec3f:
    """Gap vector x_u - xcp_v for a vertex-edge contact pair.

    u : vertex primitive index (into V -> global point)
    v : half-edge index; its endpoints are the edge vertices
    t : cached closest-point parameter on the edge (from OgcData.ve_bary)
    """
    xi = x[V[u]]  # type: ignore
    i_he = halfedges.incoming_vertex(F, v)  # global point index
    j_he = halfedges.outgoing_vertex(F, v)  # global point index
    xcp = (wp.float32(1) - t) * x[i_he] + t * x[j_he]  # type: ignore
    return xi - xcp


@wp.func
def _gap_vf(
    x: wp.array[wp.vec3f],
    V: wp.array[wp.int32],
    F: wp.array[wp.vec3i],
    bary: wp.vec2f,
    u: wp.int32,
    v: wp.int32,
) -> wp.vec3f:
    """Gap vector x_u - xcp_v for a vertex-face contact pair.

    u    : vertex primitive index (into V -> global point)
    v    : face (triangle) index
    bary : cached (b0, b1) barycentric coords; b2 = 1 - b0 - b1 (from OgcData.vf_bary)
    """
    xi = x[V[u]]  # type: ignore
    finds = F[v]
    xj, xk, xl = x[finds[0]], x[finds[1]], x[finds[2]]  # type: ignore
    b0, b1 = bary[0], bary[1]
    b2 = wp.float32(1) - b0 - b1
    xcp = b0 * xj + b1 * xk + b2 * xl  # type: ignore
    return xi - xcp


@wp.func
def _gap_ee(
    x: wp.array[wp.vec3f],
    F: wp.array[wp.vec3i],
    bary: wp.vec2f,
    u: wp.int32,
    v: wp.int32,
) -> wp.vec3f:
    """Gap vector xcp_u - xcp_v for an edge-edge contact pair.

    u    : half-edge index (edge 1)
    v    : half-edge index (edge 2)
    bary : cached (s, t) parameters on edge 1 and edge 2 (from OgcData.ee_bary)
    """
    s, t = bary[0], bary[1]
    i_u = halfedges.incoming_vertex(F, u)
    j_u = halfedges.outgoing_vertex(F, u)
    i_v = halfedges.incoming_vertex(F, v)
    j_v = halfedges.outgoing_vertex(F, v)
    xcp_u = (wp.float32(1) - s) * x[i_u] + s * x[j_u]  # type: ignore
    xcp_v = (wp.float32(1) - t) * x[i_v] + t * x[j_v]  # type: ignore
    return xcp_u - xcp_v


# ---------------------------------------------------------------------------
# Project gap onto contact basis
# ---------------------------------------------------------------------------


@wp.func
def _project_gap(
    gap: wp.vec3f,
    basis: wp.mat33f,
) -> Tuple[wp.float32, wp.vec2f]:  # type: ignore
    """Decompose gap into normal and tangential components.

    basis rows: [n (normal), t1 (first tangent), t2 (bitangent)].
    Returns (c_n, c_f) where c_n = dot(gap, n) and c_f = (dot(gap, t1), dot(gap, t2)).
    """
    n  = wp.vec3f(basis[0, 0], basis[0, 1], basis[0, 2])
    t1 = wp.vec3f(basis[1, 0], basis[1, 1], basis[1, 2])
    t2 = wp.vec3f(basis[2, 0], basis[2, 1], basis[2, 2])
    c_n = wp.dot(gap, n)
    c_f = wp.vec2f(wp.dot(gap, t1), wp.dot(gap, t2))
    return c_n, c_f


# ---------------------------------------------------------------------------
# AL dual update core
# ---------------------------------------------------------------------------


@wp.func
def _apply_dual_update(
    c_n: wp.float32,
    c_f: wp.vec2f,
    s_prev: wp.float32,
    gamma_k: wp.float32,
    lambda_n_k: wp.float32,
    lambda_f_k: wp.vec2f,
    mu_n: wp.float32,
    mu_f: wp.float32,
    mu_friction: wp.float32,
    dmin: wp.float32,
    decay_rate: wp.float32,
) -> Tuple[wp.float32, wp.float32, wp.float32, wp.vec2f]:  # type: ignore
    """Augmented-Lagrangian dual variable update.

    GPU equivalent of ``MeshDynamics::UpdateDual<Slack|Decay|LagrangeMultiplier>``
    from the C++ side.  Rather than evaluating a cached Taylor expansion, the
    caller supplies the already-projected gap components ``(c_n, c_f)``.

    Update order (mirrors C++):
      1. Slack   : s = max(0, c_n - dmin - lambda_n / mu_n)
      2. Decay   : gamma = 1 if s == 0 (contact active);
                         gamma *= decay_rate if s > s_prev (separating).
      3. Multipliers: if s == 0
             lambda_n -= mu_n * (c_n - dmin)
             lambda_f -= mu_f * c_f  then project into Coulomb cone
         else
             lambda_n = 0,  lambda_f = 0
    """
    # --- Slack ---
    new_s = wp.max(wp.float32(0), c_n - dmin - lambda_n_k / mu_n)
    # --- Decay ---
    new_gamma = gamma_k
    if new_s == wp.float32(0):
        new_gamma = wp.float32(1)
    elif new_s > s_prev:  # contact gap growing → separating
        new_gamma = gamma_k * decay_rate
    # --- Lagrange multipliers ---
    new_lambda_n = lambda_n_k
    new_lambda_f = lambda_f_k
    if new_s == wp.float32(0):
        new_lambda_n = lambda_n_k - mu_n * (c_n - dmin)
        new_lambda_f = lambda_f_k - mu_f * c_f
        # Coulomb friction cone: |lambda_f| <= mu * lambda_n
        friction_limit = mu_friction * new_lambda_n
        lf_sq = wp.dot(new_lambda_f, new_lambda_f)
        if lf_sq > friction_limit * friction_limit:
            new_lambda_f = new_lambda_f * (friction_limit / wp.sqrt(lf_sq))
    else:
        new_lambda_n = wp.float32(0)
        new_lambda_f = wp.vec2f(wp.float32(0), wp.float32(0))
    return new_s, new_gamma, new_lambda_n, new_lambda_f


# ---------------------------------------------------------------------------
# Per-type dual update kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _update_dual_vv(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    dmin: wp.float32,
    mu_friction: wp.float32,
    decay_rate: wp.float32,
):
    """Dual update for vertex-vertex contacts."""
    k = wp.tid()  # type: ignore
    if k >= ogc.vv.prefix[n_u]:
        return
    u = ogc.vv.u[k]
    v = ogc.vv.v[k]
    gap = _gap_vv(x, meshes.V, u, v)  # type: ignore
    c_n, c_f = _project_gap(gap, ogc.vv_bases[k])  # type: ignore
    new_s, new_gamma, new_lambda_n, new_lambda_f = _apply_dual_update(  # type: ignore
        c_n, c_f,
        data.s[k], data.gamma[k], data.lambda_n[k], data.lambda_f[k],
        data.mu_n[0], data.mu_f[0],
        mu_friction, dmin, decay_rate,
    )
    data.s[k] = new_s  # type: ignore
    data.gamma[k] = new_gamma  # type: ignore
    data.lambda_n[k] = new_lambda_n  # type: ignore
    data.lambda_f[k] = new_lambda_f  # type: ignore


@wp.kernel
def _update_dual_ve(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    dmin: wp.float32,
    mu_friction: wp.float32,
    decay_rate: wp.float32,
):
    """Dual update for vertex-edge contacts."""
    k = wp.tid()  # type: ignore
    if k >= ogc.ve.prefix[n_u]:
        return
    u = ogc.ve.u[k]
    v = ogc.ve.v[k]
    t = ogc.ve_bary[k]
    gap = _gap_ve(x, meshes.V, meshes.F, t, u, v)  # type: ignore
    c_n, c_f = _project_gap(gap, ogc.ve_bases[k])  # type: ignore
    new_s, new_gamma, new_lambda_n, new_lambda_f = _apply_dual_update(  # type: ignore
        c_n, c_f,
        data.s[k], data.gamma[k], data.lambda_n[k], data.lambda_f[k],
        data.mu_n[0], data.mu_f[0],
        mu_friction, dmin, decay_rate,
    )
    data.s[k] = new_s  # type: ignore
    data.gamma[k] = new_gamma  # type: ignore
    data.lambda_n[k] = new_lambda_n  # type: ignore
    data.lambda_f[k] = new_lambda_f  # type: ignore


@wp.kernel
def _update_dual_vf(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    dmin: wp.float32,
    mu_friction: wp.float32,
    decay_rate: wp.float32,
):
    """Dual update for vertex-face contacts."""
    k = wp.tid()  # type: ignore
    if k >= ogc.vf.prefix[n_u]:
        return
    u = ogc.vf.u[k]
    v = ogc.vf.v[k]
    bary = ogc.vf_bary[k]
    gap = _gap_vf(x, meshes.V, meshes.F, bary, u, v)  # type: ignore
    c_n, c_f = _project_gap(gap, ogc.vf_bases[k])  # type: ignore
    new_s, new_gamma, new_lambda_n, new_lambda_f = _apply_dual_update(  # type: ignore
        c_n, c_f,
        data.s[k], data.gamma[k], data.lambda_n[k], data.lambda_f[k],
        data.mu_n[0], data.mu_f[0],
        mu_friction, dmin, decay_rate,
    )
    data.s[k] = new_s  # type: ignore
    data.gamma[k] = new_gamma  # type: ignore
    data.lambda_n[k] = new_lambda_n  # type: ignore
    data.lambda_f[k] = new_lambda_f  # type: ignore


@wp.kernel
def _update_dual_ee(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    dmin: wp.float32,
    mu_friction: wp.float32,
    decay_rate: wp.float32,
):
    """Dual update for edge-edge contacts."""
    k = wp.tid()  # type: ignore
    if k >= ogc.ee.prefix[n_u]:
        return
    u = ogc.ee.u[k]
    v = ogc.ee.v[k]
    bary = ogc.ee_bary[k]
    gap = _gap_ee(x, meshes.F, bary, u, v)  # type: ignore
    c_n, c_f = _project_gap(gap, ogc.ee_bases[k])  # type: ignore
    new_s, new_gamma, new_lambda_n, new_lambda_f = _apply_dual_update(  # type: ignore
        c_n, c_f,
        data.s[k], data.gamma[k], data.lambda_n[k], data.lambda_f[k],
        data.mu_n[0], data.mu_f[0],
        mu_friction, dmin, decay_rate,
    )
    data.s[k] = new_s  # type: ignore
    data.gamma[k] = new_gamma  # type: ignore
    data.lambda_n[k] = new_lambda_n  # type: ignore
    data.lambda_f[k] = new_lambda_f  # type: ignore


# ---------------------------------------------------------------------------
# MeshDynamicsData struct and MeshDynamics class
# ---------------------------------------------------------------------------


@wp.struct
class MeshDynamicsData:

    ogc: OgcData  # type: ignore
    cvv: ConstraintSetData  # type: ignore
    cve: ConstraintSetData  # type: ignore
    cvf: ConstraintSetData  # type: ignore
    cee: ConstraintSetData  # type: ignore


class MeshDynamics:

    _data: MeshDynamicsData  # type: ignore

    def __init__(self, ogc: Ogc):
        self.ogc = ogc
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.ogc.capacity
        n_verts, n_edges, n_half_edges, n_triangles = self.ogc.n_primitives
        self.cvv = ConstraintSet(n_verts, vv_capacity)
        self.cve = ConstraintSet(n_verts, ve_capacity)
        self.cvf = ConstraintSet(n_verts, vf_capacity)
        self.cee = ConstraintSet(n_half_edges, ee_capacity)
        self._data = MeshDynamicsData()
        self._data.ogc = self.ogc.data
        self._data.cvv = self.cvv.data
        self._data.cve = self.cve.data
        self._data.cvf = self.cvf.data
        self._data.cee = self.cee.data

    def update_dual(
        self,
        x: wp.array,
        dmin: float = 5e-4,
        mu: float = 0.5,
        decay: float = 0.9,
    ):
        """Update dual variables (slack, decay, Lagrange multipliers) for all contact types.

        GPU port of ``MeshDynamics::UpdateDual<Slack|Decay|LagrangeMultiplier>``.
        For each active contact k the kernel:
          1. Computes the gap vector between cached closest points using current positions ``x``.
          2. Projects the gap onto the cached contact basis (normal + 2 tangents).
          3. Runs the AL update: slack → decay → (lambda_n, lambda_f with cone projection).

        Unlike the C++ side, this implementation does not cache a Taylor expansion; it
        recomputes the linearized gap directly from the stored barycentric weights and basis.

        Args:
            x         : Current vertex positions (``wp.array[wp.vec3f]``, indexed by global
                        point index).
            dmin      : Minimum contact distance margin (separating threshold).
            mu        : Coulomb friction coefficient.
            decay     : Decay multiplier applied to ``gamma`` when a contact is separating
                        (i.e. slack is growing).  Should satisfy ``0 < decay < 1``.
        """
        meshes = self.ogc._meshes.data
        ogc = self.ogc.data
        stream = wp.get_stream()
        dmin_f = wp.float32(dmin)
        mu_f = wp.float32(mu)
        decay_f = wp.float32(decay)
        for kernel, cs in (
            (_update_dual_vv, self.cvv),
            (_update_dual_ve, self.cve),
            (_update_dual_vf, self.cvf),
            (_update_dual_ee, self.cee),
        ):
            wp.launch(
                kernel,
                dim=cs.capacity,
                inputs=[x, meshes, ogc, cs.data, wp.int32(cs.n_u), dmin_f, mu_f, decay_f],
                stream=stream,
            )

    @property
    def data(self) -> MeshDynamicsData:  # type: ignore
        return self._data
