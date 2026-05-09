from ...common.fields import DocField

import warp as wp
import cupy as cp
import cuda.compute
import numpy as np

from .ogc import Ogc, OgcData
from .constraints import ConstraintSet, ConstraintSetData
from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from .. import common


@wp.struct
class MeshDynamicsData:
    ogc: OgcData  # type: ignore
    meshes: MultiMeshData  # type: ignore
    cvv: ConstraintSetData  # type: ignore
    cve: ConstraintSetData  # type: ignore
    cvf: ConstraintSetData  # type: ignore
    cee: ConstraintSetData  # type: ignore
    gamma_n: wp.float32  # Multiplier of sigma_n
    gamma_f: wp.float32  # Multiplier of sigma_f
    sigma_n: wp.array[wp.float32]  # (1,) normal contact penalty parameter
    sigma_f: wp.array[wp.float32]  # (1,) friction contact penalty parameter
    dmin: wp.float32
    mu_f: wp.float32  # Friction coefficient
    decay: wp.float32  # Rate of decay for deactivating constraints


@wp.func
def _gap_vv(
    x: wp.array[wp.vec3f],
    V: wp.array[wp.int32],
    u: wp.int32,
    v: wp.int32,
) -> wp.vec3f:
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
    xi = x[V[u]]  # type: ignore
    finds = F[v]
    xj, xk, xl = x[finds[0]], x[finds[1]], x[finds[2]]  # type: ignore
    b0, b1 = bary[0], bary[1]  # type: ignore
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
    s, t = bary[0], bary[1]  # type: ignore
    i_u = halfedges.incoming_vertex(F, u)
    j_u = halfedges.outgoing_vertex(F, u)
    i_v = halfedges.incoming_vertex(F, v)
    j_v = halfedges.outgoing_vertex(F, v)
    xcp_u = (wp.float32(1) - s) * x[i_u] + s * x[j_u]  # type: ignore
    xcp_v = (wp.float32(1) - t) * x[i_v] + t * x[j_v]  # type: ignore
    return xcp_u - xcp_v


@wp.func
def _project_gap(
    gap: wp.vec3f,
    basis: wp.mat33f,
) -> wp.vec3f:  # type: ignore
    c = basis @ gap  # type: ignore
    return c


@wp.func
def _apply_dual_update(
    c_n: wp.float32,
    c_f: wp.vec2f,
    k: wp.int32,
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
    sigma_n: wp.float32,
    sigma_f: wp.float32,
    mu_friction: wp.float32,
    dmin: wp.float32,
    decay_rate: wp.float32,
    request_slack_update: bool,
    request_decay_update: bool,
    request_lagrange_multiplier_update: bool,
):
    # lambda_n[k] and mu_n[0] are always needed to compute new_s, which drives all branches.
    lambda_n_k = lambda_n[k]
    new_s = wp.max(wp.float32(0), c_n - dmin - lambda_n_k / sigma_n)
    # --- Decay: read old s[k] before any write ---
    if request_decay_update:
        s_prev = s[k]
        gamma_k = gamma[k]
        new_gamma = gamma_k
        if new_s == wp.float32(0):
            new_gamma = wp.float32(1)
        elif new_s > s_prev:  # contact gap growing -> separating
            new_gamma = gamma_k * decay_rate
        gamma[k] = new_gamma  # type: ignore
    # --- Slack ---
    if request_slack_update:
        s[k] = new_s  # type: ignore
    # --- Lagrange multipliers ---
    if request_lagrange_multiplier_update:
        if new_s == wp.float32(0):
            lambda_f_k = lambda_f[k]
            new_lambda_n = lambda_n_k - sigma_n * (c_n - dmin)
            new_lambda_f = lambda_f_k - sigma_f * c_f
            # Coulomb friction cone: |lambda_f| <= mu_friction * lambda_n
            friction_limit = mu_friction * new_lambda_n
            lf_sq = wp.dot(new_lambda_f, new_lambda_f)  # type: ignore
            if lf_sq > friction_limit * friction_limit:  # type: ignore
                new_lambda_f = new_lambda_f * (friction_limit / wp.sqrt(lf_sq))
            lambda_n[k] = new_lambda_n  # type: ignore
            lambda_f[k] = new_lambda_f  # type: ignore
        else:
            lambda_n[k] = wp.float32(0)  # type: ignore
            lambda_f[k] = wp.vec2f(wp.float32(0), wp.float32(0))  # type: ignore


@wp.kernel
def _update_dual_vv(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    k = wp.tid()  # type: ignore
    if k >= ogc.vv.prefix[n_u]:
        return
    u = ogc.vv.u[k]
    v = ogc.vv.v[k]
    gap = _gap_vv(x, meshes.V, u, v)  # type: ignore
    c = _project_gap(gap, ogc.vv_bases[k])  # type: ignore
    c_n, c_f = c[0], wp.vec2f(c[1], c[2])  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        k,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.dmin,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_ve(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    k = wp.tid()  # type: ignore
    if k >= ogc.ve.prefix[n_u]:
        return
    u = ogc.ve.u[k]
    v = ogc.ve.v[k]
    t = ogc.ve_bary[k]
    gap = _gap_ve(x, meshes.V, meshes.F, t, u, v)  # type: ignore
    c = _project_gap(gap, ogc.ve_bases[k])  # type: ignore
    c_n, c_f = c[0], wp.vec2f(c[1], c[2])  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        k,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.dmin,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_vf(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    k = wp.tid()  # type: ignore
    if k >= ogc.vf.prefix[n_u]:
        return
    u = ogc.vf.u[k]
    v = ogc.vf.v[k]
    bary = ogc.vf_bary[k]
    gap = _gap_vf(x, meshes.V, meshes.F, bary, u, v)  # type: ignore
    c = _project_gap(gap, ogc.vf_bases[k])  # type: ignore
    c_n, c_f = c[0], wp.vec2f(c[1], c[2])  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        k,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.dmin,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_ee(
    x: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    ogc: OgcData,  # pyright: ignore[reportGeneralTypeIssues]
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    k = wp.tid()  # type: ignore
    if k >= ogc.ee.prefix[n_u]:
        return
    u = ogc.ee.u[k]
    v = ogc.ee.v[k]
    bary = ogc.ee_bary[k]
    gap = _gap_ee(x, meshes.F, bary, u, v)  # type: ignore
    c = _project_gap(gap, ogc.ee_bases[k])  # type: ignore
    c_n, c_f = c[0], wp.vec2f(c[1], c[2])  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        k,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.dmin,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


class Params:
    """Parameters for :class:`MeshDynamics`.

    Attributes:
        dmin   : Minimum separation distance used as the contact distance margin
                 (separating threshold in the AL slack update).
        mu_f   : Coulomb friction coefficient.
        decay  : Multiplicative decay applied to the contact-activity weight ``gamma``
                 when a contact is separating (slack growing). Must satisfy ``0 < decay < 1``.
        gamman : Normal AL penalty scaling factor, equivalent to
                 ``MeshDynamics::Params::gamma`` in the C++ side.
        gammaf : Friction AL penalty scaling factor, equivalent to
                 ``MeshDynamics::Params::gammaf`` in the C++ side.
    """

    dmin = DocField(2e-3, "Minimum separation distance margin (contact threshold)")
    mu_f = DocField(0.2, "Coulomb friction coefficient")
    decay = DocField(0.5, "Decay rate for contact deactivation")
    gamman = DocField(5.0, "Normal contact AL penalty scaling factor")
    gammaf = DocField(0.1, "Friction contact AL penalty scaling factor")


class MeshDynamics:

    params: Params
    meshes: MultiMesh
    ogc: Ogc
    cvv: ConstraintSet
    cve: ConstraintSet
    cvf: ConstraintSet
    cee: ConstraintSet
    _data: MeshDynamicsData  # type: ignore

    _streams: list[wp.Stream]

    def __init__(self, ogc: Ogc, params: Params | None = None):
        self.params = params if params is not None else Params()
        self.ogc = ogc
        self.meshes = self.ogc.meshes
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.ogc.capacity
        n_verts, n_edges, n_half_edges, n_triangles = self.ogc.n_primitives
        self.cvv = ConstraintSet(n_verts, vv_capacity)
        self.cve = ConstraintSet(n_verts, ve_capacity)
        self.cvf = ConstraintSet(n_verts, vf_capacity)
        self.cee = ConstraintSet(n_half_edges, ee_capacity)
        self._data = MeshDynamicsData()
        self._data.meshes = self.meshes.data
        self._data.ogc = self.ogc.data
        self._data.cvv = self.cvv.data
        self._data.cve = self.cve.data
        self._data.cvf = self.cvf.data
        self._data.cee = self.cee.data
        self._data.gamma_n = self.params.gamman
        self._data.gamma_f = self.params.gammaf
        self._data.sigma_n = wp.array([self.params.gamman], dtype=wp.float32)
        self._data.sigma_f = wp.array([self.params.gammaf], dtype=wp.float32)
        self._data.dmin = self.params.dmin  # type: ignore
        self._data.mu_f = self.params.mu_f
        self._data.decay = self.params.decay
        self._streams = [wp.Stream() for _ in range(4)]  # one stream per contact type

    def update_constraint_set(self, xk: wp.array[wp.vec3f]):
        """Prepare constraint sets for a new step, warm-starting from the previous snapshot."""
        self.ogc.prepare_for_execution(xk)
        self.ogc.detect_contacts()
        self.ogc.update_displacement_bounds()
        ogc_data = self.ogc.data
        main_stream = wp.get_stream()
        # Update all constraint sets
        for stream, cset, cuv in zip(
            self._streams[:4],
            (self.cvv, self.cve, self.cvf, self.cee),
            (
                ogc_data.vv,
                ogc_data.ve,
                ogc_data.vf,
                ogc_data.ee,
            ),
        ):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False):
                cset.update_constraint_set(cuv)
            main_stream.wait_stream(stream)

    def update_dual(
        self,
        x: wp.array,
        request_slack_update: bool = True,
        request_decay_update: bool = True,
        request_lagrange_multiplier_update: bool = True,
    ):
        """Update dual variables (slack, decay, Lagrange multipliers) for all contact types.

        GPU port of ``MeshDynamics::UpdateDual<Slack|Decay|LagrangeMultiplier>``.
        For each active contact k the kernel:
          1. Computes the gap vector between cached closest points using current positions ``x``.
          2. Projects the gap onto the cached contact basis (normal + 2 tangents).
          3. Runs the AL update: slack -> decay -> (lambda_n, lambda_f with cone projection).

        Unlike the C++ side, this implementation does not cache a Taylor expansion; it
        recomputes the linearized gap directly from the stored barycentric weights and basis.

        Args:
            x         : Current vertex positions (``wp.array[wp.vec3f]``, indexed by global
                        point index).
            dmin      : Minimum contact distance margin (separating threshold).
            mu_f      : Coulomb friction coefficient.
            decay     : Decay multiplier applied to ``gamma`` when a contact is separating
                        (i.e. slack is growing).  Should satisfy ``0 < decay < 1``.
            request_slack_update: Whether to update the slack variable.
            request_decay_update: Whether to update the decay variable.
            request_lagrange_multiplier_update: Whether to update the Lagrange multipliers.
        """
        meshes = self.ogc._meshes.data
        ogc = self.ogc.data
        main_stream = wp.get_stream()
        # Fork all side streams into the current capture context before launching on them.
        for stream in self._streams[:4]:
            stream.wait_stream(main_stream)
        for kernel, cs, stream in zip(
            [_update_dual_vv, _update_dual_ve, _update_dual_vf, _update_dual_ee],
            [self.cvv, self.cve, self.cvf, self.cee],
            self._streams[:4],
        ):
            wp.launch(
                kernel,
                dim=cs.capacity,
                inputs=[
                    x,
                    meshes,
                    ogc,
                    cs.data,
                    cs.n_u,
                    self._data,
                    request_slack_update,
                    request_decay_update,
                    request_lagrange_multiplier_update,
                ],
                stream=stream,
            )
        for stream in self._streams[:4]:
            main_stream.wait_stream(stream)

    def enable_adaptive_penalty_parameters(
        self, maxQnv: wp.array[wp.float32], maxQfv: wp.array[wp.float32]
    ):
        n_verts = self.meshes.n_verts
        self._adaptive_penalty_d_in_n = cp.asarray(maxQnv)
        self._adaptive_penalty_d_in_f = cp.asarray(maxQfv)
        self._adaptive_penalty_d_out_n = cp.asarray(self._data.sigma_n)
        self._adaptive_penalty_d_out_f = cp.asarray(self._data.sigma_f)
        self._adaptive_penalty_h_init = np.zeros((1,), dtype=np.float32)
        self._adaptive_penalty_reduce_op = cuda.compute.OpKind.MAXIMUM
        self._adaptive_penalty_reduce_n = cuda.compute.make_reduce_into(
            d_in=self._adaptive_penalty_d_in_n,
            d_out=self._adaptive_penalty_d_out_n,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
        )
        self._adaptive_penalty_reduce_f = cuda.compute.make_reduce_into(
            d_in=self._adaptive_penalty_d_in_f,
            d_out=self._adaptive_penalty_d_out_f,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
        )
        temp_size_n = self._adaptive_penalty_reduce_n(
            temp_storage=None,
            d_in=self._adaptive_penalty_d_in_n,
            d_out=self._adaptive_penalty_d_out_n,
            num_items=n_verts,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
        )
        self._adaptive_penalty_reduce_storage = cp.empty((temp_size_n,), dtype=np.uint8)
        temp_size_f = self._adaptive_penalty_reduce_f(
            temp_storage=None,
            d_in=self._adaptive_penalty_d_in_f,
            d_out=self._adaptive_penalty_d_out_f,
            num_items=n_verts,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
        )
        self._adaptive_penalty_reduce_storage_f = cp.empty(
            (temp_size_f,), dtype=np.uint8
        )

    def adapt_penalty_parameters(self):
        main_stream = wp.get_stream()
        # Fork
        for stream in self._streams[:2]:
            stream.wait_stream(main_stream)
        n_verts = self.meshes.n_verts
        self._adaptive_penalty_reduce_n(
            temp_storage=self._adaptive_penalty_reduce_storage,
            d_in=self._adaptive_penalty_d_in_n,
            d_out=self._adaptive_penalty_d_out_n,
            num_items=n_verts,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
            stream=common.Stream(self._streams[0]),
        )
        self._adaptive_penalty_reduce_f(
            temp_storage=self._adaptive_penalty_reduce_storage_f,
            d_in=self._adaptive_penalty_d_in_f,
            d_out=self._adaptive_penalty_d_out_f,
            num_items=n_verts,
            op=self._adaptive_penalty_reduce_op,
            h_init=self._adaptive_penalty_h_init,
            stream=common.Stream(self._streams[1]),
        )
        # Join
        for stream in self._streams[:2]:
            main_stream.wait_stream(stream)

    def restore_feasibility(self, x: wp.array):
        self.ogc.truncate(x)

    @property
    def data(self) -> MeshDynamicsData:  # type: ignore
        return self._data
