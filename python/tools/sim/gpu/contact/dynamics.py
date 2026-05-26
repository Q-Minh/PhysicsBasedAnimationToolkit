import enum

from ...common.fields import DocField

import warp as wp
import cupy as cp
import cuda.compute
import numpy as np

from .mesh import pairs
from .constraints import ConstraintSet, ConstraintSetData
from .multimesh import MultiMesh, MultiMeshData
from . import halfedges
from .. import common


@wp.struct
class MeshDynamicsData:
    meshes: MultiMeshData  # type: ignore
    contacts: pairs.ContactPairsData  # Forward contacts # type: ignore
    rcontacts: (
        pairs.ReverseContactPairsData
    )  # Reverse contacts, if available # type: ignore
    cvv: ConstraintSetData  # type: ignore
    cve: ConstraintSetData  # type: ignore
    cvf: ConstraintSetData  # type: ignore
    cee: ConstraintSetData  # type: ignore

    Qc: wp.array[
        wp.float32
    ]  # (cvv capacity + cve capacity + cvf capacity + cee capacity,) constraint Rayleigh quotients w.r.t. dynamics hessian

    gamma_n: wp.float32  # Multiplier of sigma_n
    gamma_f: wp.float32  # Multiplier of sigma_f
    sigma_n: wp.array[wp.float32]  # (1,) normal contact penalty parameter
    sigma_f: wp.array[wp.float32]  # (1,) friction contact penalty parameter
    dmin: wp.float32
    mu_f: wp.float32  # Friction coefficient
    decay: wp.float32  # Rate of decay for deactivating constraints


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
    decay_rate: wp.float32,
    request_slack_update: bool,
    request_decay_update: bool,
    request_lagrange_multiplier_update: bool,
):
    # lambda_n[k] and mu_n[0] are always needed to compute new_s, which drives all branches.
    lambda_n_k = lambda_n[k]
    assert sigma_n > wp.float32(0)
    new_s = wp.max(wp.float32(0), c_n - lambda_n_k / sigma_n)
    # --- Decay: read old s[k] before any write ---
    if request_decay_update:
        s_prev = s[k]
        gamma_k = gamma[k]
        new_gamma = gamma_k
        if new_s == wp.float32(0):
            new_gamma = wp.float32(1)
        elif new_s > s_prev:  # contact gap growing -> separating # type: ignore
            new_gamma = gamma_k * decay_rate
        gamma[k] = new_gamma  # type: ignore
    # --- Slack ---
    if request_slack_update:
        s[k] = new_s  # type: ignore
    # --- Lagrange multipliers ---
    if request_lagrange_multiplier_update:
        if new_s == wp.float32(0):
            lambda_f_k = lambda_f[k]
            new_lambda_n = lambda_n_k - sigma_n * c_n
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
    xt: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: pairs.ContactPairsData,  # Forward contacts # type: ignore
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.vv.prefix[n_u]:  # type: ignore
        return
    u = contacts.vv.u[c]
    v = contacts.vv.v[c]
    bases = contacts.vv_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    i, j = meshes.V[u], meshes.V[v]
    xi, xj = x[i], x[j]
    xti, xtj = xt[i], xt[j]
    c_n = wp.dot(xi - xj, n) - contact.dmin  # type: ignore
    du = (xi - xti) - (xj - xtj)
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        c,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_ve(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: pairs.ContactPairsData,  # Forward contacts # type: ignore
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.ve.prefix[n_u]:  # type: ignore
        return
    v = contacts.ve.u[c]
    he = contacts.ve.v[c]
    bases = contacts.ve_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    b1 = contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    i = meshes.V[v]
    j, k = halfedges.incoming_vertex(meshes.F, he), halfedges.outgoing_vertex(
        meshes.F, he
    )
    xi, xj, xk = x[i], x[j], x[k]
    xti, xtj, xtk = xt[i], xt[j], xt[k]
    xc = b0 * xj + b1 * xk
    xtc = b0 * xtj + b1 * xtk
    dx = xi - xtc
    du = (xi - xti) - (xc - xtc)
    c_n = wp.dot(dx, n) - contact.dmin  # type: ignore
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        c,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_vf(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: pairs.ContactPairsData,  # Forward contacts # type: ignore
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.vf.prefix[n_u]:  # type: ignore
        return
    v = contacts.vf.u[c]
    f = contacts.vf.v[c]
    bases = contacts.vf_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    bary = contacts.vf_bary[c]
    finds = meshes.F[f]
    i = meshes.V[v]
    j, k, l = finds[0], finds[1], finds[2]
    xi, xj, xk, xl = x[i], x[j], x[k], x[l]
    xti, xtj, xtk, xtl = xt[i], xt[j], xt[k], xt[l]
    b1, b2 = bary[0], bary[1]
    b0 = wp.float32(1) - b1 - b2
    xc = b0 * xj + b1 * xk + b2 * xl
    xtc = b0 * xtj + b1 * xtk + b2 * xtl
    dx = xi - xtc
    du = (xi - xti) - (xc - xtc)
    c_n = wp.dot(dx, n) - contact.dmin  # type: ignore
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))  # type: ignore
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        c,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _update_dual_ee(
    x: wp.array[wp.vec3f],
    xt: wp.array[wp.vec3f],
    meshes: MultiMeshData,  # pyright: ignore[reportGeneralTypeIssues]
    contacts: pairs.ContactPairsData,  # Forward contacts # type: ignore
    data: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,
    contact: MeshDynamicsData,  # pyright: ignore[reportGeneralTypeIssues]
    request_slack_update: bool = True,
    request_decay_update: bool = True,
    request_lagrange_multiplier_update: bool = True,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.ee.prefix[n_u]:  # type: ignore
        return
    he1 = contacts.ee.u[c]
    he2 = contacts.ee.v[c]
    bases = contacts.ee_bases
    n, t, b = bases.n[c], bases.t[c], bases.b[c]
    bary = contacts.ee_bary[c]
    i, j = halfedges.incoming_vertex(meshes.F, he1), halfedges.outgoing_vertex(
        meshes.F, he1
    )
    k, l = halfedges.incoming_vertex(meshes.F, he2), halfedges.outgoing_vertex(
        meshes.F, he2
    )
    xi, xj, xk, xl = x[i], x[j], x[k], x[l]
    xti, xtj, xtk, xtl = xt[i], xt[j], xt[k], xt[l]
    s0, t0 = wp.float32(1) - bary[0], bary[0]
    s1, t1 = wp.float32(1) - bary[1], bary[1]
    xc1 = s0 * xi + t0 * xj
    xtc1 = s0 * xti + t0 * xtj
    xc2 = s1 * xk + t1 * xl
    xtc2 = s1 * xtk + t1 * xtl
    dx = xc1 - xc2
    du = (xc1 - xtc1) - (xc2 - xtc2)
    c_n = wp.dot(dx, n) - contact.dmin
    c_f = wp.vec2f(wp.dot(du, t), wp.dot(du, b))
    sigma_n = contact.gamma_n * contact.sigma_n[0]
    sigma_f = contact.gamma_f * contact.sigma_f[0]
    _apply_dual_update(  # type: ignore
        c_n,
        c_f,
        c,  # type: ignore
        data.s,
        data.gamma,
        data.lambda_n,
        data.lambda_f,
        sigma_n,
        sigma_f,
        contact.mu_f,
        contact.decay,
        request_slack_update,
        request_decay_update,
        request_lagrange_multiplier_update,
    )


@wp.kernel
def _compute_rayleigh_vv(
    Hi: wp.array[wp.mat33f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
    n_u: wp.int32,
    Qc: wp.array[wp.float32],
    offset: wp.int32,
    Qcmin: wp.float32,
    penalty_adaptivity: wp.int32,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.vv.prefix[n_u]:  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_SUBPROBLEM:
            Qc[offset + c] = wp.float32(0)  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_TIMESTEP:
            Qc[offset + c] = Qcmin  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_CONSTANT:
            assert False, "No Rayleigh quotients when constant penalty adaptivity"
        return
    u = contacts.vv.u[c]
    v = contacts.vv.v[c]
    n = contacts.vv_bases.n[c]
    i, j = meshes.V[u], meshes.V[v]
    Qc[offset + c] = wp.float32(0.5) * (wp.dot(n, Hi[i] @ n) + wp.dot(n, Hi[j] @ n))  # type: ignore


@wp.kernel
def _compute_rayleigh_ve(
    Hi: wp.array[wp.mat33f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
    n_u: wp.int32,
    Qc: wp.array[wp.float32],
    offset: wp.int32,
    Qcmin: wp.float32,
    penalty_adaptivity: wp.int32,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.ve.prefix[n_u]:  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_SUBPROBLEM:
            Qc[offset + c] = wp.float32(0)  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_TIMESTEP:
            Qc[offset + c] = Qcmin  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_CONSTANT:
            assert False, "No Rayleigh quotients when constant penalty adaptivity"
        return
    v = contacts.ve.u[c]
    he = contacts.ve.v[c]
    n = contacts.ve_bases.n[c]
    b1 = contacts.ve_bary[c]
    b0 = wp.float32(1) - b1
    i = meshes.V[v]
    j = halfedges.incoming_vertex(meshes.F, he)  # type: ignore
    k = halfedges.outgoing_vertex(meshes.F, he)  # type: ignore
    Qc[offset + c] = wp.float32(0.5) * (  # type: ignore
        wp.dot(n, Hi[i] @ n)
        + b0 * b0 * wp.dot(n, Hi[j] @ n)
        + b1 * b1 * wp.dot(n, Hi[k] @ n)
    )


@wp.kernel
def _compute_rayleigh_vf(
    Hi: wp.array[wp.mat33f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
    n_u: wp.int32,
    Qc: wp.array[wp.float32],
    offset: wp.int32,
    Qcmin: wp.float32,
    penalty_adaptivity: wp.int32,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.vf.prefix[n_u]:  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_SUBPROBLEM:
            Qc[offset + c] = wp.float32(0)  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_TIMESTEP:
            Qc[offset + c] = Qcmin  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_CONSTANT:
            assert False, "No Rayleigh quotients when constant penalty adaptivity"
        return
    v = contacts.vf.u[c]
    f = contacts.vf.v[c]
    n = contacts.vf_bases.n[c]
    bary = contacts.vf_bary[c]
    finds = meshes.F[f]
    i = meshes.V[v]
    j, k, l = finds[0], finds[1], finds[2]
    b1 = bary[0]
    b2 = bary[1]
    b0 = wp.float32(1) - b1 - b2
    Qc[offset + c] = wp.float32(0.5) * (  # type: ignore
        wp.dot(n, Hi[i] @ n)
        + b0 * b0 * wp.dot(n, Hi[j] @ n)
        + b1 * b1 * wp.dot(n, Hi[k] @ n)
        + b2 * b2 * wp.dot(n, Hi[l] @ n)
    )


@wp.kernel
def _compute_rayleigh_ee(
    Hi: wp.array[wp.mat33f],
    meshes: MultiMeshData,  # type: ignore
    contacts: pairs.ContactPairsData,  # type: ignore
    n_u: wp.int32,
    Qc: wp.array[wp.float32],
    offset: wp.int32,
    Qcmin: wp.float32,
    penalty_adaptivity: wp.int32,
):
    c = wp.tid()  # type: ignore
    if wp.uint64(c) >= contacts.ee.prefix[n_u]:  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_SUBPROBLEM:
            Qc[offset + c] = wp.float32(0)  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_TIMESTEP:
            Qc[offset + c] = Qcmin  # type: ignore
        if penalty_adaptivity == PENALTY_ADAPTIVITY_CONSTANT:
            assert False, "No Rayleigh quotients when constant penalty adaptivity"
        return
    he1 = contacts.ee.u[c]
    he2 = contacts.ee.v[c]
    n = contacts.ee_bases.n[c]
    bary = contacts.ee_bary[c]
    i = halfedges.incoming_vertex(meshes.F, he1)  # type: ignore
    j = halfedges.outgoing_vertex(meshes.F, he1)  # type: ignore
    k = halfedges.incoming_vertex(meshes.F, he2)  # type: ignore
    l = halfedges.outgoing_vertex(meshes.F, he2)  # type: ignore
    s0 = wp.float32(1) - bary[0]
    t0 = bary[0]
    s1 = wp.float32(1) - bary[1]
    t1 = bary[1]
    Qc[offset + c] = wp.float32(0.5) * (  # type: ignore
        s0 * s0 * wp.dot(n, Hi[i] @ n)
        + t0 * t0 * wp.dot(n, Hi[j] @ n)
        + s1 * s1 * wp.dot(n, Hi[k] @ n)
        + t1 * t1 * wp.dot(n, Hi[l] @ n)
    )


class PenaltyAdaptivity(enum.Enum):
    CONSTANT = 0
    TIMESTEP = 1
    SUBPROBLEM = 2


PENALTY_ADAPTIVITY_CONSTANT = wp.constant(int(PenaltyAdaptivity.CONSTANT.value))
PENALTY_ADAPTIVITY_TIMESTEP = wp.constant(int(PenaltyAdaptivity.TIMESTEP.value))
PENALTY_ADAPTIVITY_SUBPROBLEM = wp.constant(int(PenaltyAdaptivity.SUBPROBLEM.value))


class Params:
    """Parameters for :class:`MeshDynamics`.

    Attributes:
        dmin        : Minimum separation distance used as the contact distance margin
                      (separating threshold in the AL slack update).
        mu_f        : Coulomb friction coefficient.
        decay       : Multiplicative decay applied to the contact-activity weight ``gamma``
                      when a contact is separating (slack growing). Must satisfy ``0 < decay < 1``.
        gamman      : Normal AL penalty scaling factor, equivalent to
                      ``MeshDynamics::Params::gamma`` in the C++ side.
        gammaf      : Friction AL penalty scaling factor, equivalent to
                      ``MeshDynamics::Params::gammaf`` in the C++ side.
        min_sigma_n : Floor for the adaptive normal penalty ``sigma_n``.
        penalty_adaptivity : Strategy for adapting the penalty parameters.
    """

    dmin = DocField(2e-3, "Minimum separation distance margin (contact threshold)")
    mu_f = DocField(0.2, "Coulomb friction coefficient")
    decay = DocField(0.5, "Decay rate for contact deactivation")
    gamman = DocField(1e4, "Normal contact AL penalty scaling factor")
    gammaf = DocField(1e3, "Friction contact AL penalty scaling factor")
    min_sigma_n = DocField(1.0, "Minimum normal penalty")
    penalty_adaptivity = DocField(
        PenaltyAdaptivity.CONSTANT, "Strategy for adapting penalty parameters"
    )


class MeshDynamics:

    params: Params
    meshes: MultiMesh
    contacts: pairs.ContactPairs
    cvv: ConstraintSet
    cve: ConstraintSet
    cvf: ConstraintSet
    cee: ConstraintSet
    _data: MeshDynamicsData  # type: ignore

    _streams: list[wp.Stream]

    def __init__(
        self, dt: float, contacts: pairs.ContactPairs, params: Params | None = None
    ):
        self.params = params if params is not None else Params()
        self.contacts = contacts
        self.meshes = self.contacts.meshes
        vv_capacity, ve_capacity, vf_capacity, ee_capacity = self.contacts.capacity
        n_verts, n_half_edges = self.meshes.n_verts, self.meshes.n_half_edges
        self.cvv = ConstraintSet(n_verts, vv_capacity)
        self.cve = ConstraintSet(n_verts, ve_capacity)
        self.cvf = ConstraintSet(n_verts, vf_capacity)
        self.cee = ConstraintSet(n_half_edges, ee_capacity)
        self._data = MeshDynamicsData()
        self._data.meshes = self.meshes.data
        self._data.contacts, self._data.rcontacts = self.contacts.read_data
        self._data.cvv = self.cvv.data
        self._data.cve = self.cve.data
        self._data.cvf = self.cvf.data
        self._data.cee = self.cee.data
        self._data.Qc = wp.zeros(
            vv_capacity + ve_capacity + vf_capacity + ee_capacity, dtype=wp.float32
        )
        self._data.gamma_n = self.params.gamman  # type: ignore
        self._data.gamma_f = self.params.gammaf  # type: ignore
        self._data.sigma_n = wp.array([dt * dt], dtype=wp.float32)
        self._data.sigma_f = wp.array([dt * dt], dtype=wp.float32)
        self._data.dmin = self.params.dmin
        self._data.mu_f = self.params.mu_f
        self._data.decay = self.params.decay
        self.sigma_n_min = self.params.min_sigma_n
        self._streams = [wp.Stream() for _ in range(4)]  # one stream per contact type
        n_points = self.meshes.data.GXV.shape[0]
        self.Hi = wp.zeros(n_points, dtype=wp.mat33f)
        self._normal_penalty_reduction = common.reduce.Reduce(
            d_in=cp.asarray(self._data.Qc),
            d_out=cp.asarray(self._data.sigma_n),
            num_items=self._data.Qc.shape[0],
            op=cuda.compute.OpKind.MAXIMUM,
        )

    def update_constraint_set(self):
        """Prepare constraint sets for a new step, warm-starting from the previous snapshot.

        Preconditions:
        - self.contacts must be up-to-date
        """
        contacts, _ = self.contacts.read_data
        main_stream = wp.get_stream()
        # Update all constraint sets
        for stream, cset, cuv in zip(
            self._streams[:4],
            (self.cvv, self.cve, self.cvf, self.cee),
            (
                contacts.vv,
                contacts.ve,
                contacts.vf,
                contacts.ee,
            ),
        ):
            stream.wait_stream(main_stream)
            with wp.ScopedStream(stream, sync_enter=False):
                cset.update_constraint_set(cuv)
            main_stream.wait_stream(stream)

    def update_dual(
        self,
        x: wp.array[wp.vec3f],
        xt: wp.array[wp.vec3f],
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
            xt        : Reference vertex positions at the start of the time step
                        (``wp.array[wp.vec3f]``), used to compute the relative displacement
                        for the friction constraint ``c_f``.
            request_slack_update: Whether to update the slack variable.
            request_decay_update: Whether to update the decay variable.
            request_lagrange_multiplier_update: Whether to update the Lagrange multipliers.
        """
        meshes = self.meshes.data
        contacts = self.contacts._data
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
                    xt,
                    meshes,
                    contacts,
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

    def adapt_penalty_parameters(self):
        if self.params.penalty_adaptivity == PenaltyAdaptivity.CONSTANT:
            return
        vv_cap, ve_cap, vf_cap, _ = self.contacts.capacity
        contacts = self.contacts._data
        meshes = self.meshes.data
        main_stream = wp.get_stream()
        # Fork
        for stream in self._streams[:4]:
            stream.wait_stream(main_stream)
        for kernel, cs, stream, offset in zip(
            [
                _compute_rayleigh_vv,
                _compute_rayleigh_ve,
                _compute_rayleigh_vf,
                _compute_rayleigh_ee,
            ],
            [self.cvv, self.cve, self.cvf, self.cee],
            self._streams[:4],
            [0, vv_cap, vv_cap + ve_cap, vv_cap + ve_cap + vf_cap],
        ):
            wp.launch(
                kernel,
                dim=cs.capacity,
                inputs=[
                    self.Hi,
                    meshes,
                    contacts,
                    cs.n_u,
                    self._data.Qc,
                    offset,
                    self.sigma_n_min / self._data.gamma_n,
                    int(self.params.penalty_adaptivity.value),  # type: ignore
                ],
                stream=stream,
            )
        # Join
        for stream in self._streams[:4]:
            main_stream.wait_stream(stream)
        self._normal_penalty_reduction(main_stream)

    @property
    def data(self) -> MeshDynamicsData:  # type: ignore
        return self._data
