import unittest

import numpy as np
import warp as wp

from . import halfedges
from . import pairs
from .mesh import pairs as mesh_pairs
from .multimesh import MultiMeshData
from .. import common
from ..common.buffer import DoubleBuffer

# friction_limit of a contact whose Coulomb limit isn't frozen yet in this time step (limits are >= 0)
FRICTION_LIMIT_UNSET = wp.constant(-1.0)


@wp.func
def frozen_friction_limit(
    friction_limit: wp.array[wp.float32],
    k: wp.int32,
    estimate: wp.float32,
):
    """Contact k's Coulomb limit for the rest of the time step: ``estimate`` (the limit at the current
    normal force) while no positive estimate is stored, then, from the contact's first strictly positive
    estimate in the step, which it stores in ``friction_limit[k]``, the stored value until
    :meth:`ConstraintSet.reset_friction_limits` (GPU Newton). A contact detected but not yet pressed
    (estimate 0) thus gets friction once it carries load, within the same step.

    Kept on purpose (maintainer, 2026-10-22: "for Newton, we best over-estimate the normal contact force"): the
    first estimate over-estimates the normal force at impacts (friction ratio 1.25 on the incline impact), but the
    alternatives under-estimate it while sliding. mu_f lambda_n lags the load the penalty carries, and a re-freeze at
    each subproblem reads the multiplier's dip between outer iterations (a 0.65-0.75 friction regime). Evidence and
    the alternatives' branches: python/tools/sim/docs/solver-accuracy-notes.md, entry 6."""
    C = friction_limit[k]
    if C < wp.float32(0):
        C = estimate
        if estimate > wp.float32(0):
            friction_limit[k] = C
    return C


@wp.func
def stored_friction_limit(friction_limit: wp.array[wp.float32], k: wp.int32):
    """Contact k's Coulomb limit for GPU Newton's Hessian and line-search energies, read-only: the frozen
    value, or 0 while it is unset. The augmented gradient at the step's base point runs first
    (:func:`frozen_friction_limit`) and stores every positive estimate, so an unset limit means its
    estimate there was 0, and all three use the same C. ``FRICTION_LIMIT_UNSET`` itself is not a limit:
    read as one, it gave the Hessian a friction scale -1/|f| (-inf at f = 0, NaN blocks, CG at its cap)
    and the energy a -|f| / sigma_f term."""
    return wp.max(friction_limit[k], wp.float32(0))


@wp.kernel
def _snapshot_and_reset_constraints(
    contacts: pairs.PairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.uint32,
    u: wp.array[wp.uint32],
    v: wp.array[wp.uint32],
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
    friction_limit: wp.array[wp.float32],
    u_prev: wp.array[wp.uint32],
    v_prev: wp.array[wp.uint32],
    s_prev: wp.array[wp.float32],
    gamma_prev: wp.array[wp.float32],
    lambda_n_prev: wp.array[wp.float32],
    lambda_f_prev: wp.array[wp.vec2f],
    friction_limit_prev: wp.array[wp.float32],
):
    """Copy slot k of the current arrays to the previous snapshot, then reset it to defaults.

    Slot k gets the key (u, v) of the new contact k, or the empty-slot sentinel u = n_u past the
    contacts, so that the next update can find it. gamma (1), the multipliers (0), the slack (0) and
    the frozen Coulomb limit (unset) are reset; the warm start then restores a persisting contact's
    state. The solvers refresh s from the positions before using it (VBD before every sweep, Newton
    before every step's gradient), but without the reset a new contact would inherit the slack of the
    slot's previous contact until then.
    """
    k = wp.tid()  # type: ignore
    u_prev[k] = u[k]
    v_prev[k] = v[k]
    s_prev[k] = s[k]
    gamma_prev[k] = gamma[k]
    lambda_n_prev[k] = lambda_n[k]
    lambda_f_prev[k] = lambda_f[k]
    friction_limit_prev[k] = friction_limit[k]
    n_contacts = contacts.prefix[n_u]
    if wp.uint64(k) < n_contacts:  # type: ignore
        u[k] = contacts.u[k]
        v[k] = contacts.v[k]
    else:
        u[k] = n_u
    s[k] = wp.float32(0)
    gamma[k] = wp.float32(1)
    lambda_n[k] = wp.float32(0)
    lambda_f[k] = wp.vec2f(wp.float32(0), wp.float32(0))
    friction_limit[k] = FRICTION_LIMIT_UNSET


@wp.kernel
def _warm_start_constraints(
    contacts: pairs.PairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.uint32,  # Number of u primitives (also the sentinel value for invalid slots)
    u_prev: wp.array[wp.uint32],
    v_prev: wp.array[wp.uint32],
    s_prev: wp.array[wp.float32],
    gamma_prev: wp.array[wp.float32],
    lambda_n_prev: wp.array[wp.float32],
    lambda_f_prev: wp.array[wp.vec2f],
    friction_limit_prev: wp.array[wp.float32],
    u: wp.array[wp.uint32],
    v: wp.array[wp.uint32],
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
    friction_limit: wp.array[wp.float32],
):
    """Restore persisting AL state from the previous snapshot into the current arrays.

    Thread k reads previous slot k. If it holds a valid contact pair (u, v) >= 0
    that can be located in the new forward contact list, the AL state
    (s, gamma, lambda_n, lambda_f, friction_limit) is copied to its new position.
    """
    k = wp.tid()  # type: ignore
    pu = u_prev[k]
    pv = v_prev[k]
    if pu >= n_u:  # sentinel: n_u means invalid slot
        return
    n_contacts = contacts.prefix[n_u]
    l = common.lower_bound(contacts.u, contacts.v, n_contacts, pu, pv)  # type: ignore
    if l >= n_contacts or contacts.u[l] != pu or contacts.v[l] != pv:
        return
    u[l] = pu  # type: ignore
    v[l] = pv  # type: ignore
    s[l] = s_prev[k]  # type: ignore
    gamma[l] = gamma_prev[k]  # type: ignore
    lambda_n[l] = lambda_n_prev[k]  # type: ignore
    lambda_f[l] = lambda_f_prev[k]  # type: ignore
    friction_limit[l] = friction_limit_prev[k]  # type: ignore


# Vertex contact kinds, by the dimension of the partner feature (feature transitions)
KIND_VV = wp.constant(0)
KIND_VE = wp.constant(1)
KIND_VF = wp.constant(2)


@wp.struct
class FeatureTransitionData:
    """One vertex contact set's current and previous keys and AL state, and the tangent bases of its previous
    contacts, for :func:`carry_across_features`."""

    u: wp.array[wp.uint32]
    v: wp.array[wp.uint32]
    s: wp.array[wp.float32]
    gamma: wp.array[wp.float32]
    lambda_n: wp.array[wp.float32]
    lambda_f: wp.array[wp.vec2f]
    t: wp.array[wp.vec3f]  # the current contacts' tangent bases
    b: wp.array[wp.vec3f]
    u_prev: wp.array[wp.uint32]
    v_prev: wp.array[wp.uint32]
    s_prev: wp.array[wp.float32]
    gamma_prev: wp.array[wp.float32]
    lambda_n_prev: wp.array[wp.float32]
    lambda_f_prev: wp.array[wp.vec2f]
    t_prev: wp.array[wp.vec3f]  # the previous contacts' tangent bases, in which lambda_f_prev is expressed
    b_prev: wp.array[wp.vec3f]


@wp.func
def _feature_nodes(meshes: MultiMeshData, kind: wp.int32, v: wp.uint32) -> wp.vec3i:  # type: ignore
    """The points of a contact's partner feature: a vertex (VV), a half-edge's endpoints (VE) or a face's corners
    (VF), padded with -1."""
    if kind == KIND_VV:
        return wp.vec3i(meshes.V[wp.int32(v)], -1, -1)
    if kind == KIND_VE:
        he = wp.int32(v)
        return wp.vec3i(halfedges.incoming_vertex(meshes.F, he), halfedges.outgoing_vertex(meshes.F, he), -1)
    return meshes.F[wp.int32(v)]


@wp.func
def are_adjacent_features(
    meshes: MultiMeshData,  # type: ignore
    ka: wp.int32,
    va: wp.uint32,
    kb: wp.int32,
    vb: wp.uint32,
) -> bool:
    """Whether partner features a and b (kinds ``KIND_*``) share a mesh vertex, i.e. b is in a's closed one-ring:
    a face, its edges and corners and the faces and edges around them; an edge, its endpoints and the edges and
    faces at them; a vertex, its edges and faces."""
    a, b = _feature_nodes(meshes, ka, va), _feature_nodes(meshes, kb, vb)
    for i in range(3):
        if a[i] >= 0 and (a[i] == b[0] or a[i] == b[1] or a[i] == b[2]):
            return True
    return False


@wp.func
def _has_key(u: wp.array[wp.uint32], v: wp.array[wp.uint32], ku: wp.uint32, kv: wp.uint32) -> bool:
    """Whether the sorted key arrays (u, v) (sentinel u = n_u past the contacts) hold the key (ku, kv)."""
    n = wp.uint32(u.shape[0])
    l = common.lower_bound(u, v, n, ku, kv)  # type: ignore
    return l < n and u[l] == ku and v[l] == kv


@wp.func
def _best_previous(
    meshes: MultiMeshData,  # type: ignore
    S: FeatureTransitionData,  # type: ignore
    kind: wp.int32,
    kind_new: wp.int32,
    v_new: wp.uint32,
    u: wp.uint32,
    best_kind: wp.int32,
    best_slot: wp.int32,
    best_lambda: wp.float32,
):
    """Vertex u's previous contacts of kind ``kind`` (set S) that a new contact (kind_new, v_new) may inherit from:
    adjacent to its feature and not found again by their exact key. Keeps the one with the largest lambda_n
    (strictly larger replaces, so earlier sets and slots win ties)."""
    n = wp.uint32(S.u_prev.shape[0])
    i = common.lower_bound(S.u_prev, n, u)  # type: ignore
    while i < n and S.u_prev[i] == u:
        vp = S.v_prev[i]
        if (
            S.lambda_n_prev[i] > best_lambda
            and are_adjacent_features(meshes, kind_new, v_new, kind, vp)
            and not _has_key(S.u, S.v, u, vp)
        ):
            best_kind, best_slot, best_lambda = kind, wp.int32(i), wp.float32(S.lambda_n_prev[i])
        i += wp.uint32(1)
    return best_kind, best_slot, best_lambda


@wp.func
def _first_new(
    meshes: MultiMeshData,  # type: ignore
    S: FeatureTransitionData,  # type: ignore
    kind: wp.int32,
    kind_prev: wp.int32,
    v_prev: wp.uint32,
    u: wp.uint32,
    first_kind: wp.int32,
    first_slot: wp.int32,
):
    """The first of vertex u's new contacts of kind ``kind`` (set S), in slot order, that may inherit from a
    previous contact (kind_prev, v_prev): adjacent to its feature and without an exact previous key. Keeps an earlier
    set's choice."""
    if first_kind >= 0:
        return first_kind, first_slot
    n = wp.uint32(S.u.shape[0])
    j = common.lower_bound(S.u, n, u)  # type: ignore
    while j < n and S.u[j] == u:
        vj = S.v[j]
        if are_adjacent_features(meshes, kind, vj, kind_prev, v_prev) and not _has_key(S.u_prev, S.v_prev, u, vj):
            return kind, wp.int32(j)
        j += wp.uint32(1)
    return first_kind, first_slot


@wp.func
def _inherit(me: FeatureTransitionData, k: wp.int32, src: FeatureTransitionData, i: wp.int32):  # type: ignore
    """New contact k of ``me`` takes previous contact i of ``src``'s state; lambda_f goes through 3D from the
    previous tangent basis into the new one."""
    me.s[k] = src.s_prev[i]
    me.gamma[k] = src.gamma_prev[i]
    me.lambda_n[k] = src.lambda_n_prev[i]
    lf = src.lambda_f_prev[i]
    f = lf[0] * src.t_prev[i] + lf[1] * src.b_prev[i]
    me.lambda_f[k] = wp.vec2f(wp.dot(f, me.t[k]), wp.dot(f, me.b[k]))


@wp.kernel
def _carry_across_features(
    meshes: MultiMeshData,  # type: ignore
    kind: wp.int32,
    me: FeatureTransitionData,  # type: ignore
    vv: FeatureTransitionData,  # type: ignore
    ve: FeatureTransitionData,  # type: ignore
    vf: FeatureTransitionData,  # type: ignore
    n_u: wp.uint32,
):
    """Thread k: new contact k of ``me`` (kind ``kind``, one of vv, ve, vf), if it has no exact previous key, takes
    the state of the previous contact p of the same vertex chosen by :func:`carry_across_features`."""
    k = wp.tid()
    u = me.u[k]
    if u >= n_u:
        return
    v = me.v[k]
    if _has_key(me.u_prev, me.v_prev, u, v):
        return
    # Its preferred previous contact: the largest lambda_n, ties to VF, VE, VV, then the lowest slot
    pk, pi, plam = wp.int32(-1), wp.int32(-1), wp.float32(-1)
    pk, pi, plam = _best_previous(meshes, vf, KIND_VF, kind, v, u, pk, pi, plam)
    pk, pi, plam = _best_previous(meshes, ve, KIND_VE, kind, v, u, pk, pi, plam)
    pk, pi, plam = _best_previous(meshes, vv, KIND_VV, kind, v, u, pk, pi, plam)
    if pk < 0:
        return
    pv = wp.uint32(0)
    if pk == KIND_VF:
        pv = vf.v_prev[pi]
    elif pk == KIND_VE:
        pv = ve.v_prev[pi]
    else:
        pv = vv.v_prev[pi]
    # p's preferred new contact: VF, VE, VV, then the lowest slot. Only a mutual choice inherits, so no previous
    # contact's state goes to two new contacts
    nk, nj = wp.int32(-1), wp.int32(-1)
    nk, nj = _first_new(meshes, vf, KIND_VF, pk, pv, u, nk, nj)
    nk, nj = _first_new(meshes, ve, KIND_VE, pk, pv, u, nk, nj)
    nk, nj = _first_new(meshes, vv, KIND_VV, pk, pv, u, nk, nj)
    if nk != kind or nj != k:
        return
    if pk == KIND_VF:
        _inherit(me, k, vf, pi)
    elif pk == KIND_VE:
        _inherit(me, k, ve, pi)
    else:
        _inherit(me, k, vv, pi)


@wp.struct
class ConstraintSetData:
    """Struct of arrays for GPU contact constraints, used for kernel inputs/outputs.

    See :class:`ConstraintSet` for the corresponding CPU-side storage class.
    """

    s: wp.array[wp.float32]
    gamma: wp.array[wp.float32]
    lambda_n: wp.array[wp.float32]
    lambda_f: wp.array[wp.vec2f]
    friction_limit: wp.array[wp.float32]


class ConstraintSet:
    """Storage class for augmented-Lagrangian contact constraints.

    Each field is a :class:`~common.buffer.DoubleBuffer` of ``capacity``-sized arrays:

    * ``buf.current``   — active state for the current step.
    * ``buf.alternate`` — snapshot of the prior step used for warm-starting.

    :meth:`update_constraint_set` snapshots current into alternate with a copy
    kernel (not a pointer swap), so the pipeline is compatible with CUDA graph
    capture.

    Fields
    ------
    u, v      : uint32  contact pair keys (n_u = empty slot sentinel).
    s         : float32 inequality slack variable.
    gamma     : float32 constraint decay factor (default 1).
    lambda_n  : float32 normal Lagrange multiplier.
    lambda_f  : vec2f   friction Lagrange multiplier (tangential plane).
    friction_limit : float32 GPU Newton's Coulomb limit of the primal friction force, frozen for the
                time step at the contact's first positive estimate in it (:func:`frozen_friction_limit`;
                ``FRICTION_LIMIT_UNSET`` before it, which the Hessian and energies read as 0 through
                :func:`stored_friction_limit`). It is carried across re-detection with the rest of
                the state, so a contact found again keeps its limit, while a contact dropped by one
                detection and found by a later one starts over, like its multipliers. Newton resets it
                at the start of each time step (:meth:`reset_friction_limits`); the VBD family doesn't
                use it. Feature transitions (:func:`carry_across_features`) don't carry it.
    tangent, bitangent : vec3f (single arrays) the tangent basis of each contact of the last update, in
                which the dual updates express lambda_f (:meth:`store_tangent_bases`), for the next
                update's feature transitions.

    Parameters
    ----------
    n_u : int
        Number of u primitives (vertices, half-edges, or faces) in the mesh.
        Also used as the sentinel value for empty slots in u and v arrays.
    capacity : int
        Maximum number of contacts.
    """

    capacity: int
    n_u: int
    u: DoubleBuffer
    v: DoubleBuffer
    s: DoubleBuffer
    gamma: DoubleBuffer
    lambda_n: DoubleBuffer
    lambda_f: DoubleBuffer
    friction_limit: DoubleBuffer
    _data: ConstraintSetData  # type: ignore

    def __init__(self, n_u: int, capacity: int):
        self.n_u = n_u
        self.capacity = capacity

        def _make(value, dtype):
            return DoubleBuffer(
                wp.full(shape=(capacity,), value=value, dtype=dtype),
                wp.full(shape=(capacity,), value=value, dtype=dtype),
            )

        self.u = _make(n_u, wp.uint32)
        self.v = _make(n_u, wp.uint32)
        self.s = _make(0.0, wp.float32)
        self.gamma = _make(1.0, wp.float32)
        self.lambda_n = _make(0.0, wp.float32)
        self.lambda_f = _make(0.0, wp.vec2f)
        self.friction_limit = _make(FRICTION_LIMIT_UNSET, wp.float32)
        self.tangent = wp.zeros(shape=(capacity,), dtype=wp.vec3f)
        self.bitangent = wp.zeros(shape=(capacity,), dtype=wp.vec3f)
        self._data = ConstraintSetData()
        self._data.s = self.s.current
        self._data.gamma = self.gamma.current
        self._data.lambda_n = self.lambda_n.current
        self._data.lambda_f = self.lambda_f.current
        self._data.friction_limit = self.friction_limit.current

    # TODO: ThingsFallingInBasket with the default parameters goes non-finite at about step 39 on dev with this warm
    # start (ve/vf pairs at capacity, about 100 s per step). Whether it predates the warm start is unknown; 45 steps
    # of ea968473 (before it) would settle it. A heavy scene: run it only with the maintainer.
    def update_constraint_set(self, contacts: pairs.Pairs):  # type: ignore
        """Prepare the current arrays for a new step, warm-starting from the previous snapshot.

        Two kernels, so the update adds only two nodes to a captured graph:

        1. **Snapshot and reset** (:func:`_snapshot_and_reset_constraints`): copy every field's
           current slot to alternate (the previous snapshot), then set each slot's key ``(u, v)``
           to its new contact's (``u = n_u`` past the contacts), gamma to 1, the multipliers
           and the slack to 0 and the frozen Coulomb limit to unset.
        2. **Warm start** (:func:`_warm_start_constraints`): for each valid previous pair whose
           ``(u, v)`` appears in the new forward contact list, copy ``s``, ``gamma``,
           ``lambda_n``, ``lambda_f``, ``friction_limit`` to the corresponding current slot. It writes arbitrary
           slots, so it must follow the whole reset.

        After this call, every contact's slot holds its key, so the next update can warm-start
        it, and persisting contacts hold their previous AL state.

        Args:
            contacts: Forward contact pairs produced by a collision detector.
        """
        fields = (self.u, self.v, self.s, self.gamma, self.lambda_n, self.lambda_f, self.friction_limit)
        # Step 1: snapshot current into alternate and reset current to defaults.
        wp.launch(
            _snapshot_and_reset_constraints,
            dim=self.capacity,
            inputs=[contacts, self.n_u]
            + [buf.current for buf in fields]
            + [buf.alternate for buf in fields],
        )

        # Step 2: warm-start from alternate (previous) into current.
        wp.launch(
            _warm_start_constraints,
            dim=self.capacity,
            inputs=[
                contacts,
                self.n_u,
                self.u.alternate,
                self.v.alternate,
                self.s.alternate,
                self.gamma.alternate,
                self.lambda_n.alternate,
                self.lambda_f.alternate,
                self.friction_limit.alternate,
                self.u.current,
                self.v.current,
                self.s.current,
                self.gamma.current,
                self.lambda_n.current,
                self.lambda_f.current,
                self.friction_limit.current,
            ],
        )

    def transition_data(self, bases: mesh_pairs.ContactBasesData) -> FeatureTransitionData:  # type: ignore
        """This set's arrays for :func:`carry_across_features`, with its contacts' bases ``bases``."""
        data = FeatureTransitionData()
        data.u, data.v = self.u.current, self.v.current
        data.s, data.gamma = self.s.current, self.gamma.current
        data.lambda_n, data.lambda_f = self.lambda_n.current, self.lambda_f.current
        data.t, data.b = bases.t, bases.b
        data.u_prev, data.v_prev = self.u.alternate, self.v.alternate
        data.s_prev, data.gamma_prev = self.s.alternate, self.gamma.alternate
        data.lambda_n_prev, data.lambda_f_prev = self.lambda_n.alternate, self.lambda_f.alternate
        data.t_prev, data.b_prev = self.tangent, self.bitangent
        return data

    def store_tangent_bases(self, bases: mesh_pairs.ContactBasesData):  # type: ignore
        """Keep the current contacts' tangent bases, in which the next dual updates express lambda_f, for the next
        :func:`carry_across_features` (the detection overwrites ``bases`` before it). Two capture-safe copies."""
        wp.copy(self.tangent, bases.t)
        wp.copy(self.bitangent, bases.b)

    def reset_friction_limits(self):
        """Unfreeze every contact's Coulomb limit (GPU Newton, at the start of each time step), so each
        contact's first positive estimate in the step freezes it again (:func:`frozen_friction_limit`). A
        capture-safe fill, no host synchronization."""
        self.friction_limit.current.fill_(FRICTION_LIMIT_UNSET)

    @property
    def data(self) -> ConstraintSetData:  # type: ignore
        """Struct of arrays for kernel inputs/outputs."""
        return self._data


def carry_across_features(
    meshes: MultiMeshData,  # type: ignore
    sets: tuple[FeatureTransitionData, FeatureTransitionData, FeatureTransitionData],  # type: ignore
    n_u: int,
):
    """Feature transitions: a vertex contact whose partner feature changed to an adjacent one keeps its AL state.

    Call after every vertex contact set's :meth:`ConstraintSet.update_constraint_set` (the exact-key warm start),
    then :meth:`ConstraintSet.store_tangent_bases`. ``sets`` are the VV, VE and VF sets'
    :meth:`ConstraintSet.transition_data`; ``n_u`` is their number of vertices (the empty-slot sentinel).

    A new contact (u, v') with no exact previous key inherits the slack, gamma and multipliers of a previous
    contact (u, v) of the same vertex, in any of the three sets, whose feature v shares a mesh vertex with v'
    (:func:`are_adjacent_features`: v' is in v's closed one-ring, so on the same mesh) and which no new contact has
    the exact key of. So a vertex sliding from a face onto its edge, a corner or a neighbouring face keeps its
    contact force, and the outer test doesn't see a pressed contact without one; a jump to a disjoint feature
    starts over. Sliding capsules on CapsulesIntoBowl change features at ~3% of their contacts per detection,
    2/3 of them to a neighbouring face, and ended ~2% of their contacts pressed without a multiplier (the outer
    test's residual floor, docs/solver-accuracy-notes.md section 4).

    Matching is one-to-one: the new contact's preferred previous contact is the one with the largest lambda_n
    (ties: VF, VE, VV, then the lowest slot), the previous contact's preferred new one is the first VF, VE, VV in
    slot order, and only mutual choices inherit, so multipliers are never summed or given twice. Vertex SDF gives a
    vertex at most one contact per partner mesh, so there each side has at most one candidate.

    lambda_n carries as is. lambda_f is re-expressed through 3D from the previous contact's tangent basis into the
    new one (its norm can only shrink, so it stays in the Coulomb cone). The slack is refreshed from the positions
    before it is used; gamma, the decay of a separating contact, describes the vertex's contact rather than its
    feature, so both carry, as for an exact key. GPU Newton's frozen Coulomb limit doesn't carry.

    A GPU-only difference from C++ ``DenseAdjacencySet::Assign``, which warm starts by exact key only. Capture
    safe: three kernel launches over the sets' capacities, no allocation.
    """
    # TODO: Edge-edge contacts (OGC) keep exact-key warm starts only. The same idea would match an EE contact to a
    # previous one whose edges share a vertex with its own on both sides. OGC's VV/VE/VF contacts already go through
    # this, untested; with several contacts per vertex and mesh, the mutual choice can leave a transition unmatched.
    # TODO: A geodesic transfer instead of the one-ring rule: a new contact inherits a previous contact's multipliers
    # scaled by a weight that decays with the geodesic distance between the old and new partner features and vanishes
    # at a set geodesic radius. Why: a vertex that slides past its one-ring in one detection (fast sliding, large
    # substeps) restarts at lambda = 0 today, so the outer test sees a pressed contact without a force (notes entry 4).
    # Maintainer's idea for future work (2026-10-26), out of scope for the current paper.
    vv, ve, vf = sets
    for kind, me in ((KIND_VV, vv), (KIND_VE, ve), (KIND_VF, vf)):
        wp.launch(
            _carry_across_features,
            dim=me.u.shape[0],
            inputs=[meshes, kind, me, vv, ve, vf, wp.uint32(n_u)],
        )


@wp.kernel
def _evaluate_friction_limits(
    cs: ConstraintSetData,  # pyright: ignore[reportGeneralTypeIssues]
    estimate: wp.array[wp.float32],
    n_contacts: wp.int32,
    limit: wp.array[wp.float32],
):
    k = wp.tid()
    if k < n_contacts:
        limit[k] = frozen_friction_limit(cs.friction_limit, k, estimate[k])


class TestFrozenFrictionLimit(unittest.TestCase):
    """GPU Newton's frozen Coulomb limits across re-detections and time steps (Warp's CPU device)."""

    n_u = 4
    capacity = 8

    def setUp(self):
        self._device = wp.ScopedDevice("cpu")
        self._device.__enter__()
        self.contacts = pairs.PairsData()
        self.cs = ConstraintSet(self.n_u, self.capacity)

    def tearDown(self):
        self._device.__exit__(None, None, None)

    def _detect(self, uv):
        """Set the forward contacts (sorted (u, v) pairs) and update the constraint set."""
        u = np.full(self.capacity, self.n_u, dtype=np.uint32)
        v = np.full(self.capacity, self.n_u, dtype=np.uint32)
        u[: len(uv)] = [p[0] for p in uv]
        v[: len(uv)] = [p[1] for p in uv]
        prefix = np.zeros(self.n_u + 1, dtype=np.uint64)
        np.add.at(prefix, u[: len(uv)].astype(np.int64) + 1, 1)
        self.contacts.prefix = wp.array(np.cumsum(prefix).astype(np.uint64), dtype=wp.uint64)
        self.contacts.u = wp.array(u, dtype=wp.uint32)
        self.contacts.v = wp.array(v, dtype=wp.uint32)
        self.cs.update_constraint_set(self.contacts)
        self.n_contacts = len(uv)

    def _evaluate(self, estimates):
        """The limits a gradient evaluation uses when the current normal forces give ``estimates``."""
        limit = wp.zeros(self.capacity, dtype=wp.float32)
        wp.launch(
            _evaluate_friction_limits,
            dim=self.capacity,
            inputs=[self.cs.data, wp.array(estimates, dtype=wp.float32), self.n_contacts, limit],
        )
        return limit.numpy()[: self.n_contacts].tolist()

    def test_frozen_for_the_time_step(self):
        # Time step 1: the first detection's contacts freeze at the first evaluation
        self._detect([(0, 1), (1, 2)])
        self.cs.reset_friction_limits()
        self.assertEqual(self._evaluate([1.0, 2.0]), [1.0, 2.0])
        self.assertEqual(self._evaluate([5.0, 6.0]), [1.0, 2.0])
        # Second outer iteration: (0, 3) appears between them, moving (1, 2) to another slot
        self._detect([(0, 1), (0, 3), (1, 2)])
        self.assertEqual(self._evaluate([7.0, 8.0, 9.0]), [1.0, 8.0, 2.0])
        self.assertEqual(self._evaluate([10.0, 11.0, 12.0]), [1.0, 8.0, 2.0])
        # A contact dropped by one detection and found by a later one starts over, like its multipliers
        self._detect([(0, 1), (0, 3)])
        self.assertEqual(self._evaluate([13.0, 14.0]), [1.0, 8.0])
        self._detect([(0, 1), (0, 3), (1, 2)])
        self.assertEqual(self._evaluate([15.0, 16.0, 17.0]), [1.0, 8.0, 17.0])
        # Time step 2: every limit is recomputed, persisting contacts included
        self._detect([(0, 1), (0, 3), (1, 2)])
        self.cs.reset_friction_limits()
        self.assertEqual(self._evaluate([0.5, 0.0, 3.0]), [0.5, 0.0, 3.0])
        # A contact not yet pressed (estimate 0) isn't frozen: it freezes at its first positive estimate
        self.assertEqual(self._evaluate([4.0, 0.0, 4.0]), [0.5, 0.0, 3.0])
        self.assertEqual(self._evaluate([4.0, 2.5, 4.0]), [0.5, 2.5, 3.0])
        self.assertEqual(self._evaluate([4.0, 6.0, 4.0]), [0.5, 2.5, 3.0])
        # ... also across a re-detection within the step
        self._detect([(0, 1), (0, 2), (0, 3), (1, 2)])
        self.assertEqual(self._evaluate([1.0, 0.0, 1.0, 1.0]), [0.5, 0.0, 2.5, 3.0])
        self._detect([(0, 1), (0, 2), (0, 3), (1, 2)])
        self.assertEqual(self._evaluate([1.0, 0.75, 1.0, 1.0]), [0.5, 0.75, 2.5, 3.0])
        self.assertEqual(self._evaluate([1.0, 9.0, 1.0, 1.0]), [0.5, 0.75, 2.5, 3.0])

    def test_new_contacts_start_at_zero_slack(self):
        # Persisting contacts keep their slack; a new contact in a slot that held another contact's
        # slack starts at 0
        self._detect([(0, 1), (1, 2)])
        self.cs.s.current.assign(np.array([0.3, 0.5] + [7.0] * (self.capacity - 2), dtype=np.float32))
        self._detect([(0, 1), (0, 3), (1, 2)])
        self.assertEqual(self.cs.s.current.numpy()[:3].tolist(), [np.float32(0.3), 0.0, 0.5])


class TestFeatureTransitions(unittest.TestCase):
    """Vertex contacts that change their partner feature to an adjacent one keep their AL state
    (:func:`carry_across_features`, Warp's CPU device).

    Partner faces f0 = (0, 1, 2), f1 = (2, 1, 3) (shares f0's edge 1-2), f2 = (2, 3, 4) (shares f0's corner 2) and
    f3 = (5, 6, 7) (disjoint); half-edge 3 is f1's edge 2 -> 1; vertex contacts of points 8 and 9 (vertex = point).
    """

    n_u = 10
    capacity = 8
    F = [(0, 1, 2), (2, 1, 3), (2, 3, 4), (5, 6, 7)]

    def setUp(self):
        self._device = wp.ScopedDevice("cpu")
        self._device.__enter__()
        self.meshes = MultiMeshData()
        self.meshes.V = wp.array(np.arange(self.n_u), dtype=wp.int32)
        self.meshes.F = wp.array(np.array(self.F, dtype=np.int32), dtype=wp.vec3i)
        self.sets = {kind: ConstraintSet(self.n_u, self.capacity) for kind in ("vv", "ve", "vf")}
        self.bases = {}
        for kind in self.sets:
            bases = mesh_pairs.ContactBasesData()
            bases.n = wp.array(np.tile([0.0, 0.0, 1.0], (self.capacity, 1)), dtype=wp.vec3f)
            bases.t = wp.array(np.tile([1.0, 0.0, 0.0], (self.capacity, 1)), dtype=wp.vec3f)
            bases.b = wp.array(np.tile([0.0, 1.0, 0.0], (self.capacity, 1)), dtype=wp.vec3f)
            self.bases[kind] = bases
        self.transition = tuple(self.sets[k].transition_data(self.bases[k]) for k in ("vv", "ve", "vf"))

    def tearDown(self):
        self._device.__exit__(None, None, None)

    def _detect(self, vv=(), ve=(), vf=()):
        """Detect the (u, v) keys of each kind and update the sets, as MeshDynamics.update_constraint_set."""
        for kind, uv in (("vv", vv), ("ve", ve), ("vf", vf)):
            uv = sorted(uv)
            u = np.full(self.capacity, self.n_u, dtype=np.uint32)
            v = np.full(self.capacity, self.n_u, dtype=np.uint32)
            u[: len(uv)] = [p[0] for p in uv]
            v[: len(uv)] = [p[1] for p in uv]
            prefix = np.zeros(self.n_u + 1, dtype=np.uint64)
            np.add.at(prefix, u[: len(uv)].astype(np.int64) + 1, 1)
            contacts = pairs.PairsData()
            contacts.prefix = wp.array(np.cumsum(prefix).astype(np.uint64), dtype=wp.uint64)
            contacts.u = wp.array(u, dtype=wp.uint32)
            contacts.v = wp.array(v, dtype=wp.uint32)
            self.sets[kind].update_constraint_set(contacts)
        carry_across_features(self.meshes, self.transition, self.n_u)
        for kind, cs in self.sets.items():
            cs.store_tangent_bases(self.bases[kind])

    def _set(self, kind, slot, lambda_n, lambda_f=(0.0, 0.0), s=0.0, gamma=1.0):
        cs = self.sets[kind]
        for buf, value in ((cs.lambda_n, lambda_n), (cs.lambda_f, lambda_f), (cs.s, s), (cs.gamma, gamma)):
            a = buf.current.numpy()
            a[slot] = value
            buf.current.assign(a)

    def _lambda(self, kind):
        return self.sets[kind].lambda_n.current.numpy().tolist()

    def test_adjacent_transitions(self):
        # Face -> its edge, with slack and gamma
        self._detect(vf=[(8, 0)])
        self._set("vf", 0, 2.0, (0.5, 0.25), s=0.1, gamma=0.5)
        self._detect(ve=[(8, 3)])
        cs = self.sets["ve"]
        self.assertEqual(self._lambda("ve")[0], 2.0)
        self.assertEqual(cs.lambda_f.current.numpy()[0].tolist(), [0.5, 0.25])
        self.assertAlmostEqual(float(cs.s.current.numpy()[0]), 0.1)
        self.assertEqual(float(cs.gamma.current.numpy()[0]), 0.5)
        # Edge -> the other face at it; face -> the face across an edge, and across a corner
        self._detect(vf=[(8, 1)])
        self.assertEqual(self._lambda("vf")[0], 2.0)
        self._detect(vf=[(8, 0)])
        self.assertEqual(self._lambda("vf")[0], 2.0)
        self._detect(vf=[(8, 2)])
        self.assertEqual(self._lambda("vf")[0], 2.0)
        # Face -> its corner -> an edge at it (half-edge 4 is f1's edge 1 -> 3)
        self._detect(vv=[(8, 3)])
        self.assertEqual(self._lambda("vv")[0], 2.0)
        self._detect(ve=[(8, 4)])
        self.assertEqual(self._lambda("ve")[0], 2.0)

    def test_disjoint_and_other_vertices_start_over(self):
        self._detect(vf=[(8, 0)])
        self._set("vf", 0, 2.0)
        self._detect(vf=[(8, 3)])
        self.assertEqual(self._lambda("vf")[0], 0.0)
        self._set("vf", 0, 2.0)
        self._detect(vf=[(9, 2)])
        self.assertEqual(self._lambda("vf")[0], 0.0)

    def test_one_to_one(self):
        # One previous contact, two adjacent new ones: only the first (VF, VE, VV) inherits
        self._detect(vf=[(8, 0)])
        self._set("vf", 0, 2.0)
        self._detect(vv=[(8, 1)], ve=[(8, 3)])
        self.assertEqual((self._lambda("ve")[0], self._lambda("vv")[0]), (2.0, 0.0))
        # Two previous contacts, one new one: the largest lambda_n, never their sum
        self._detect(vf=[(8, 0), (8, 1)])
        self._set("vf", 0, 1.0)
        self._set("vf", 1, 3.0)
        self._detect(ve=[(8, 3)])
        self.assertEqual(self._lambda("ve")[0], 3.0)

    def test_exact_keys_win(self):
        # The exact key keeps its own state, and a previous contact found again isn't given to another one
        self._detect(vf=[(8, 0)])
        self._set("vf", 0, 1.0)
        self._detect(ve=[(8, 3)], vf=[(8, 0)])
        self.assertEqual((self._lambda("vf")[0], self._lambda("ve")[0]), (1.0, 0.0))
        # A new face takes the state of the previous edge, not of the face found again
        self._set("ve", 0, 5.0)
        self._detect(vf=[(8, 0), (8, 1)])
        self.assertEqual(self._lambda("vf")[:2], [1.0, 5.0])

    def test_friction_goes_through_3d(self):
        self._detect(vf=[(8, 0)])
        self._set("vf", 0, 2.0, (1.0, 0.5))
        # The new contact's tangent basis is the previous one turned by 90 degrees about the normal
        self.bases["ve"].t.assign(np.tile([0.0, 1.0, 0.0], (self.capacity, 1)))
        self.bases["ve"].b.assign(np.tile([-1.0, 0.0, 0.0], (self.capacity, 1)))
        self._detect(ve=[(8, 3)])
        self.assertEqual(self.sets["ve"].lambda_f.current.numpy()[0].tolist(), [0.5, -1.0])


if __name__ == "__main__":
    wp.init()
    unittest.main()
