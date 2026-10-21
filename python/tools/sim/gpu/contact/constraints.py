import unittest

import numpy as np
import warp as wp

from . import pairs
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
    (estimate 0) thus gets friction once it carries load, within the same step."""
    C = friction_limit[k]
    if C < wp.float32(0):
        C = estimate
        if estimate > wp.float32(0):
            friction_limit[k] = C
    return C


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
    contacts, so that the next update can find it. gamma (1), the multipliers (0) and the frozen
    Coulomb limit (unset) are reset, and s is recomputed by the solver's dual updates.
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
                ``FRICTION_LIMIT_UNSET`` before it). It is carried across re-detection with the rest of
                the state, so a contact found again keeps its limit, while a contact dropped by one
                detection and found by a later one starts over, like its multipliers. Newton resets it
                at the start of each time step (:meth:`reset_friction_limits`); the VBD family doesn't
                use it.

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
        self._data = ConstraintSetData()
        self._data.s = self.s.current
        self._data.gamma = self.gamma.current
        self._data.lambda_n = self.lambda_n.current
        self._data.lambda_f = self.lambda_f.current
        self._data.friction_limit = self.friction_limit.current

    def update_constraint_set(self, contacts: pairs.Pairs):  # type: ignore
        """Prepare the current arrays for a new step, warm-starting from the previous snapshot.

        Two kernels, so the update adds only two nodes to a captured graph:

        1. **Snapshot and reset** (:func:`_snapshot_and_reset_constraints`): copy every field's
           current slot to alternate (the previous snapshot), then set each slot's key ``(u, v)``
           to its new contact's (``u = n_u`` past the contacts), gamma to 1, the multipliers
           to 0 and the frozen Coulomb limit to unset.
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

    def reset_friction_limits(self):
        """Unfreeze every contact's Coulomb limit (GPU Newton, at the start of each time step), so each
        contact's first positive estimate in the step freezes it again (:func:`frozen_friction_limit`). A
        capture-safe fill, no host synchronization."""
        self.friction_limit.current.fill_(FRICTION_LIMIT_UNSET)

    @property
    def data(self) -> ConstraintSetData:  # type: ignore
        """Struct of arrays for kernel inputs/outputs."""
        return self._data


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


if __name__ == "__main__":
    wp.init()
    unittest.main()
