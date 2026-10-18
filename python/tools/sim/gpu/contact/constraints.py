import warp as wp

from . import pairs
from .. import common
from ..common.buffer import DoubleBuffer


@wp.kernel
def _snapshot_and_reset_constraints(
    n_u: wp.uint32,
    u: wp.array[wp.uint32],
    v: wp.array[wp.uint32],
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
    u_prev: wp.array[wp.uint32],
    v_prev: wp.array[wp.uint32],
    s_prev: wp.array[wp.float32],
    gamma_prev: wp.array[wp.float32],
    lambda_n_prev: wp.array[wp.float32],
    lambda_f_prev: wp.array[wp.vec2f],
):
    """Copy slot k of the current arrays to the previous snapshot, then reset it to defaults.

    Only u (empty-slot sentinel n_u), gamma (1) and the multipliers (0) are reset: v is only read
    where u is valid, and s is recomputed by the solver's dual updates.
    """
    k = wp.tid()  # type: ignore
    u_prev[k] = u[k]
    v_prev[k] = v[k]
    s_prev[k] = s[k]
    gamma_prev[k] = gamma[k]
    lambda_n_prev[k] = lambda_n[k]
    lambda_f_prev[k] = lambda_f[k]
    u[k] = n_u
    gamma[k] = wp.float32(1)
    lambda_n[k] = wp.float32(0)
    lambda_f[k] = wp.vec2f(wp.float32(0), wp.float32(0))


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
    u: wp.array[wp.uint32],
    v: wp.array[wp.uint32],
    s: wp.array[wp.float32],
    gamma: wp.array[wp.float32],
    lambda_n: wp.array[wp.float32],
    lambda_f: wp.array[wp.vec2f],
):
    """Restore persisting AL state from the previous snapshot into the current arrays.

    Thread k reads previous slot k. If it holds a valid contact pair (u, v) >= 0
    that can be located in the new forward contact list, the AL state
    (s, gamma, lambda_n, lambda_f) is copied to its new position.
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


@wp.struct
class ConstraintSetData:
    """Struct of arrays for GPU contact constraints, used for kernel inputs/outputs.

    See :class:`ConstraintSet` for the corresponding CPU-side storage class.
    """

    s: wp.array[wp.float32]
    gamma: wp.array[wp.float32]
    lambda_n: wp.array[wp.float32]
    lambda_f: wp.array[wp.vec2f]


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
        self._data = ConstraintSetData()
        self._data.s = self.s.current
        self._data.gamma = self.gamma.current
        self._data.lambda_n = self.lambda_n.current
        self._data.lambda_f = self.lambda_f.current

    def update_constraint_set(self, contacts: pairs.Pairs):  # type: ignore
        """Prepare the current arrays for a new step, warm-starting from the previous snapshot.

        Two kernels, so the update adds only two nodes to a captured graph:

        1. **Snapshot and reset** (:func:`_snapshot_and_reset_constraints`): copy every field's
           current slot to alternate (the previous snapshot), then reset u to the empty-slot
           sentinel ``n_u``, gamma to 1 and the multipliers to 0.
        2. **Warm start** (:func:`_warm_start_constraints`): for each valid previous pair whose
           ``(u, v)`` appears in the new forward contact list, copy ``s``, ``gamma``,
           ``lambda_n``, ``lambda_f`` to the corresponding current slot. It writes arbitrary
           slots, so it must follow the whole reset.

        After this call, current slots that were warm-started have a valid u value;
        new contacts keep the default sentinel ``u.current[k] = n_u``.

        Args:
            contacts: Forward contact pairs produced by a collision detector.
        """
        fields = (self.u, self.v, self.s, self.gamma, self.lambda_n, self.lambda_f)
        # Step 1: snapshot current into alternate and reset current to defaults.
        wp.launch(
            _snapshot_and_reset_constraints,
            dim=self.capacity,
            inputs=[self.n_u]
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
                self.u.current,
                self.v.current,
                self.s.current,
                self.gamma.current,
                self.lambda_n.current,
                self.lambda_f.current,
            ],
        )

    @property
    def data(self) -> ConstraintSetData:  # type: ignore
        """Struct of arrays for kernel inputs/outputs."""
        return self._data
