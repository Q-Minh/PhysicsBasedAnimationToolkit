import warp as wp

from .ogc import ContactPairsData
from .. import common
from ..common.buffer import DoubleBuffer


@wp.kernel
def _warm_start_constraints(
    contacts: ContactPairsData,  # pyright: ignore[reportGeneralTypeIssues]
    n_u: wp.int32,  # Number of u primitives
    u_prev: wp.array[wp.int32],
    v_prev: wp.array[wp.int32],
    s_prev: wp.array[wp.float32],
    gamma_prev: wp.array[wp.float32],
    lambda_n_prev: wp.array[wp.float32],
    lambda_f_prev: wp.array[wp.vec2f],
    u: wp.array[wp.int32],
    v: wp.array[wp.int32],
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
    if pu < wp.int32(0):
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

    :meth:`prepare_warm_start` uses :meth:`~DoubleBuffer.copy_to_alternate` (a
    real ``wp.copy``, not a pointer swap) so the pipeline is compatible with
    CUDA graph capture.

    Fields
    ------
    u, v      : int32   contact pair keys (-1 = empty slot).
    s         : float32 inequality slack variable.
    gamma     : float32 constraint decay factor (default 1).
    lambda_n  : float32 normal Lagrange multiplier.
    lambda_f  : vec2f   friction Lagrange multiplier (tangential plane).

    Parameters
    ----------
    n_u : int
        Number of u primitives (vertices, half-edges, or faces) in the mesh.
    capacity : int
        Maximum number of contacts for one step (matches the OGC contact capacity).
    """

    capacity: int
    n_u: int
    u: DoubleBuffer
    v: DoubleBuffer
    s: DoubleBuffer
    gamma: DoubleBuffer
    lambda_n: DoubleBuffer
    lambda_f: DoubleBuffer
    _streams: list[list[wp.Stream]]
    _data: ConstraintSetData  # type: ignore

    def __init__(self, n_u: int, capacity: int):
        self.n_u = n_u
        self.capacity = capacity
        self._streams = [
            [wp.Stream() for _ in range(6)] for _ in range(4)
        ]  # list of streams for each field per contact type

        def _make(value, dtype):
            return DoubleBuffer(
                wp.full(shape=(capacity,), value=value, dtype=dtype),
                wp.full(shape=(capacity,), value=value, dtype=dtype),
            )

        self.u = _make(-1, wp.int32)
        self.v = _make(-1, wp.int32)
        self.s = _make(0.0, wp.float32)
        self.gamma = _make(1.0, wp.float32)
        self.lambda_n = _make(0.0, wp.float32)
        self.lambda_f = _make(0.0, wp.vec2f)
        self._data = ConstraintSetData()
        self._data.s = self.s.current
        self._data.gamma = self.gamma.current
        self._data.lambda_n = self.lambda_n.current
        self._data.lambda_f = self.lambda_f.current

    def update_constraint_set(self, contacts: ContactPairsData):  # type: ignore
        """Prepare the current arrays for a new step, warm-starting from the previous snapshot.

        The four-step pipeline maximises GPU concurrency while remaining
        compatible with CUDA graph capture:

        1. **Async copy** current -> alternate (previous snapshot) on an internal
           copy stream via :meth:`~DoubleBuffer.copy_to_alternate`.
        2. **Fill** current arrays to defaults via ``fill_()`` on ``main_stream``.
           Safely overlaps step 1 — current and alternate are distinct allocations.
        3. **Fence** — ``main_stream`` waits for the copy stream to finish.
        4. **Warm-start kernel** — for each valid previous pair whose ``(u, v)``
           appears in the new OGC forward contact list, copy ``s``, ``gamma``,
           ``lambda_n``, ``lambda_f`` to the corresponding current slot.

        After this call, current slots that were warm-started have
        ``u.current[k] >= 0``; new contacts keep the default ``u.current[k] = -1``.

        Args:
            contacts: Forward contact pairs produced by ``Ogc.detect_contacts()``.
        """
        main_stream = wp.get_stream()
        fields = (self.u, self.v, self.s, self.gamma, self.lambda_n, self.lambda_f)

        # Step 1: async-copy current -> alternate on _copy_stream.
        for streams in self._streams:
            for buf, stream in zip(fields, streams):
                stream.wait_stream(main_stream)
                buf.copy_to_alternate(stream)
                main_stream.wait_stream(stream)

        # Step 2: fill current with defaults on main_stream.
        # Overlaps step 1 safely — current and alternate are distinct allocations.
        for streams in self._streams:
            with wp.ScopedStream(streams[0], sync_enter=False):
                self.u.current.fill_(-1)
            main_stream.wait_stream(streams[0])
            with wp.ScopedStream(streams[1], sync_enter=False):
                self.v.current.fill_(-1)
            main_stream.wait_stream(streams[1])
            # NOTE: The slack doesn't need any initialization, it is always updated during before/after solver iterations
            with wp.ScopedStream(streams[3], sync_enter=False):
                self.gamma.current.fill_(1.0)
            main_stream.wait_stream(streams[3])
            with wp.ScopedStream(streams[4], sync_enter=False):
                self.lambda_n.current.fill_(0.0)
            main_stream.wait_stream(streams[4])
            with wp.ScopedStream(streams[5], sync_enter=False):
                self.lambda_f.current.fill_(0.0)
            main_stream.wait_stream(streams[5])

        # Step 3: warm-start from alternate (previous) into current.
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
            stream=main_stream,
        )

    @property
    def data(self) -> ConstraintSetData:  # type: ignore
        """Struct of arrays for kernel inputs/outputs."""
        return self._data
