"""
Profiling scopes for the GPU simulator.

Wrap algorithm phases in ``with profiling.scope("anderson.gram"): ...`` (or decorate a function with
``@profiling.scoped("vbd.sweep")``) using hierarchical, dotted names. Scopes nest at run time, and the
nesting forms a path such as ``solve/solve.finalize_subproblem/contact.detect``.

Backends (see plans/02-profiling.md):

- ``Backend.OFF`` (default): ``scope`` returns a shared no-op context manager. It performs no allocation
  and no device work, so it is safe inside CUDA graph capture.
- ``Backend.KERNELS``: Warp activity timing records the GPU duration of every Warp kernel launch, memset
  and memcpy. Each activity is attributed to the innermost scope that was open when it was launched,
  giving per-scope GPU busy time. Scope boundaries are also bracketed with CUDA events on the current
  stream, whose elapsed time (the scope's GPU timeline, including idle gaps) serves as a diagnostic for
  work Warp does not time (e.g. ``cuda.compute`` and cupy kernels). Requires eager execution.
- ``Backend.NVTX``: scopes emit NVTX ranges in the ``pbat`` domain, for Nsight Systems. Each profiled
  step is wrapped in a ``step`` range, so the importer (``python/tools/sim/profiling/nsys.py``) can
  tell measured steps from warm-up steps.

Profile one simulation step with::

    profiling.set_backend(profiling.Backend.KERNELS)
    profiling.begin_step()
    state.step()
    result = profiling.end_step()  # StepProfile for KERNELS, None otherwise
"""

import contextlib
import dataclasses
import enum
import functools

import nvtx
import warp as wp

_NO_OP_SCOPE = contextlib.nullcontext()
_NVTX_DOMAIN = "pbat"
_NVTX_STEP = "step"
_PATH_SEPARATOR = "/"


class Backend(enum.Enum):
    OFF = "off"
    KERNELS = "kernels"
    NVTX = "nvtx"


@dataclasses.dataclass
class StepProfile:
    """GPU timings of one profiled step (``Backend.KERNELS``).

    Paths join nested scope names with ``/``. The empty path ``""`` collects work outside any scope.
    """

    busy_ms: dict[str, float]
    """GPU busy time per scope path, exclusive of nested scopes (sum of activity durations)."""
    launches: dict[str, int]
    """Number of timed activities (kernel launches, memsets, memcpys) per scope path."""
    calls: dict[str, int]
    """Number of times each scope path was entered (e.g. VBD sweeps per step)."""
    kernels: dict[tuple[str, str], tuple[float, int]]
    """(scope path, activity name) -> (GPU busy time in ms, count)."""
    timeline_ms: dict[str, float]
    """GPU timeline per scope path, inclusive of nested scopes, from CUDA events at scope boundaries.
    Includes idle gaps where the GPU waits for the host, so it is only a diagnostic in eager mode."""
    step_timeline_ms: float
    """GPU timeline of the whole step, from CUDA events around it."""

    @property
    def total_busy_ms(self) -> float:
        return sum(self.busy_ms.values())


class _KernelRecorder:
    """Attributes Warp activity timings to scope paths for ``Backend.KERNELS``."""

    def __init__(self):
        self._stack: list[str] = []
        self._path = ""
        # (activity index at which the current path starts, path), in launch order
        self._marks: list[tuple[int, str]] = []
        # (path, start event index, end event index), one per closed scope instance
        self._brackets: list[tuple[str, int, int]] = []
        self._open: list[int] = []
        self._events: list[wp.Event] = []
        self._n_events = 0
        self._active = False

    @staticmethod
    def _activity_count() -> int:
        # Warp has no public accessor for the number of activities recorded so far.
        return wp._src.context.runtime.core.wp_cuda_timing_get_result_count()

    def _record_event(self) -> int:
        if self._n_events == len(self._events):
            self._events.append(wp.Event(enable_timing=True))
        index = self._n_events
        wp.get_stream().record_event(self._events[index])
        self._n_events += 1
        return index

    def _set_path(self):
        self._path = _PATH_SEPARATOR.join(self._stack)
        self._marks.append((self._activity_count(), self._path))

    def begin_step(self):
        self._stack.clear()
        self._marks.clear()
        self._brackets.clear()
        self._open.clear()
        self._n_events = 0
        wp.timing_begin(cuda_filter=wp.TIMING_ALL, synchronize=True)
        self._active = True
        self._set_path()
        self._step_begin_event = self._record_event()

    def end_step(self) -> StepProfile:
        step_end_event = self._record_event()
        activities = wp.timing_end(synchronize=True)
        self._active = False
        busy_ms: dict[str, float] = {}
        launches: dict[str, int] = {}
        kernels: dict[tuple[str, str], tuple[float, int]] = {}
        bounds = [index for index, _ in self._marks[1:]] + [len(activities)]
        for (begin, path), end in zip(self._marks, bounds):
            for activity in activities[begin:end]:
                busy_ms[path] = busy_ms.get(path, 0.0) + activity.elapsed
                launches[path] = launches.get(path, 0) + 1
                ms, count = kernels.get((path, activity.name), (0.0, 0))
                kernels[(path, activity.name)] = (ms + activity.elapsed, count + 1)
        timeline_ms: dict[str, float] = {}
        calls: dict[str, int] = {}
        for path, begin, end in self._brackets:
            elapsed = wp.get_event_elapsed_time(
                self._events[begin], self._events[end], synchronize=False
            )
            timeline_ms[path] = timeline_ms.get(path, 0.0) + elapsed
            calls[path] = calls.get(path, 0) + 1
        step_timeline_ms = wp.get_event_elapsed_time(
            self._events[self._step_begin_event],
            self._events[step_end_event],
            synchronize=False,
        )
        return StepProfile(busy_ms, launches, calls, kernels, timeline_ms, step_timeline_ms)

    def push(self, name: str):
        if not self._active:
            return
        self._stack.append(name)
        self._set_path()
        self._open.append(self._record_event())

    def pop(self):
        if not self._active:
            return
        end = self._record_event()
        self._brackets.append((self._path, self._open.pop(), end))
        self._stack.pop()
        self._set_path()


_backend = Backend.OFF
_recorder = _KernelRecorder()


def set_backend(backend: Backend):
    """Select the profiling backend. Call between steps, not while a step is being profiled."""
    global _backend
    _backend = backend


def get_backend() -> Backend:
    return _backend


def is_enabled() -> bool:
    """Whether a backend other than ``Backend.OFF`` is active. Profiled steps run eagerly."""
    return _backend is not Backend.OFF


def push(name: str):
    """Open a scope named ``name`` without a ``with`` block. Must be matched by :func:`pop`."""
    if _backend is Backend.KERNELS:
        _recorder.push(name)
    elif _backend is Backend.NVTX:
        nvtx.push_range(name, domain=_NVTX_DOMAIN)


def pop():
    """Close the scope opened by the matching :func:`push`."""
    if _backend is Backend.KERNELS:
        _recorder.pop()
    elif _backend is Backend.NVTX:
        nvtx.pop_range(domain=_NVTX_DOMAIN)


class _Scope:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        push(self.name)
        return self

    def __exit__(self, *exc):
        pop()
        return False


def scope(name: str) -> contextlib.AbstractContextManager:
    """Return a context manager that profiles the enclosed code under ``name``.

    Args:
        name (str): Hierarchical, dot-separated phase name, e.g. ``"vbd.sweep"``.

    Returns:
        contextlib.AbstractContextManager: A no-op while profiling is off.
    """
    if _backend is Backend.OFF:
        return _NO_OP_SCOPE
    return _Scope(name)


def scoped(name: str):
    """Decorator that runs the decorated function inside ``scope(name)``."""

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if _backend is Backend.OFF:
                return f(*args, **kwargs)
            with _Scope(name):
                return f(*args, **kwargs)

        return wrapper

    return decorator


def begin_step():
    """Start profiling one step. Pair with :func:`end_step`."""
    if _backend is Backend.KERNELS:
        _recorder.begin_step()
    elif _backend is Backend.NVTX:
        nvtx.push_range(_NVTX_STEP, domain=_NVTX_DOMAIN)


def end_step() -> StepProfile | None:
    """Finish the step started by :func:`begin_step`.

    Returns:
        StepProfile | None: The step's timings for ``Backend.KERNELS``, else ``None``.
    """
    if _backend is Backend.KERNELS:
        return _recorder.end_step()
    if _backend is Backend.NVTX:
        nvtx.pop_range(domain=_NVTX_DOMAIN)
    return None


import unittest


@wp.kernel
def _test_fill_kernel(a: wp.array[wp.float32], value: wp.float32):
    a[wp.tid()] = value


class TestProfiling(unittest.TestCase):

    def tearDown(self):
        set_backend(Backend.OFF)

    def test_off_is_shared_no_op(self):
        set_backend(Backend.OFF)
        self.assertIs(scope("a"), scope("b"))
        self.assertIsNone(end_step())

    def test_kernels_attributed_to_innermost_scope(self):
        a = wp.zeros(1024, dtype=wp.float32)
        launch = lambda: wp.launch(_test_fill_kernel, dim=a.shape[0], inputs=[a, 1.0])

        @scoped("inner.decorated")
        def decorated():
            launch()

        set_backend(Backend.KERNELS)
        launch()  # compile and load outside the profiled step
        wp.synchronize()
        begin_step()
        launch()
        with scope("outer"):
            launch()
            with scope("inner"):
                launch()
                launch()
            decorated()
            launch()
        profile = end_step()
        self.assertEqual(profile.launches, {"": 1, "outer": 2, "outer/inner": 2, "outer/inner.decorated": 1})
        self.assertGreater(profile.total_busy_ms, 0.0)
        self.assertEqual(profile.calls, {"outer": 1, "outer/inner": 1, "outer/inner.decorated": 1})
        self.assertGreaterEqual(profile.timeline_ms["outer"], profile.timeline_ms["outer/inner"])
        self.assertGreaterEqual(profile.step_timeline_ms, profile.timeline_ms["outer"])


if __name__ == "__main__":
    wp.init()
    unittest.main()
