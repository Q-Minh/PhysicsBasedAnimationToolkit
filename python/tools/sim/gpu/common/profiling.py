"""
Profiling scopes for the GPU simulator.

Wrap algorithm phases in ``with profiling.scope("anderson.gram"): ...`` (or decorate a function with
``@profiling.scoped("vbd.sweep")``) using hierarchical, dotted names. Scopes nest at run time, and the
nesting forms a path such as ``solve/solve.finalize_subproblem/contact.detect``.

Backends (see plans/02-profiling.md):

- ``Backend.OFF`` (default): ``scope`` returns a shared no-op context manager. It performs no allocation
  and no device work, so it is safe inside CUDA graph capture.
- ``Backend.KERNELS``: CUPTI (``gpu/common/cupti.py``) records the GPU start and end of every kernel
  (Warp, ``cuda.compute``/CUB, cupy), memset and memcpy. Each activity is attributed to the innermost
  scope that was open when it was launched, through CUPTI external correlation ids, giving per-scope
  GPU busy time with the same accuracy as Nsight Systems. Requires eager execution.
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

from ...profiling.scope_paths import PATH_SEPARATOR as _PATH_SEPARATOR, path_ancestors

_NO_OP_SCOPE = contextlib.nullcontext()
_NVTX_DOMAIN = "pbat"
_NVTX_STEP = "step"


class Backend(enum.Enum):
    """Profiling backend, selected with set_backend (see the module docstring)."""
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
    union_ms: dict[str, float]
    """Duration of the union of activity intervals per scope path, which does not double-count
    activities running concurrently on several streams."""
    union_incl_ms: dict[str, float]
    """Like ``union_ms``, but inclusive of nested scopes: the union over the path and all paths below it,
    for every path and ancestor path. Unions of nested scopes overlap, so they can't be summed instead."""
    step_union_ms: float
    """Duration of the union of all activity intervals in the step."""
    step_span_ms: float
    """From the first activity's start to the last activity's end in the step."""

    @property
    def total_busy_ms(self) -> float:
        """Sum of the exclusive busy times of all scope paths, in ms."""
        return sum(self.busy_ms.values())


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    """Total length of the union of [start, end) intervals."""
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def inclusive_union_ns(intervals: dict[str, list[tuple[int, int]]]) -> dict[str, int]:
    """Per path and ancestor path, the union length of the intervals of that path and all paths below it."""
    grouped: dict[str, list[tuple[int, int]]] = {}
    for path, path_intervals in intervals.items():
        for ancestor in path_ancestors(path):
            grouped.setdefault(ancestor, []).extend(path_intervals)
    return {path: interval_union_ns(group) for path, group in grouped.items()}


class _KernelRecorder:
    """Attributes CUPTI activity records to scope paths for ``Backend.KERNELS``."""

    def __init__(self):
        self._collector = None
        self._stack: list[str] = []
        self._path_ids: dict[str, int] = {}
        self._calls: dict[str, int] = {}
        self._active = False

    def _push_path(self):
        path = _PATH_SEPARATOR.join(self._stack)
        # External ids start at 1, since 0 can't be told apart from "no id".
        self._collector.push(self._path_ids.setdefault(path, len(self._path_ids) + 1))
        return path

    def load(self):
        """Load CUPTI (raises RuntimeError if it is not installed)."""
        if self._collector is None:
            from .cupti import ActivityCollector

            self._collector = ActivityCollector()

    def begin_step(self):
        """Enable CUPTI, drop activity recorded before the step and open the root scope."""
        self.load()
        self._collector.enable()
        wp.synchronize()
        self._collector.flush()  # discard activity from before the step
        self._stack.clear()
        self._calls.clear()
        self._active = True
        self._push_path()

    def end_step(self) -> StepProfile:
        """Close the root scope, collect the step's CUPTI records and attribute them to scope paths."""
        self._collector.pop()
        self._active = False
        wp.synchronize()
        activities, external = self._collector.flush()
        paths = {i: path for path, i in self._path_ids.items()}
        intervals: dict[str, list[tuple[int, int]]] = {}
        kernels: dict[tuple[str, str], tuple[float, int]] = {}
        for activity in activities:
            path = paths.get(external.get(activity.correlation_id))
            if path is None:
                continue
            intervals.setdefault(path, []).append((activity.start, activity.end))
            ms, count = kernels.get((path, activity.name), (0.0, 0))
            kernels[(path, activity.name)] = (ms + (activity.end - activity.start) * 1e-6, count + 1)
        everything = [interval for per_path in intervals.values() for interval in per_path]
        return StepProfile(
            busy_ms={p: sum(e - s for s, e in iv) * 1e-6 for p, iv in intervals.items()},
            launches={p: len(iv) for p, iv in intervals.items()},
            calls=dict(self._calls),
            kernels=kernels,
            union_ms={p: interval_union_ns(iv) * 1e-6 for p, iv in intervals.items()},
            union_incl_ms={p: ns * 1e-6 for p, ns in inclusive_union_ns(intervals).items()},
            step_union_ms=interval_union_ns(everything) * 1e-6,
            step_span_ms=(
                (max(e for _, e in everything) - min(s for s, _ in everything)) * 1e-6
                if everything
                else 0.0
            ),
        )

    def disable(self):
        """Stop CUPTI collection (when switching to another backend)."""
        if self._collector is not None:
            self._collector.disable()

    def push(self, name: str):
        """Enter a nested scope: correlate subsequent launches with its path."""
        if not self._active:
            return
        self._stack.append(name)
        path = self._push_path()
        self._calls[path] = self._calls.get(path, 0) + 1

    def pop(self):
        """Leave the innermost scope."""
        if not self._active:
            return
        self._collector.pop()
        self._stack.pop()


_backend = Backend.OFF
_recorder = _KernelRecorder()


def set_backend(backend: Backend):
    """Select the profiling backend. Call between steps, not while a step is being profiled."""
    global _backend
    if backend is Backend.KERNELS:
        _recorder.load()
    elif _backend is Backend.KERNELS:
        _recorder.disable()
    _backend = backend


def get_backend() -> Backend:
    """The active profiling backend."""
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
    """Context manager returned by scope() while a backend is active."""
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
        self.assertLessEqual(profile.step_union_ms, profile.step_span_ms)
        self.assertLessEqual(profile.step_union_ms, profile.total_busy_ms + 1e-9)
        self.assertEqual(set(profile.union_incl_ms), {"", "outer", "outer/inner", "outer/inner.decorated"})
        self.assertGreaterEqual(profile.union_incl_ms["outer"], profile.union_incl_ms["outer/inner"])
        self.assertGreaterEqual(profile.union_incl_ms["outer"], profile.union_ms["outer"])

    def test_inclusive_union(self):
        self.assertEqual(path_ancestors("a/b/c"), ["a", "a/b", "a/b/c"])
        self.assertEqual(path_ancestors(""), [""])
        unions = inclusive_union_ns(
            {"a": [(0, 10)], "a/b": [(5, 20)], "a/b/c": [(30, 40)], "x/y": [(0, 3)], "": [(50, 60)]}
        )
        # "a" covers [0, 20) and [30, 40), its own and its descendants' overlapping intervals; "x" has no
        # activity of its own but gets its child's; unscoped work stays separate.
        self.assertEqual(unions, {"a": 30, "a/b": 25, "a/b/c": 10, "x": 3, "x/y": 3, "": 10})

    def test_interval_union(self):
        self.assertEqual(interval_union_ns([(0, 10), (5, 15), (20, 30)]), 25)
        self.assertEqual(interval_union_ns([]), 0)


if __name__ == "__main__":
    wp.init()
    unittest.main()
