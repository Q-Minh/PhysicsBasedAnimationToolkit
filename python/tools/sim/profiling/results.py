"""
Profiling result files: collection, JSON (de)serialization and text summaries.

The same schema is written by ``simulate.py --profile kernels`` (CUPTI, in-process) and by the
Nsight Systems importer, so reports can consume either. Layout::

    {
      "schema": "pbat-sim-profile/1",
      "source": "kernels" | "nsys" | "production",
      "metadata": {...},               # GPU, software versions, scene, solver and parameters
      "steps": [                       # one entry per measured (post-warm-up) step
        {"t": int,
         "busy_ms": {path: ms},        # GPU busy time per scope path, exclusive of nested scopes
         "launches": {path: n},        # timed activities per scope path
         "calls": {path: n},           # times each scope path was entered
         "union_ms": {path: ms},       # union of activity intervals per path (no double-counting
                                       # of concurrent streams)
         "union_incl_ms": {path: ms},  # the same union over the path and all paths below it, for every
                                       # path and ancestor path (inclusive unions can't be summed)
         "step_union_ms": ms,          # union over the whole step
         "step_span_ms": ms,           # first activity start to last activity end
         "step_timeline_ms": ms,       # production runs only (no per-phase timings): step GPU
                                       # time from CUDA events around it, i.e. graph replay
         "status": {...},              # simulate.py: the step's solve statuses, one entry per substep
                                       # (status.StatusLog.step_entry: outcome, iterations, ratios)
         "peak_speed": m/s}            # profiling/trajectory.py: the largest node speed over the
                                       # frame interval that contains the step (from saved frames)
      ],
      "kernels": [{"path": str, "name": str, "busy_ms": ms, "count": n}]  # summed over steps
    }

Scope paths join nested scope names with ``/``; ``""`` is work outside any scope.
"""

import datetime
import enum
import inspect
import json
import subprocess
from typing import TYPE_CHECKING

import numpy as np

from .scope_paths import path_ancestors

if TYPE_CHECKING:
    from ..gpu.common.profiling import StepProfile

SCHEMA = "pbat-sim-profile/1"
UNSCOPED = "(unscoped)"


def leaf(path: str) -> str:
    """The innermost scope name of a path."""
    return path.rsplit("/", 1)[-1] if path else UNSCOPED


class ProfileRecorder:
    """Accumulates per-step profiles of one run into the result schema."""

    def __init__(self, source: str, metadata: dict):
        self.source = source
        self.metadata = metadata
        self.steps: list[dict] = []
        self._kernels: dict[tuple[str, str], list] = {}

    def add_step(self, t: int, profile: "StepProfile", status: dict | None = None):
        """Add one profiled step's timings, and its solve statuses if given (``status.StatusLog.step_entry``),
        and accumulate its per-kernel totals."""
        step = {
            "t": t,
            "busy_ms": profile.busy_ms,
            "launches": profile.launches,
            "calls": profile.calls,
            "union_ms": profile.union_ms,
            "union_incl_ms": profile.union_incl_ms,
            "step_union_ms": profile.step_union_ms,
            "step_span_ms": profile.step_span_ms,
        }
        if status is not None:
            step["status"] = status
        self.steps.append(step)
        for key, (ms, count) in profile.kernels.items():
            acc = self._kernels.setdefault(key, [0.0, 0])
            acc[0] += ms
            acc[1] += count

    def add_step_timeline(self, t: int, step_timeline_ms: float, status: dict | None = None):
        """Record only a step's GPU timeline (production runs, without per-phase timings), and its solve
        statuses if given."""
        step = {"t": t, "busy_ms": {}, "launches": {}, "step_timeline_ms": step_timeline_ms}
        if status is not None:
            step["status"] = status
        self.steps.append(step)

    def to_dict(self) -> dict:
        """The run as a profiling result dict (schema above), kernels sorted by busy time."""
        kernels = [
            {"path": path, "name": name, "busy_ms": ms, "count": count}
            for (path, name), (ms, count) in self._kernels.items()
        ]
        kernels.sort(key=lambda k: -k["busy_ms"])
        return {
            "schema": SCHEMA,
            "source": self.source,
            "metadata": self.metadata,
            "steps": self.steps,
            "kernels": kernels,
        }


def save(result: dict, path: str):
    """Write a profiling result (or any JSON-serializable dict) to path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


def load(path: str) -> dict:
    """Read a profiling result, checking its schema."""
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    if result.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, got {result.get('schema')}")
    return result


def mean_busy_ms_per_step(result: dict, by_leaf: bool = False) -> dict[str, float]:
    """Mean GPU busy time per step, per scope path (or per innermost scope name if ``by_leaf``)."""
    totals: dict[str, float] = {}
    for step in result["steps"]:
        for path, ms in step["busy_ms"].items():
            key = leaf(path) if by_leaf else path
            totals[key] = totals.get(key, 0.0) + ms
    n = max(len(result["steps"]), 1)
    return {key: ms / n for key, ms in totals.items()}


def inclusive(per_path: dict[str, float]) -> dict[str, float]:
    """Inclusive values from exclusive ones: per path and ancestor path, the sum over the path and all
    paths below it. Valid for additive quantities (busy time, launches), not for interval unions."""
    out: dict[str, float] = {}
    for path, value in per_path.items():
        for ancestor in path_ancestors(path):
            out[ancestor] = out.get(ancestor, 0.0) + value
    return out


def _mean_per_step(result: dict, key: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for step in result["steps"]:
        for path, value in step.get(key, {}).items():
            totals[path] = totals.get(path, 0.0) + value
    n = max(len(result["steps"]), 1)
    return {path: value / n for path, value in totals.items()}


def scope_tree(result: dict) -> list[dict]:
    """Scope tree rows with per-step means, depth first, siblings by decreasing total busy time.

    Each row has ``path``, ``name`` (innermost scope), ``depth``, ``self_ms`` and ``total_ms`` (busy time,
    exclusive and inclusive of nested scopes), ``self_launches`` and ``total_launches``, ``calls`` and
    ``union_incl_ms`` (``None`` for results recorded without inclusive unions).
    """
    self_ms = _mean_per_step(result, "busy_ms")
    self_launches = _mean_per_step(result, "launches")
    calls = _mean_per_step(result, "calls")
    has_unions = any("union_incl_ms" in step for step in result["steps"])
    unions = _mean_per_step(result, "union_incl_ms")
    total_ms, total_launches = inclusive(self_ms), inclusive(self_launches)
    children: dict[str | None, list[str]] = {}
    for path in total_ms:
        parent = path.rsplit("/", 1)[0] if "/" in path else None
        children.setdefault(parent, []).append(path)
    rows = []

    def visit(path: str, depth: int):
        rows.append(
            {
                "path": path,
                "name": leaf(path),
                "depth": depth,
                "self_ms": self_ms.get(path, 0.0),
                "total_ms": total_ms[path],
                "self_launches": self_launches.get(path, 0.0),
                "total_launches": total_launches.get(path, 0.0),
                "calls": calls.get(path, 0.0),
                "union_incl_ms": unions.get(path) if has_unions else None,
            }
        )
        for child in sorted(children.get(path, []), key=lambda c: -total_ms[c]):
            visit(child, depth + 1)

    for root in sorted(children.get(None, []), key=lambda r: -total_ms[r]):
        visit(root, 0)
    return rows


def format_summary(result: dict) -> str:
    """Text table of mean GPU busy time per step, by innermost scope and by scope path."""
    meta = result["metadata"]
    n = len(result["steps"])
    lines = [
        f"{meta.get('solver', '?')} on {meta.get('scene', '?')}: {n} measured steps "
        f"({result['source']} backend, {meta.get('gpu', '?')})"
    ]
    for title, by_leaf in (("By phase", True), ("By scope path", False)):
        rows = sorted(mean_busy_ms_per_step(result, by_leaf).items(), key=lambda r: -r[1])
        if not rows:
            continue
        total = sum(ms for _, ms in rows)
        lines.append(f"\n{title} (mean GPU busy ms/step, total {total:.3f} ms):")
        for key, ms in rows:
            share = 100.0 * ms / total if total > 0.0 else 0.0
            lines.append(f"  {ms:10.4f} ms  {share:5.1f}%  {key or UNSCOPED}")
    unions = [s["step_union_ms"] for s in result["steps"] if "step_union_ms" in s]
    if unions:
        lines.append(f"\nStep GPU busy time, union over streams: mean {np.mean(unions):.3f} ms/step")
    timelines = [s["step_timeline_ms"] for s in result["steps"] if "step_timeline_ms" in s]
    if timelines:
        execution = meta.get("execution", "eager")
        note = "graph replay" if execution == "graph" else "eager, includes idle gaps"
        lines.append(f"\nStep GPU timeline ({note}): mean {np.mean(timelines):.3f} ms/step")
    return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, enum.Enum):
        return value.name
    return str(value)


def reflect_params(obj, _depth: int = 0) -> dict:
    """Public scalar attributes of a parameter object, recursing into nested parameter objects."""
    out = {}
    for name, value in inspect.getmembers(obj):
        if name.startswith("_") or value is None or callable(value):
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (bool, int, float, str, enum.Enum)):
            out[name] = _jsonable(value)
        elif _depth < 2 and not isinstance(value, (list, tuple, dict)):
            nested = reflect_params(value, _depth + 1)
            if nested:
                out[name] = nested
        elif hasattr(value, "name") and hasattr(value, "value"):
            out[name] = _jsonable(value)
    return out


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_metadata(state, scene: str, warmup_steps: int) -> dict:
    """Describe the hardware, software and simulation configuration of a profiled run.

    Args:
        state: The ``state.SimulationState`` being profiled.
        scene (str): The scene specification (``file.h5:group``).
        warmup_steps (int): Number of initial steps excluded from the measurement.
    """
    import warp as wp  # only when recording: reading results and reports needs no GPU stack

    device = wp.get_device()
    runtime = wp._src.context.runtime
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "gpu": device.name,
        "gpu_arch": device.arch,
        "display_driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "cuda_driver": ".".join(map(str, runtime.driver_version)),
        "cuda_toolkit": ".".join(map(str, runtime.toolkit_version)),
        "warp": wp.__version__,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "scene": scene,
        "solver": state.solver.name,
        "cd": state.cd_type.name,
        "dt": state.dt,
        "substeps": state.substeps,
        "bdf_order": state.bdf_scheme,
        "n_nodes": int(state.fem.data.x.shape[0]),
        "n_elements": int(state.fem.data.E.shape[0]),
        "warmup_steps": warmup_steps,
        "solver_params": reflect_params(state.params_cpu[state.solver]),
        "contact_params": reflect_params(state.contact_params),
        "cd_params": reflect_params(state.cd_params[state.cd_type]),
    }


import unittest


class TestScopeTree(unittest.TestCase):

    def test_inclusive_sums_descendants(self):
        self.assertEqual(
            inclusive({"a": 1.0, "a/b": 2.0, "a/b/c": 4.0, "x/y": 8.0, "": 16.0}),
            {"a": 7.0, "a/b": 6.0, "a/b/c": 4.0, "x": 8.0, "x/y": 8.0, "": 16.0},
        )

    def test_scope_tree(self):
        result = {
            "steps": [
                {"busy_ms": {"s": 1.0, "s/sweep": 6.0, "s/detect": 2.0, "": 0.5},
                 "launches": {"s/sweep": 9, "s/detect": 3},
                 "calls": {"s": 1, "s/sweep": 3},
                 "union_incl_ms": {"s": 8.0, "s/sweep": 6.0, "s/detect": 2.0, "": 0.5}},
                {"busy_ms": {"s": 1.0, "s/sweep": 8.0, "s/detect": 2.0, "": 0.5},
                 "launches": {"s/sweep": 11, "s/detect": 3},
                 "calls": {"s": 1, "s/sweep": 3},
                 "union_incl_ms": {"s": 10.0, "s/sweep": 8.0, "s/detect": 2.0, "": 0.5}},
            ]
        }
        rows = scope_tree(result)
        self.assertEqual([(r["path"], r["depth"]) for r in rows],
                         [("s", 0), ("s/sweep", 1), ("s/detect", 1), ("", 0)])
        s = rows[0]
        self.assertEqual((s["self_ms"], s["total_ms"], s["total_launches"], s["union_incl_ms"]), (1.0, 10.0, 13.0, 9.0))
        self.assertEqual(rows[1]["calls"], 3.0)


class TestReportImports(unittest.TestCase):

    def test_reports_load_no_gpu_stack(self):
        """Reading results and writing reports must not import the GPU stack: an older pbatoolkit that
        ships its own hdf5.dll then breaks h5py, and reporting needs none of it."""
        import os
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
        heavy = ["warp", "cupy", "pbatoolkit", "h5py", "nvtx"]
        code = (
            "import sys; "
            "import python.tools.sim.profiling.report, python.tools.sim.profiling.overheads; "
            f"print([m for m in {heavy!r} if m in sys.modules])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(out.strip(), "[]")


if __name__ == "__main__":
    unittest.main()

