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
         "step_union_ms": ms,          # union over the whole step
         "step_span_ms": ms,           # first activity start to last activity end
         "step_timeline_ms": ms}       # production runs only (no per-phase timings): step GPU
                                       # time from CUDA events around it, i.e. graph replay
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

import numpy as np
import warp as wp

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

    def add_step(self, t: int, profile: StepProfile):
        self.steps.append(
            {
                "t": t,
                "busy_ms": profile.busy_ms,
                "launches": profile.launches,
                "calls": profile.calls,
                "union_ms": profile.union_ms,
                "step_union_ms": profile.step_union_ms,
                "step_span_ms": profile.step_span_ms,
            }
        )
        for key, (ms, count) in profile.kernels.items():
            acc = self._kernels.setdefault(key, [0.0, 0])
            acc[0] += ms
            acc[1] += count

    def add_step_timeline(self, t: int, step_timeline_ms: float):
        """Record only a step's GPU timeline (production runs, without per-phase timings)."""
        self.steps.append({"t": t, "busy_ms": {}, "launches": {}, "step_timeline_ms": step_timeline_ms})

    def to_dict(self) -> dict:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


def load(path: str) -> dict:
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
