"""
Benchmark results: accuracy-level crossings, per-level aggregates, time budgets, and the results file.

This module has no GPU dependencies, so the plot script (``benchmark_plot.py``) can read results
without Warp. ``benchmark.py`` writes the results file; ``python -m python.tools.sim.benchmark_results``
runs this module's tests.

Accuracy levels are error reductions 1 - e/e0 (0.5 = half the initial error removed). For one solve's
per-iteration log (entry j is the state after j iterations, with cumulative solver time t_j):

- level l is **crossed** at the first entry j with 1 - e_j/e0 >= l, at time t_j;
- the **increment** of a level is the time from the previous crossing to its own, but an iteration's
  time goes only to the **highest** level it crosses: levels it jumps over are reached with a zero
  increment (``skipped``). Newton's first iteration reaching 80% thus puts all of its time on 80%.
  The increments of the reached levels sum to the time of the highest one.
- a level crossed after the solve's **time budget** isn't reached: the budget is tested after each
  iteration, so the last one can end past it (by up to one Newton step).

Results file layout (HDF5, ``FORMAT``):

    /                                attrs: format, version, levels_options, budget_factor, budget_options,
                                     newton_stall_iterations, newton_stall_reduction, dataset, device,
                                     git_commit, date, sim_params, solvers
    /StaticSolveSpec                 the run's spec (static.StaticSolveSpec), attributes
    /dataset_attrs                   the dataset's scalar attributes
    /<benchmark>                     attrs: levels (the benchmark's), cases
    /<benchmark>/<case>              attrs: benchmark, problem, h, scene, label, energy, set, [budget] (a
                                     fixed --budget), budget_source, max_seconds (Newton's cap),
                                     scene_params, sim_params_toml (every solver parameter of the
                                     case's simulation, TOML), newton_cg_* (Newton's CG limits), gpu_start_*,
                                     gpu_end_* (the GPU's state, nvidia-smi: time, pstate, clocks_sm,
                                     clocks_mem, clocks_max_sm, temperature_gpu, power_draw,
                                     power_limit, clocks_event_reasons_active)
    /<benchmark>/<case>/<solver>     attrs: solver, n_samples
        sample, e0, iterations, sweeps, solver_time, host_time, final_relative_error, termination,
        deformation_energy, budget, normalization, [min_J],
        [reference_error_bound, reference_relative_error_bound]                                 (S,)
        crossing_time, crossing_iteration, increment, normalized_increment, reached, skipped,
        [resolvable]                                                                            (S, L)
        aggregate/{increment,normalized_increment,crossing_time}/{count,mean,std,min,max,median,q25,q75}
                                                                                                (L,)
        logs/<sample:06d>/{error,time,sweeps}
        logs/<sample:06d>/newton/{status,cg_iterations,cg_capped,cg_residual,cg_tolerance,backtracks}
                                                                                (Newton, per step)
        trajectories/<sample:06d>/{iteration,t,t_target,x,level_frames,level_exact}   (opt-in, --snapshots)

``<benchmark>`` is ``static`` or ``dynamic_h<h>`` (e.g. ``dynamic_h0.0001``). A ``<case>`` is one x-axis
entry of the benchmark's figure: a scene, or ``<scene>_E<E>`` when the run holds several energy levels
of a scene. Times are seconds of solver time. ``budget`` is each solve's time budget: Newton's cap for
Newton, ``budget_factor`` x T_N for the others, or the fixed ``--budget``; ``normalization`` is the time
the sample's increments are divided by (``normalized_increment``): T_N, Newton's crossing time of the
highest level it reached on that problem (the same for every solver), or the fixed budget. Files written
before per-benchmark levels have the run's ``levels`` at the root instead (:func:`benchmark_levels`).
Newton's ``logs/<sample>/newton/*`` hold one entry per Newton step attempted (one ``solve_subproblem`` call):
its ``status`` (``stepped``, or why the last one was rejected), the CG iterations, whether CG stopped at its
cap, the CG residual norm and the absolute tolerance it was tested against, and the line-search
backtracks; log entry j + 1 follows step j when it stepped. NaN marks unreached levels (``reached`` False); an
unreached level's ``crossing_iteration`` is -1. Aggregates are over the samples that reached the
level (``count``); a skipped level counts with a zero increment. ``host_time`` is the solve's host
wall-clock time (loading the problem, the replay or Newton loop, reading the log back); minus
``solver_time``, it is the per-solve overhead outside the timed solve.

Trajectories (``benchmark.py --snapshots``): positions at chosen log entries of each solve, for videos.
``iteration`` (F',) are the log entries (sorted), ``t`` (F',) their solver times, ``t_target`` (F',) the
solver time each frame stands for (NaN for frames kept only as level crossings; |t - t_target| is the
spacing error, at most half a ring stride of iterations), ``x`` (F', N, 3) float32
the positions in the dataset FEM's node order. The frames are the snapshots nearest to the solver times
k * budget / F (k = 0, 1, ..., up to the solve's end) and to its end, so frame k of every solver shows the
same solver time and side-by-side playback runs at true speed; plus each reached level's crossing.
``level_frames`` (L,) indexes the frame of each level's crossing (-1: not reached), and ``level_exact``
says whether it is the crossing's own state (else the nearest snapshot). Attributes: ``requested_frames``,
``ring_slots`` (fewer than ``requested_frames`` when the memory cap is too small), ``stride`` (log entries
between ring snapshots at the end of the solve), ``device_bytes`` (the snapshots' device memory),
``video_seconds`` (the duration the frames play in) and ``playback_factor`` = video_seconds / budget: how
many times slower than solver time the video plays (1: real time; 200: slowed down 200x; below 1: sped up).
"""

from dataclasses import dataclass
import math

import h5py
import numpy as np

FORMAT = "pbat-benchmark-results"
VERSION = 1
# Default levels, decades of error reduction: the static references (the rest shape) are exact; the dynamic
# references resolve about 99.99% at best (float32 Newton solves)
STATIC_LEVELS = (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
DYNAMIC_LEVELS = (0.5, 0.9, 0.99, 0.999, 0.9999)
STATISTICS = ("count", "mean", "std", "min", "max", "median", "q25", "q75")


def parse_levels(text: str) -> tuple[float, ...]:
    """Comma-separated error reductions, e.g. ``"0.5,0.9,0.99"``, sorted and checked to lie in (0, 1)."""
    levels = tuple(sorted({float(v) for v in text.split(",") if v.strip()}))
    if not levels or levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError(f"Accuracy levels must lie in (0, 1), got {text!r}")
    return levels


def parse_levels_options(entries: list[str]) -> dict:
    """``[BENCHMARK=]L1,L2,...`` entries -> {key: levels}, with key ``None`` for an entry without a
    benchmark (every benchmark without its own) and e.g. ``"static"`` for one benchmark."""
    out = {}
    for entry in entries:
        key, sep, text = entry.rpartition("=")
        key = key if sep else None
        if key in out:
            raise ValueError(f"--levels given twice for {key or 'every benchmark'}")
        out[key] = parse_levels(text)
    return out


def default_levels(benchmark: str) -> tuple[float, ...]:
    """``STATIC_LEVELS`` for the static benchmark, ``DYNAMIC_LEVELS`` for the dynamic ones."""
    return STATIC_LEVELS if benchmark == "static" else DYNAMIC_LEVELS


def levels_for(table: dict, benchmark: str) -> tuple[float, ...]:
    """A benchmark's levels (:func:`parse_levels_options` table): its own entry, else the unkeyed one,
    else :func:`default_levels`."""
    return table.get(benchmark, table.get(None, default_levels(benchmark)))


def benchmark_levels(f: h5py.File, benchmark: str) -> np.ndarray:
    """The levels of a benchmark in a results file: its group's ``levels`` (else the file's)."""
    b = f[benchmark]
    return np.asarray(b.attrs["levels"] if "levels" in b.attrs else f.attrs["levels"], dtype=np.float64)


def benchmark_name(problem: str, h: float | None) -> str:
    """``static`` or ``dynamic_h<h>`` (as the dataset's set names, e.g. ``dynamic_h0.0001``)."""
    return "static" if problem == "static" else f"dynamic_h{h:g}"


@dataclass
class LevelCrossings:
    """Level crossings of one solve, one entry per level (in increasing order)."""

    time: np.ndarray  # (L,) solver time of the first crossing (s), NaN if not reached
    iteration: np.ndarray  # (L,) iterations at the first crossing, -1 if not reached
    increment: np.ndarray  # (L,) time attributed to the level (s), 0 if skipped, NaN if not reached
    reached: np.ndarray  # (L,) bool
    skipped: np.ndarray  # (L,) bool: reached by an iteration that also crossed a higher level


def level_crossings(
    relative_error: np.ndarray, time: np.ndarray, levels, time_limit: float = math.inf
) -> LevelCrossings:
    """Crossings and incremental times of a solve's log.

    Args:
        relative_error (np.ndarray): (n,) e_j / e0 per log entry (entry 0 = the initial state).
        time (np.ndarray): (n,) cumulative solver time per entry (s), with time[0] = 0.
        levels: Increasing error reductions in (0, 1).
        time_limit (float): The solve's time budget (s): entries after it don't count.

    Returns:
        LevelCrossings: Per level, the first crossing and the time attributed to it.
    """
    levels = np.asarray(levels, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    r = np.asarray(relative_error, dtype=np.float64)[: np.searchsorted(t, time_limit, side="right")]
    L = levels.size
    iteration = np.full(L, -1, dtype=np.int64)
    for i, level in enumerate(levels):
        # NaN errors never cross (comparisons are False)
        hit = np.nonzero(1.0 - r >= level)[0]
        if hit.size:
            iteration[i] = hit[0]
    reached = iteration >= 0
    crossing = np.where(reached, t[np.maximum(iteration, 0)], math.nan)
    # Crossings are monotone in the level: reaching a level reaches every lower one at or before it.
    skipped = np.zeros(L, dtype=bool)
    skipped[:-1] = reached[:-1] & reached[1:] & (iteration[:-1] == iteration[1:])
    increment = np.full(L, math.nan)
    previous = 0.0
    for i in range(L):
        if not reached[i]:
            break
        if skipped[i]:
            increment[i] = 0.0
            continue
        increment[i] = crossing[i] - previous
        previous = crossing[i]
    return LevelCrossings(crossing, iteration, increment, reached, skipped)


def normalization_time(crossings: LevelCrossings) -> tuple[float, int]:
    """T_N: the crossing time of the highest level a (Newton) solve reached, and that level's index;
    (NaN, -1) if it reached none."""
    reached = np.nonzero(crossings.reached)[0]
    if reached.size == 0:
        return math.nan, -1
    top = int(reached[-1])
    return float(crossings.time[top]), top


def aggregate(values: np.ndarray, reached: np.ndarray) -> dict[str, np.ndarray]:
    """Per-level statistics over the samples that reached each level.

    Args:
        values (np.ndarray): (S, L) per sample and level (e.g. increments or crossing times).
        reached (np.ndarray): (S, L) bool.

    Returns:
        dict[str, np.ndarray]: ``STATISTICS`` -> (L,) arrays: ``count`` (int) and float statistics,
        NaN where no sample reached the level. ``std`` is the population standard deviation.
    """
    values = np.asarray(values, dtype=np.float64).reshape(len(reached), -1)
    reached = np.asarray(reached, dtype=bool).reshape(values.shape)
    L = values.shape[1]
    out = {name: np.full(L, math.nan) for name in STATISTICS}
    out["count"] = reached.sum(axis=0).astype(np.int64)
    for j in range(L):
        v = values[reached[:, j], j]
        if v.size == 0:
            continue
        out["mean"][j] = v.mean()
        out["std"][j] = v.std()
        out["min"][j] = v.min()
        out["max"][j] = v.max()
        out["median"][j] = np.median(v)
        out["q25"][j] = np.quantile(v, 0.25)
        out["q75"][j] = np.quantile(v, 0.75)
    return out


def parse_budget_options(entries: list[str], option: str = "--budget") -> dict:
    """``[BENCHMARK[/CASE]=]SECONDS`` entries -> {key: seconds}, with key ``None`` for an entry without a
    target (every case), ``"static"`` for a benchmark, ``"static/Y1e6"`` for one case."""
    out = {}
    for entry in entries:
        key, sep, value = entry.rpartition("=")
        key = key if sep else None
        try:
            seconds = float(value)
        except ValueError:
            raise ValueError(f"{option} expects [BENCHMARK[/CASE]=]SECONDS, got {entry!r}") from None
        if not seconds > 0:
            raise ValueError(f"{option} needs a positive number of seconds, got {entry!r}")
        if key in out:
            raise ValueError(f"{option} given twice for {key or 'every case'}")
        out[key] = seconds
    return out


def _lookup(table: dict, benchmark: str, case: str):
    """The most specific entry of a :func:`parse_budget_options` table: case, then benchmark, then all."""
    for key in (f"{benchmark}/{case}", benchmark, None):
        if key in table:
            return key, table[key]
    return None, None


def fixed_budget(budgets: dict, benchmark: str, case: str) -> tuple[float | None, str]:
    """A case's fixed budget (``--budget``), the most specific entry: the case's, its benchmark's, then
    every case's; (None, "") if none applies (Newton's times set the budgets).

    Args:
        budgets: :func:`parse_budget_options` table.

    Returns:
        tuple[float | None, str]: The budget (s) and its source, as stored in the case's ``budget_source``.
    """
    key, seconds = _lookup(budgets, benchmark, case)
    if seconds is None:
        return None, ""
    return seconds, f"--budget {key + '=' if key else ''}{seconds:g}"


def parse_scene_params(entries: list[str]) -> dict[str, list[str]]:
    """``[BENCHMARK/]SCENE=A.toml[,B.toml]`` entries -> {key: [A.toml, B.toml]}, with key ``"Y1e8"`` for
    every benchmark of a scene and ``"static/Y1e8"`` for one benchmark's problems of it."""
    out = {}
    for entry in entries:
        key, sep, files = entry.partition("=")
        if not sep or not key or not files:
            raise ValueError(f"--scene-params expects [BENCHMARK/]SCENE=A.toml[,B.toml], got {entry!r}")
        if key in out:
            raise ValueError(f"--scene-params given twice for {key!r}")
        out[key] = [p for p in files.split(",") if p]
    return out


def scene_params_for(table: dict, benchmark: str, scene: str) -> list[str]:
    """A scene's parameter files in one benchmark (:func:`parse_scene_params` table): the scene's entry,
    then its ``benchmark/scene`` entry, merged in that order (later files win)."""
    return table.get(scene, []) + table.get(f"{benchmark}/{scene}", [])


def unmatched_scene_params(table: dict, benchmark_scenes) -> list[str]:
    """The table's keys that match none of the (benchmark, scene) pairs, sorted."""
    known = set()
    for benchmark, scene in benchmark_scenes:
        known.update((scene, f"{benchmark}/{scene}"))
    return sorted(set(table) - known)


def select_frames(
    time: np.ndarray,
    entries: np.ndarray,
    n_frames: int,
    span: float,
    return_targets: bool = False,
    end: float | None = None,
):
    """Snapshots to keep as video frames: the nearest in solver time to k * span / n_frames for k = 0, 1, ...
    up to the end time, and to the end time itself.

    Args:
        time (np.ndarray): (n,) cumulative solver time per log entry.
        entries (np.ndarray): Sorted log entries that have a snapshot.
        n_frames (int): Frames over ``span``.
        span (float): Solver time the frames are spread over (the budget; the solve's end if not finite).
        return_targets (bool): Also return each target time and the index picked for it.
        end (float | None): The last time to show, at most the solve's end (default: the solve's end),
            e.g. the span, for a solve that ran past it.

    Returns:
        np.ndarray: Sorted, unique indices into ``entries``; with ``return_targets``, (indices, targets)
        with one picked index per target time instead (duplicates kept).
    """
    entries = np.asarray(entries, dtype=np.int64)
    if entries.size == 0 or n_frames <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return (empty, np.zeros(0)) if return_targets else empty
    t = np.asarray(time, dtype=np.float64)[entries]
    end = float(time[-1]) if end is None else min(float(end), float(time[-1]))
    span = span if math.isfinite(span) and span > 0 else end
    targets = span * np.arange(n_frames + 1) / n_frames
    targets = np.append(targets[targets <= end], end)
    pick = np.abs(t[None, :] - targets[:, None]).argmin(axis=1)
    return (pick, targets) if return_targets else np.unique(pick)


def write_trajectory(group: h5py.Group, trajectory: dict) -> None:
    """Write one solve's frames (:func:`select_frames`, plus level crossings) under ``group``."""
    for key in ("iteration", "t", "t_target", "level_frames", "level_exact"):
        group.create_dataset(key, data=trajectory[key])
    group.create_dataset("x", data=np.asarray(trajectory["x"], dtype=np.float32))
    keys = ("requested_frames", "ring_slots", "stride", "device_bytes", "video_seconds", "playback_factor")
    group.attrs.update({k: trajectory[k] for k in keys})


# --- Results file ---


def check_format(f: h5py.File, path: str = "") -> None:
    if f.attrs.get("format") != FORMAT:
        raise ValueError(f"{path or f.filename}: not a benchmark results file (format != {FORMAT!r})")


def iter_cases(f: h5py.File):
    """Yield (benchmark, case, group) in file order (``benchmarks`` and ``cases`` attributes)."""
    for benchmark in f.attrs.get("benchmarks", []):
        b = f[str(benchmark)]
        for case in b.attrs.get("cases", []):
            yield str(benchmark), str(case), b[str(case)]


def _append_name(group: h5py.Group, attr: str, name: str) -> None:
    names = [str(n) for n in group.attrs.get(attr, [])]
    if name not in names:
        group.attrs[attr] = names + [name]


def require_case(f: h5py.File, benchmark: str, case: str, attrs: dict, levels=None) -> h5py.Group:
    """The ``/<benchmark>/<case>`` group with ``attrs``, recorded in the file's order attributes;
    ``levels`` sets the benchmark's levels (its group's ``levels`` attribute)."""
    _append_name(f, "benchmarks", benchmark)
    b = f.require_group(benchmark)
    if levels is not None:
        b.attrs["levels"] = np.asarray(levels, dtype=np.float64)
    _append_name(b, "cases", case)
    g = b.require_group(case)
    g.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return g


def write_solver(
    group: h5py.Group, solver: str, levels, summaries: list[dict], logs: list, trajectories: list | None = None
) -> h5py.Group:
    """Write one solver's samples of a case: per-sample arrays, crossings, aggregates and logs.

    Args:
        group (h5py.Group): The case group (:func:`require_case`).
        solver (str): Solver name.
        levels: The accuracy levels.
        summaries (list[dict]): Per sample: ``sample``, ``e0``, ``iterations``, ``sweeps``,
            ``solver_time``, ``final_relative_error``, ``termination``, ``deformation_energy``,
            ``crossings`` (:class:`LevelCrossings`), and optionally ``min_J``,
            ``reference_error_bound``, ``reference_relative_error_bound``, ``budget`` (the solve's time
            budget) and ``normalization`` (the time the sample's increments are divided by, T_N; with it,
            ``normalized_increment`` and its aggregates are written).
        logs (list): Per sample, (error, time, sweeps) arrays, optionally followed by a dict of further
            per-sample arrays written next to them (e.g. Newton's per-step records, ``newton/cg_iterations``).
        trajectories (list | None): Per sample, a trajectory dict (:func:`write_trajectory`), if any.
    """
    g = group.create_group(solver)
    g.attrs.update(solver=solver, n_samples=len(summaries))
    scalar_keys = ["sample", "e0", "iterations", "sweeps", "solver_time", "host_time", "final_relative_error",
                   "deformation_energy", "min_J", "reference_error_bound", "reference_relative_error_bound",
                   "budget", "normalization"]
    for key in scalar_keys:
        if summaries and all(key in s for s in summaries):
            g.create_dataset(key, data=np.array([s[key] for s in summaries]))
    g.create_dataset("termination", data=np.array([s["termination"] for s in summaries], dtype=h5py.string_dtype()))
    L = len(levels)
    stacked = {
        name: np.array([getattr(s["crossings"], name) for s in summaries]).reshape(len(summaries), L)
        for name in ["time", "iteration", "increment", "reached", "skipped"]
    }
    g.create_dataset("crossing_time", data=stacked["time"])
    g.create_dataset("crossing_iteration", data=stacked["iteration"])
    g.create_dataset("increment", data=stacked["increment"])
    g.create_dataset("reached", data=stacked["reached"])
    g.create_dataset("skipped", data=stacked["skipped"])
    if summaries and all("reference_relative_error_bound" in s for s in summaries):
        # Levels the reference resolves: 1 - level at least the reference's relative accuracy
        bound = np.array([s["reference_relative_error_bound"] for s in summaries])[:, None]
        g.create_dataset("resolvable", data=(1.0 - np.asarray(levels)[None, :]) >= bound)
    aggregated = [("increment", stacked["increment"]), ("crossing_time", stacked["time"])]
    if summaries and all("normalization" in s for s in summaries):
        # Increments over the sample's T_N, e.g. a Newton solve's reached increments sum to 1
        normalization = np.array([s["normalization"] for s in summaries], dtype=np.float64)[:, None]
        normalized = stacked["increment"] / normalization
        g.create_dataset("normalized_increment", data=normalized)
        aggregated.append(("normalized_increment", normalized))
    for name, values in aggregated:
        a = g.create_group(f"aggregate/{name}")
        for stat, v in aggregate(values, stacked["reached"]).items():
            a.create_dataset(stat, data=v)
    for s, (error, time, sweeps, *extra) in zip(summaries, logs):
        lg = g.create_group(f"logs/{int(s['sample']):06d}")
        lg.create_dataset("error", data=error)
        lg.create_dataset("time", data=time)
        lg.create_dataset("sweeps", data=sweeps)
        for name, values in (extra[0] if extra else {}).items():
            values = np.asarray(values)
            if values.dtype.kind == "U":
                lg.create_dataset(name, data=values.astype(object), dtype=h5py.string_dtype())
            else:
                lg.create_dataset(name, data=values)
    for s, trajectory in zip(summaries, trajectories or []):
        write_trajectory(g.create_group(f"trajectories/{int(s['sample']):06d}"), trajectory)
    return g


import unittest


class TestLevelCrossings(unittest.TestCase):
    levels = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)

    def test_gradual(self):
        # One level per iteration, then a final iteration crossing 95% and 99% at once
        r = np.array([1.0, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.009])
        t = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0])
        c = level_crossings(r, t, self.levels)
        np.testing.assert_array_equal(c.iteration, [1, 2, 3, 4, 5, 6, 7, 7])
        np.testing.assert_array_equal(c.time, [1, 2, 4, 5, 7, 8, 10, 10])
        np.testing.assert_array_equal(c.skipped, [0, 0, 0, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(c.increment, [1, 1, 2, 1, 2, 1, 0, 2])
        self.assertTrue(c.reached.all())
        self.assertEqual(np.sum(c.increment), c.time[-1])

    def test_skipped_levels(self):
        # Newton-like: the first iteration (3 s) reaches 80%; 50-70% get nothing
        r = np.array([1.0, 0.19, 0.02, 0.005])
        t = np.array([0.0, 3.0, 4.0, 4.5])
        c = level_crossings(r, t, self.levels)
        np.testing.assert_array_equal(c.iteration, [1, 1, 1, 1, 2, 2, 2, 3])
        np.testing.assert_array_equal(c.skipped, [1, 1, 1, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(c.increment, [0, 0, 0, 3, 0, 0, 1, 0.5])
        self.assertTrue(c.reached.all())

    def test_unreached_and_non_monotone(self):
        # The error rises again after crossing 60%: crossings are first crossings; 70%+ never reached
        r = np.array([1.0, 0.7, 0.38, 0.6, 0.31])
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        c = level_crossings(r, t, self.levels)
        np.testing.assert_array_equal(c.reached, [1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(c.iteration[:2], [2, 2])
        np.testing.assert_array_equal(c.increment[:2], [0, 2])
        self.assertTrue(np.isnan(c.increment[2:]).all() and np.isnan(c.time[2:]).all())
        self.assertTrue((c.iteration[2:] == -1).all())
        self.assertFalse(c.skipped[1])  # 70% isn't reached, so 60% is the highest crossed

    def test_time_limit(self):
        # The second iteration ends past the 2.5 s budget: its crossings don't count
        c = level_crossings(np.array([1.0, 0.45, 0.005]), np.array([0.0, 1.0, 3.0]), self.levels, 2.5)
        np.testing.assert_array_equal(c.reached, [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(c.increment[0], 1.0)
        c = level_crossings(np.array([1.0, 0.45, 0.005]), np.array([0.0, 1.0, 3.0]), self.levels, 3.0)
        self.assertTrue(c.reached.all())

    def test_non_finite(self):
        c = level_crossings(np.array([1.0, np.nan]), np.array([0.0, 1.0]), self.levels)
        self.assertFalse(c.reached.any())

    def test_aggregate(self):
        values = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, np.nan]])
        reached = np.array([[1, 1], [1, 1], [1, 0]], dtype=bool)
        a = aggregate(values, reached)
        np.testing.assert_array_equal(a["count"], [3, 2])
        np.testing.assert_allclose(a["mean"], [3.0, 1.0])
        np.testing.assert_allclose(a["min"], [1.0, 0.0])
        np.testing.assert_allclose(a["max"], [5.0, 2.0])
        np.testing.assert_allclose(a["median"], [3.0, 1.0])
        np.testing.assert_allclose(a["std"], [np.std([1, 3, 5]), 1.0])
        a = aggregate(values, np.zeros_like(reached))
        self.assertTrue(np.isnan(a["mean"]).all())
        np.testing.assert_array_equal(a["count"], [0, 0])

    def test_select_frames(self):
        time = np.arange(101) * 0.1  # 10 s over 100 iterations
        entries = np.arange(0, 101, 4)  # a snapshot every 4 iterations
        # 5 frames over a 20 s budget: targets 0, 4, 8 s (12+ are after the end) and the end, 10 s
        pick = select_frames(time, entries, 5, 20.0)
        np.testing.assert_array_equal(entries[pick], [0, 40, 80, 100])
        # Over the solve itself (no finite budget): 0, 2, 4, 6, 8, 10 s -> nearest snapshots
        pick = select_frames(time, entries, 5, math.inf)
        np.testing.assert_array_equal(entries[pick], [0, 20, 40, 60, 80, 100])
        # Few snapshots: duplicates collapse
        np.testing.assert_array_equal(select_frames(time, np.array([0, 100]), 10, 10.0), [0, 1])
        # A solve that ran past its span: frames up to the span only
        np.testing.assert_array_equal(entries[select_frames(time, entries, 5, 4.0)], [0, 8, 16, 24, 32, 40, 100])
        np.testing.assert_array_equal(entries[select_frames(time, entries, 5, 4.0, end=4.0)], [0, 8, 16, 24, 32, 40])
        self.assertEqual(select_frames(time, np.array([], dtype=np.int64), 5, 1.0).size, 0)

    def test_parse_levels(self):
        self.assertEqual(parse_levels("0.9, 0.5,0.99"), (0.5, 0.9, 0.99))
        with self.assertRaises(ValueError):
            parse_levels("0.5,1.0")
        table = parse_levels_options(["0.5,0.9", "static=0.5,0.99,0.999"])
        self.assertEqual(levels_for(table, "static"), (0.5, 0.99, 0.999))
        self.assertEqual(levels_for(table, "dynamic_h0.01"), (0.5, 0.9))
        self.assertEqual(levels_for({}, "static"), STATIC_LEVELS)
        self.assertEqual(levels_for({}, "dynamic_h0.01"), DYNAMIC_LEVELS)
        with self.assertRaises(ValueError):
            parse_levels_options(["static=0.5", "static=0.9"])

    def test_normalization_time(self):
        c = level_crossings(np.array([1.0, 0.45, 0.05, 0.02]), np.array([0.0, 1.0, 3.0, 5.0]), (0.5, 0.9, 0.99))
        self.assertEqual(normalization_time(c), (3.0, 1))  # 90% at 3 s; 99% never
        self.assertTrue(math.isnan(normalization_time(level_crossings(np.ones(2), np.arange(2.0), (0.5,)))[0]))

    def test_budget_precedence(self):
        budgets = parse_budget_options(["static=5", "static/Y1e8=10"])
        self.assertEqual(fixed_budget(budgets, "static", "Y1e6"), (5.0, "--budget static=5"))
        self.assertEqual(fixed_budget(budgets, "static", "Y1e8")[0], 10.0)  # the case's own entry first
        self.assertEqual(fixed_budget(budgets, "dynamic_h0.01", "Y1e6"), (None, ""))  # Newton's times
        self.assertEqual(fixed_budget(parse_budget_options(["2"]), "dynamic_h0.01", "Y1e6")[0], 2.0)
        with self.assertRaises(ValueError):
            parse_budget_options(["static=0"])
        with self.assertRaises(ValueError):
            parse_budget_options(["5", "5"])

    def test_scene_params(self):
        table = parse_scene_params(["Y1e8=a.toml,b.toml", "static/Y1e8=c.toml", "static/het=c.toml"])
        self.assertEqual(scene_params_for(table, "static", "Y1e8"), ["a.toml", "b.toml", "c.toml"])
        self.assertEqual(scene_params_for(table, "dynamic_h0.01", "Y1e8"), ["a.toml", "b.toml"])
        self.assertEqual(scene_params_for(table, "static", "het"), ["c.toml"])
        self.assertEqual(scene_params_for(table, "dynamic_h0.01", "het"), [])
        pairs = [(b, s) for b in ("static", "dynamic_h0.01") for s in ("Y1e8", "het")]
        self.assertEqual(unmatched_scene_params(table, pairs), [])
        self.assertEqual(unmatched_scene_params(table, pairs[2:]), ["static/Y1e8", "static/het"])
        for bad in (["Y1e8"], ["=a.toml"], ["Y1e8="], ["static/Y1e8=a.toml", "static/Y1e8=b.toml"]):
            with self.assertRaises(ValueError):
                parse_scene_params(bad)

    def test_file(self):
        import os
        import tempfile

        levels = (0.5, 0.99)
        runs = {"Newton": [[1.0, 0.4, 0.005], [1.0, 0.3, 0.2]], "VBD": [[1.0, 0.6, 0.45]]}
        times = [0.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "results.h5")
            with h5py.File(path, "w") as f:
                f.attrs.update(format=FORMAT, version=VERSION, levels=levels)
                case = require_case(f, "static", "Y1e6", {"scene": "Y1e6", "budget": 10.0}, levels=levels)
                for solver, errors in runs.items():
                    summaries, logs = [], []
                    for s, r in enumerate(errors):
                        c = level_crossings(np.array(r), np.array(times), levels)
                        summaries.append(dict(sample=s, e0=1.0, iterations=2, sweeps=0, solver_time=3.0,
                                              final_relative_error=r[-1], termination="target",
                                              deformation_energy=1.0, crossings=c, budget=10.0,
                                              normalization=[3.0, 2.0][s]))
                        extra = {"newton/cg_iterations": [40, 60], "newton/status": ["stepped", "stepped"]}
                        extra = extra if solver == "Newton" else {}
                        logs.append((np.array(r), np.array(times), np.zeros(3, dtype=np.int64), extra))
                    write_solver(case, solver, levels, summaries, logs)
            with h5py.File(path, "r") as f:
                cases = list(iter_cases(f))
                self.assertEqual([(b, c) for b, c, _ in cases], [("static", "Y1e6")])
                g = cases[0][2]["Newton"]
                np.testing.assert_array_equal(g["aggregate/increment/count"][()], [2, 1])
                np.testing.assert_allclose(g["aggregate/increment/mean"][()], [2.0, 1.0])
                # Over each sample's T_N (3 s and 2 s): Newton's reached increments sum to 1 per sample
                np.testing.assert_allclose(g["normalized_increment"][()], [[2 / 3, 1 / 3], [1.0, np.nan]])
                np.testing.assert_allclose(g["aggregate/normalized_increment/mean"][()], [5 / 6, 1 / 3])
                np.testing.assert_array_equal(benchmark_levels(f, "static"), levels)
                np.testing.assert_array_equal(g["logs/000001/newton/cg_iterations"][()], [40, 60])
                self.assertEqual(list(g["logs/000001/newton/status"].asstr()[()]), ["stepped", "stepped"])
                self.assertNotIn("newton", cases[0][2]["VBD/logs/000000"])
                self.assertEqual(list(g["termination"].asstr()[()]), ["target", "target"])


if __name__ == "__main__":
    unittest.main()
