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

    /                                attrs: format, version, levels, budget_factor, budget_statistic,
                                     budget_source, dataset, device, git_commit, date, sim_params,
                                     newton_cg_* (Newton's CG limits)
    /StaticSolveSpec                 the run's spec (static.StaticSolveSpec), attributes
    /dataset_attrs                   the dataset's scalar attributes
    /<benchmark>/<case>              attrs: benchmark, problem, h, scene, label, energy, set, budget,
                                     budget_source, scene_params
    /<benchmark>/<case>/<solver>     attrs: solver, n_samples
        sample, e0, iterations, sweeps, solver_time, final_relative_error, termination,
        deformation_energy, [min_J], [reference_error_bound, reference_relative_error_bound]   (S,)
        crossing_time, crossing_iteration, increment, reached, skipped, [resolvable]           (S, L)
        aggregate/{increment,crossing_time}/{count,mean,std,min,max,median,q25,q75}           (L,)
        logs/<sample:06d>/{error,time,sweeps}

``<benchmark>`` is ``static`` or ``dynamic_h<h>`` (e.g. ``dynamic_h0.0001``). A ``<case>`` is one x-axis
entry of the benchmark's figure: a scene, or ``<scene>_E<E>`` when the run holds several energy levels
of a scene. Times are seconds of solver time. NaN marks unreached levels (``reached`` False); an
unreached level's ``crossing_iteration`` is -1. Aggregates are over the samples that reached the
level (``count``); a skipped level counts with a zero increment.
"""

from dataclasses import dataclass
import math

import h5py
import numpy as np

FORMAT = "pbat-benchmark-results"
VERSION = 1
DEFAULT_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
STATISTICS = ("count", "mean", "std", "min", "max", "median", "q25", "q75")
BUDGET_STATISTICS = ("median", "mean")


def parse_levels(text: str) -> tuple[float, ...]:
    """Comma-separated error reductions, e.g. ``"0.5,0.9,0.99"``, sorted and checked to lie in (0, 1)."""
    levels = tuple(sorted({float(v) for v in text.split(",") if v.strip()}))
    if not levels or levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError(f"Accuracy levels must lie in (0, 1), got {text!r}")
    return levels


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


def budgets_from_pilot(
    path: str, factor: float, statistic: str = "median", solver: str = "Newton"
) -> dict[tuple[str, str], dict]:
    """Time budgets per (benchmark, case): ``factor`` times a pilot run's time to its highest level.

    Args:
        path (str): A results file holding ``solver``'s solves for the cases (e.g. a Newton-only run
            with a generous time cap).
        factor (float): Multiple of the statistic.
        statistic (str): ``median`` or ``mean`` of ``solver``'s crossing times of the pilot's highest
            level, over the samples that reached it.
        solver (str): The reference solver.

    Returns:
        dict: (benchmark, case) -> {budget, reference_time, reached, n_samples, level}. Cases where no
        sample reached the level are left out.
    """
    if statistic not in BUDGET_STATISTICS:
        raise ValueError(f"Budget statistic must be one of {BUDGET_STATISTICS}, got {statistic!r}")
    out = {}
    with h5py.File(path, "r") as f:
        check_format(f, path)
        level = float(np.asarray(f.attrs["levels"])[-1])
        for benchmark, case, group in iter_cases(f):
            if solver not in group:
                continue
            g = group[solver]
            reached = np.asarray(g["reached"])[:, -1]
            if not reached.any():
                continue
            t = np.asarray(g["crossing_time"])[reached, -1]
            reference = float(np.median(t) if statistic == "median" else np.mean(t))
            out[(benchmark, case)] = {
                "budget": factor * reference,
                "reference_time": reference,
                "reached": int(reached.sum()),
                "n_samples": int(reached.size),
                "level": level,
            }
    return out


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


def require_case(f: h5py.File, benchmark: str, case: str, attrs: dict) -> h5py.Group:
    """The ``/<benchmark>/<case>`` group with ``attrs``, recorded in the file's order attributes."""
    _append_name(f, "benchmarks", benchmark)
    b = f.require_group(benchmark)
    _append_name(b, "cases", case)
    g = b.require_group(case)
    g.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return g


def write_solver(group: h5py.Group, solver: str, levels, summaries: list[dict], logs: list) -> h5py.Group:
    """Write one solver's samples of a case: per-sample arrays, crossings, aggregates and logs.

    Args:
        group (h5py.Group): The case group (:func:`require_case`).
        solver (str): Solver name.
        levels: The accuracy levels.
        summaries (list[dict]): Per sample: ``sample``, ``e0``, ``iterations``, ``sweeps``,
            ``solver_time``, ``final_relative_error``, ``termination``, ``deformation_energy``,
            ``crossings`` (:class:`LevelCrossings`), and optionally ``min_J``,
            ``reference_error_bound`` and ``reference_relative_error_bound``.
        logs (list): Per sample, (error, time, sweeps) arrays.
    """
    g = group.create_group(solver)
    g.attrs.update(solver=solver, n_samples=len(summaries))
    scalar_keys = ["sample", "e0", "iterations", "sweeps", "solver_time", "final_relative_error",
                   "deformation_energy", "min_J", "reference_error_bound", "reference_relative_error_bound"]
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
    for name, values in [("increment", stacked["increment"]), ("crossing_time", stacked["time"])]:
        a = g.create_group(f"aggregate/{name}")
        for stat, v in aggregate(values, stacked["reached"]).items():
            a.create_dataset(stat, data=v)
    for s, (error, time, sweeps) in zip(summaries, logs):
        lg = g.create_group(f"logs/{int(s['sample']):06d}")
        lg.create_dataset("error", data=error)
        lg.create_dataset("time", data=time)
        lg.create_dataset("sweeps", data=sweeps)
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

    def test_parse_levels(self):
        self.assertEqual(parse_levels("0.9, 0.5,0.99"), (0.5, 0.9, 0.99))
        with self.assertRaises(ValueError):
            parse_levels("0.5,1.0")

    def test_file_and_budgets(self):
        import os
        import tempfile

        levels = (0.5, 0.99)
        runs = {"Newton": [[1.0, 0.4, 0.005], [1.0, 0.3, 0.2]], "VBD": [[1.0, 0.6, 0.45]]}
        times = [0.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "results.h5")
            with h5py.File(path, "w") as f:
                f.attrs.update(format=FORMAT, version=VERSION, levels=levels)
                case = require_case(f, "static", "Y1e6", {"scene": "Y1e6", "budget": 10.0})
                for solver, errors in runs.items():
                    summaries, logs = [], []
                    for s, r in enumerate(errors):
                        c = level_crossings(np.array(r), np.array(times), levels)
                        summaries.append(dict(sample=s, e0=1.0, iterations=2, sweeps=0, solver_time=3.0,
                                              final_relative_error=r[-1], termination="target",
                                              deformation_energy=1.0, crossings=c))
                        logs.append((np.array(r), np.array(times), np.zeros(3, dtype=np.int64)))
                    write_solver(case, solver, levels, summaries, logs)
            with h5py.File(path, "r") as f:
                cases = list(iter_cases(f))
                self.assertEqual([(b, c) for b, c, _ in cases], [("static", "Y1e6")])
                g = cases[0][2]["Newton"]
                np.testing.assert_array_equal(g["aggregate/increment/count"][()], [2, 1])
                np.testing.assert_allclose(g["aggregate/increment/mean"][()], [2.0, 1.0])
                self.assertEqual(list(g["termination"].asstr()[()]), ["target", "target"])
            budgets = budgets_from_pilot(path, factor=2.0)
            # Newton reached 99% in one sample, at 3 s
            self.assertEqual(budgets[("static", "Y1e6")]["budget"], 6.0)
            self.assertEqual(budgets[("static", "Y1e6")]["reached"], 1)


if __name__ == "__main__":
    unittest.main()
