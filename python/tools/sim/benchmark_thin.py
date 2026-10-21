"""
A thin copy of a benchmark results file (``benchmark.py``), for sharing: every per-problem array, aggregate and
attribute, but no trajectories (video frames), and each solve's per-iteration log thinned to at most
``--max-entries`` entries. The full trampolines results are 8.8 GB, mostly frames (5.4 GB) and the static
VBD-family logs, which log every sweep (up to 250k entries per solve); the thin copy is about 40 MB.

The kept log entries are the start, the end, each level's first crossing (over the whole log, so re-scoring
at any budget factor finds the same crossings), and the entries nearest to ``--max-entries`` times evenly
spaced on a log-time axis. So the bar charts (``benchmark_plot.py``, any ``--budget-factor``) and the
iteration tables (``benchmark_iterations.py``) are identical to the full file's, and the convergence curves
(``benchmark_curves.py``) differ only between kept entries. Logs with at most ``--max-entries`` entries,
e.g. Newton's, are kept whole, and Newton's per-step records (``logs/<sample>/newton``) are copied as they are.

In the thin file, ``logs/<sample>/entry`` holds each kept entry's index in the full log, and
``crossing_iteration`` keeps its meaning, the full log's entry (iterations) at each crossing:
``np.searchsorted(entry, crossing_iteration)`` is its row in the thinned log. The root attribute ``thin``
describes the thinning. The file keeps the results format, so every script reads it as it reads a full one.

Usage:
    python -m python.tools.sim.benchmark_thin data/benchmarks/results/trampolines.h5 \
        -o data/benchmarks/results/trampolines-thin.h5 [--max-entries 1000] [--no-check]

``--no-check`` skips the final check that the bar aggregates (at 1 and at the run's budget factor) and the
iteration counts of the thin file equal the full file's.

Tests: ``python -m unittest python.tools.sim.benchmark_thin``.
"""

import argparse
import math
import os
import unittest

import h5py
import numpy as np

from . import benchmark_results as results

COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


def kept_entries(error: np.ndarray, time: np.ndarray, levels, max_entries: int) -> np.ndarray:
    """Sorted indices of the log entries a thin log keeps: all of them for a log of at most ``max_entries``
    entries, else the first, the last, each level's first crossing, and the entries nearest to ``max_entries``
    times evenly spaced on a log axis between the first positive time and the last."""
    n = len(error)
    if n <= max_entries:
        return np.arange(n)
    keep = [np.array([0, n - 1])]
    c = results.level_crossings(results.relative_error(error), time, levels)
    keep.append(c.iteration[c.reached])
    positive = time[time > 0]
    if positive.size:
        grid = np.geomspace(positive[0], time[-1], max_entries)
        i = np.clip(np.searchsorted(time, grid), 1, n - 1)
        # The nearer of the two entries around each grid time
        i -= (grid - time[i - 1]) < (time[i] - grid)
        keep.append(i)
    return np.unique(np.concatenate(keep))


def copy_attrs(src, dst) -> None:
    dst.attrs.update(dict(src.attrs))


def copy_solver(src: h5py.Group, dst: h5py.Group, levels, max_entries: int) -> int:
    """Copy a ``/<benchmark>/<case>/<solver>`` group without trajectories, its logs thinned. Returns the
    number of log entries kept."""
    copy_attrs(src, dst)
    kept = 0
    for name, item in src.items():
        if name in ("logs", "trajectories"):
            continue
        src.copy(item, dst, name=name)
    for sample, log in src["logs"].items():
        error, time = log["error"][()], log["time"][()]
        entry = kept_entries(error, time, levels, max_entries)
        lg = dst.create_group(f"logs/{sample}")
        lg.create_dataset("entry", data=entry, **COMPRESSION)
        for key in ("error", "time", "sweeps"):
            lg.create_dataset(key, data=log[key][()][entry], **COMPRESSION)
        if "newton" in log:
            log.copy(log["newton"], lg, name="newton")
        kept += entry.size
    return kept


def thin(source: str, output: str, max_entries: int = 1000) -> None:
    with h5py.File(source, "r") as f, h5py.File(output, "w") as out:
        results.check_format(f, source)
        copy_attrs(f, out)
        out.attrs["thin"] = (
            f"Thin copy of {os.path.basename(source)} (python -m python.tools.sim.benchmark_thin): no trajectories; "
            f"per-iteration logs thinned to at most {max_entries} entries (the start, the end, every level's first "
            "crossing, and log-spaced times); logs/<sample>/entry holds each kept entry's index in the full log"
        )
        for name in ("StaticSolveSpec", "dataset_attrs"):
            if name in f:
                f.copy(f[name], out, name=name)
        total = full = 0
        for benchmark, case, g in results.iter_cases(f):
            if benchmark not in out:
                copy_attrs(f[benchmark], out.create_group(benchmark))
            levels = results.benchmark_levels(f, benchmark)
            cg = out[benchmark].create_group(case)
            copy_attrs(g, cg)
            for solver, sg in g.items():
                total += copy_solver(sg, cg.create_group(solver), levels, max_entries)
                full += sum(log["error"].size for log in sg["logs"].values())
    print(f"Wrote {output}: {os.path.getsize(output) / 1e6:.1f} MB, {total} of {full} log entries kept")


def check(source: str, output: str) -> None:
    """The thin file's bar aggregates (at 1 and at the run's budget factor) and iteration counts equal the full
    file's. AssertionError otherwise."""
    from . import benchmark_iterations as iterations

    with h5py.File(source, "r") as f, h5py.File(output, "r") as t:
        run = float(f.attrs.get("budget_factor", 1.0))
        for benchmark, case, g in results.iter_cases(f):
            levels = results.benchmark_levels(f, benchmark)
            for solver in g:
                a, b = g[solver], t[f"{benchmark}/{case}/{solver}"]
                if "normalization" in a and "budget" not in g.attrs:
                    for factor in sorted({1.0, run}):
                        full, thinned = results.rescore(a, levels, factor), results.rescore(b, levels, factor)
                        for stat in results.STATISTICS:
                            np.testing.assert_array_equal(full[stat], thinned[stat],
                                                          f"{benchmark}/{case}/{solver} {stat} at {factor:g} x T_N")
                full_counts = iterations.solver_counts(g, solver, levels, run)
                thin_counts = iterations.solver_counts(t[f"{benchmark}/{case}"], solver, levels, run)
                for x, y in zip(full_counts, thin_counts):
                    if x is not None:
                        np.testing.assert_array_equal(x, y, f"{benchmark}/{case}/{solver} iteration counts")
    print("Check passed: bar aggregates and iteration counts equal the full file's")


def main():
    parser = argparse.ArgumentParser(description="Thin copy of a benchmark results file: no trajectories, thinned logs.")
    parser.add_argument("results", help="Full results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True, help="Thin results file to write.")
    parser.add_argument("--max-entries", type=int, default=1000, dest="max_entries",
                        help="Log-spaced entries per thinned log, besides the crossings and ends (default: %(default)s).")
    parser.add_argument("--no-check", action="store_true", dest="no_check",
                        help="Skip comparing the thin file's bar aggregates and iteration counts with the full file's.")
    args = parser.parse_args()
    if os.path.abspath(args.output) == os.path.abspath(args.results):
        raise SystemExit("The output would overwrite the full results file")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    thin(args.results, args.output, args.max_entries)
    if not args.no_check:
        check(args.results, args.output)


class TestKeptEntries(unittest.TestCase):
    def test_short_log_whole(self):
        np.testing.assert_array_equal(kept_entries(np.ones(5), np.arange(5.0), (0.5,), 10), np.arange(5))

    def test_keeps_ends_and_crossings(self):
        n = 10000
        time = np.linspace(0.0, 1.0, n)
        error = np.exp(-10.0 * time)  # e/e0 crosses 50% at t = ln 2 / 10
        levels = (0.5, 0.99)
        keep = kept_entries(error, time, levels, 100)
        c = results.level_crossings(results.relative_error(error), time, levels)
        self.assertTrue(set(c.iteration[c.reached]) <= set(keep))
        self.assertEqual((keep[0], keep[-1]), (0, n - 1))
        self.assertLessEqual(keep.size, 100 + 2 + len(levels))
        # Re-scoring the thinned log finds the same crossings at every budget
        for limit in (0.05, 0.2, math.inf):
            a = results.level_crossings(results.relative_error(error), time, levels, limit)
            b = results.level_crossings(results.relative_error(error[keep]), time[keep], levels, limit)
            np.testing.assert_array_equal(a.time, b.time)
            np.testing.assert_array_equal(keep[b.iteration[b.reached]], a.iteration[a.reached])


if __name__ == "__main__":
    main()
