"""
Iteration counts to each accuracy level of a benchmark results file (``benchmark.py``), as tables: one per
benchmark (static, dynamic_h0.01, ...), one row per case (scene) and solver, one column per level. Where the
bar charts (``benchmark_plot.py``) show time, these show work, so the solvers' behavior in the tail levels
(e.g. a solver that stalls between 99% and 99.9%) reads directly.

Each cell is the median [25th-75th percentile] count over the samples that reached the level, with n/N
when not all of them did:

- VBD family (VBD, AAAVBD, Chebyshev, Anderson): VBD sweeps (an Anderson step of ``vbd_iters_per_anderson_iter``
  sweeps counts them all);
- Newton: Newton steps (accepted) and, in parentheses, the median cumulative CG iterations of every step
  attempted up to the crossing, rejected ones included.

Counts are within ``--budget-factor F`` x T_N (default: the run's factor, 2 in the full run, so the counts
cover the whole logs that ``benchmark_curves.py`` draws; 1 matches the bar charts). In each column, the
VBD-family solver with the fewest median sweeps among those that reached the level on at least half the
samples is in bold. Dynamic levels the reference can't resolve for most samples read "beyond ref." and the
samples it can't resolve are left out elsewhere, as in the bar charts (``benchmark_plot.unresolvable_levels``).

Usage:
    python -m python.tools.sim.benchmark_iterations results.h5 -o out/tables/iterations \
        [--benchmarks static dynamic_h0.01] [--solvers VBD,AAAVBD,...] [--budget-factor 2]

Writes ``<prefix>.md`` and ``<prefix>.tex`` (booktabs and multirow tables).
"""

import argparse
import math
import os
import unittest

import h5py
import numpy as np

from . import benchmark_results as results
from .benchmark_plot import SOLVER_NAMES, benchmark_title, resolvable_samples, unresolvable_levels


def crossing_counts(
    error: np.ndarray, time: np.ndarray, sweeps: np.ndarray, levels, limit: float,
    cg_iterations: np.ndarray | None = None, status: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Work done until each level's first crossing within ``limit`` seconds of solver time.

    Args:
        error (np.ndarray): (n,) logged errors (entry 0 = the initial state).
        time (np.ndarray): (n,) cumulative solver times (s).
        sweeps (np.ndarray): (n,) cumulative VBD sweeps (0 for Newton).
        levels: Increasing error reductions in (0, 1).
        limit (float): The time budget (s).
        cg_iterations (np.ndarray | None): Newton's CG iterations per step attempted.
        status (np.ndarray | None): Newton's status per step attempted; log entry j + 1 follows the j-th
            ``stepped`` one.

    Returns:
        tuple: (L,) counts, -1 where not reached: sweeps, or Newton's steps when ``cg_iterations`` is
        given; and for Newton, (L,) cumulative CG iterations (else None).
    """
    c = results.level_crossings(results.relative_error(error), time, levels, limit)
    if cg_iterations is None:
        return np.where(c.reached, np.asarray(sweeps)[np.maximum(c.iteration, 0)], -1), None
    stepped = np.nonzero(np.asarray(status) == "stepped")[0]
    cg = np.cumsum(np.asarray(cg_iterations, dtype=np.int64))
    cg_at = [cg[stepped[it - 1]] if it > 0 else 0 for it in c.iteration]
    return c.iteration.copy(), np.where(c.reached, cg_at, -1)


def solver_counts(g: h5py.Group, solver: str, levels, factor: float) -> tuple[np.ndarray, np.ndarray | None]:
    """(S, L) counts of a case's solver (:func:`crossing_counts`), within ``factor`` x its normalization
    and its budget (the case's fixed budget for fixed-budget cases)."""
    sg = g[solver]
    n = len(sg["sample"])
    budget = np.asarray(sg["budget"], dtype=np.float64) if "budget" in sg else np.full(n, math.inf)
    if "budget" not in g.attrs and "normalization" in sg:
        budget = np.minimum(budget, factor * np.asarray(sg["normalization"], dtype=np.float64))
    counts, cgs = np.full((n, len(levels)), -1, dtype=np.int64), None
    for row, sample in enumerate(np.asarray(sg["sample"])):
        log = sg[f"logs/{int(sample):06d}"]
        newton = log["newton"] if "newton" in log else None
        count, cg = crossing_counts(
            log["error"][()], log["time"][()], log["sweeps"][()], levels, budget[row],
            newton["cg_iterations"][()] if newton is not None else None,
            newton["status"].asstr()[()] if newton is not None else None,
        )
        counts[row] = count
        if cg is not None:
            cgs = np.full((n, len(levels)), -1, dtype=np.int64) if cgs is None else cgs
            cgs[row] = cg
    return counts, cgs


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.percentile(values, [25, 50, 75]))


def table_rows(f: h5py.File, benchmark: str, solvers: list[str], factor: float) -> tuple[list, list]:
    """The benchmark's levels and rows (case label, solver name, cells); a cell is a dict with ``text``
    (plain), ``bold`` and ``note`` or None."""
    levels = results.benchmark_levels(f, benchmark)
    rows = []
    for b, case, g in results.iter_cases(f):
        if b != benchmark:
            continue
        present = [s for s in solvers if s in g]
        unresolvable = unresolvable_levels(g, present, len(levels))
        resolvable = resolvable_samples(g, present)
        cells, medians = {}, {}
        for solver in present:
            counts, cgs = solver_counts(g, solver, levels, factor)
            counted = resolvable if resolvable is not None else np.ones(counts.shape, dtype=bool)
            row = []
            for j in range(len(levels)):
                if unresolvable[j]:
                    row.append({"text": "beyond ref.", "bold": False})
                    continue
                n = int(counted[:, j].sum())
                reached = (counts[:, j] >= 0) & counted[:, j]
                k = int(reached.sum())
                if k == 0:
                    row.append({"text": f"- (0/{n})", "bold": False})
                    continue
                q25, q50, q75 = quartiles(counts[reached, j])
                if cgs is not None:
                    text = f"{q50:.0f} ({np.median(cgs[reached, j]):.0f} CG)"
                else:
                    text = f"{q50:.0f} [{q25:.0f}-{q75:.0f}]"
                    if 2 * k >= n:
                        medians.setdefault(j, []).append((q50, solver))
                if k < n:
                    text += f" ({k}/{n})"
                row.append({"text": text, "bold": False})
            cells[solver] = row
        for j, candidates in medians.items():
            best = min(q for q, _ in candidates)
            for q, solver in candidates:
                cells[solver][j]["bold"] = q == best
        label = str(g.attrs.get("label", case))
        rows += [(label, SOLVER_NAMES.get(s, s), cells[s]) for s in present]
    return levels, rows


def level_header(level: float) -> str:
    return f"{100 * level:g}%"


def markdown(benchmark: str, levels, rows, factor: float) -> str:
    head = ["Scene", "Solver"] + [level_header(v) for v in levels]
    lines = [f"## {benchmark_title(benchmark)}", "",
             "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    previous = None
    for label, solver, cells in rows:
        scene = label if label != previous else ""
        previous = label
        texts = [f"**{c['text']}**" if c["bold"] else c["text"] for c in cells]
        lines.append("| " + " | ".join([scene, solver] + texts) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    """Escapes % and _, and makes hyphens (quartile ranges, unreached cells) en dashes."""
    return text.replace("%", r"\%").replace("_", r"\_").replace("-", "--")


def latex(benchmark: str, levels, rows, factor: float) -> str:
    n_cols = len(levels)
    out = [r"\begin{table}", r"\centering", r"\small",
           rf"\caption{{{benchmark_title(benchmark)}: iterations to each error reduction within {factor:g}"
           r" $\times$ $T_N$ (median [25th--75th percentile]; VBD sweeps, Newton steps and CG iterations).}",
           r"\begin{tabular}{ll" + "r" * n_cols + "}", r"\toprule",
           "Scene & Solver & " + " & ".join(latex_escape(level_header(v)) for v in levels) + r" \\", r"\midrule"]
    groups = {}
    for label, _, _ in rows:
        groups[label] = groups.get(label, 0) + 1
    previous = None
    for label, solver, cells in rows:
        if previous is not None and label != previous:
            out.append(r"\midrule")
        scene = rf"\multirow{{{groups[label]}}}{{*}}{{{latex_escape(label)}}}" if label != previous else ""
        previous = label
        texts = [rf"\textbf{{{latex_escape(c['text'])}}}" if c["bold"] else latex_escape(c["text"]) for c in cells]
        out.append(" & ".join([scene, latex_escape(solver)] + texts) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Iteration counts to each accuracy level of a benchmark results file.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True, help="Output prefix: writes <prefix>.md and .tex.")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmarks to tabulate (default: all in the file).")
    parser.add_argument("--solvers", default=None, help="Comma-separated solvers (default: the file's).")
    parser.add_argument("--budget-factor", type=float, default=None, dest="budget_factor",
                        help="Count within this factor x Newton's T_N, at most the run's (default: the run's factor).")
    args = parser.parse_args()
    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        run = float(f.attrs.get("budget_factor", 1.0))
        factor = run if args.budget_factor is None else args.budget_factor
        if not 0 < factor <= run * (1 + 1e-9):
            raise SystemExit(f"--budget-factor must be in (0, {run:g}] (the run's factor), got {factor:g}")
        solvers = args.solvers.split(",") if args.solvers else [str(s) for s in f.attrs["solvers"]]
        benchmarks = args.benchmarks or [str(b) for b in f.attrs["benchmarks"]]
        md = [f"# Iterations to each error reduction, within {factor:g} x T_N", "",
              "Median [25th-75th percentile] over the samples that reached the level; n/N when not all did. "
              "VBD family: sweeps; Newton: steps (cumulative CG iterations). Bold: the fewest sweeps in the VBD "
              "family among the solvers that reached the level on at least half the samples.", ""]
        tex = ["% Needs \\usepackage{booktabs, multirow}", ""]
        for benchmark in benchmarks:
            levels, rows = table_rows(f, benchmark, solvers, factor)
            md.append(markdown(benchmark, levels, rows, factor))
            tex.append(latex(benchmark, levels, rows, factor))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for ext, text in (("md", "\n".join(md)), ("tex", "\n".join(tex))):
        path = f"{args.output}.{ext}"
        with open(path, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"Wrote {path}")


class TestCrossingCounts(unittest.TestCase):
    def test_sweeps(self):
        error, time, sweeps = np.array([1.0, 0.6, 0.4, 0.05]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 2, 4, 6])
        counts, cg = crossing_counts(error, time, sweeps, (0.5, 0.9, 0.99), 10.0)
        np.testing.assert_array_equal(counts, [4, 6, -1])
        self.assertIsNone(cg)
        # Crossings after the budget don't count
        np.testing.assert_array_equal(crossing_counts(error, time, sweeps, (0.5, 0.9), 2.5)[0], [4, -1])

    def test_newton(self):
        # Steps: stepped (10 CG), rejected (20), stepped (30), stepped (40): log entries 1, 2, 3
        error, time = np.array([1.0, 0.6, 0.05, 0.001]), np.array([0.0, 1.0, 2.0, 3.0])
        steps, cg = crossing_counts(error, time, np.zeros(4), (0.5, 0.9, 0.99), 10.0, np.array([10, 20, 30, 40]),
                                    np.array(["stepped", "line_search_failed", "stepped", "stepped"]))
        np.testing.assert_array_equal(steps, [2, 2, 3])
        np.testing.assert_array_equal(cg, [60, 60, 100])


if __name__ == "__main__":
    main()
