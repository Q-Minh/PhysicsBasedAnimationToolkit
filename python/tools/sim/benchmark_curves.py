"""
Convergence curves of a benchmark results file (``benchmark.py``): the relative error e/e0 (log scale)
against solver time over the problem's normalization time T_N (Newton's time to its highest level, or the
fixed budget), one curve per solver, one panel per chosen problem. They show what the bar charts
(``benchmark_plot.py``) summarize per level: e.g. the VBD family's sub-linear tails, or a solver stalling
at a float32 floor. Run the benchmark with ``--budget-factor`` above 1 (e.g. 5) so the curves extend past
Newton's time.

- dotted horizontal lines: the benchmark's accuracy levels (e/e0 = 1 - level);
- dashed vertical line: t = T_N;
- shaded band (dynamic problems): errors below the reference's relative accuracy, which the reference
  can't resolve.

Usage:
    python -m python.tools.sim.benchmark_curves results.h5 -o out/figures/curves \
        [--problems static/Y1e6 dynamic_h0.01/Y1e7] [--sample 0] [--solvers VBD,AAAVBD,...] [--dpi 200]

Writes ``<prefix>.pdf`` and ``.png``: one panel per problem (default: the first case of each benchmark).
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import benchmark_results as results  # noqa: E402
from .benchmark_plot import SOLVER_NAMES, benchmark_title, case_label, solver_color  # noqa: E402


def sample_index(g: h5py.Group, sample: int) -> int:
    """Row of ``sample`` in a solver group's per-sample arrays."""
    rows = np.nonzero(np.asarray(g["sample"]) == sample)[0]
    if rows.size == 0:
        raise SystemExit(f"{g.name}: no sample {sample}")
    return int(rows[0])


def plot_problem(ax, f: h5py.File, benchmark: str, case: str, sample: int, solvers: list[str]) -> list[str]:
    """One panel: e/e0 against t / T_N for each solver of the case. Returns the solvers drawn."""
    g = f[f"{benchmark}/{case}"]
    levels = results.benchmark_levels(f, benchmark)
    drawn = []
    t_end = 1.0
    reference_bound = None
    for solver in solvers:
        if solver not in g:
            continue
        sg = g[solver]
        row = sample_index(sg, sample)
        normalization = float(sg["normalization"][row]) if "normalization" in sg else float(g.attrs["budget"])
        log = sg[f"logs/{sample:06d}"]
        e, t = np.asarray(log["error"]), np.asarray(log["time"])
        x = t / normalization
        ax.plot(x, e / e[0], color=solver_color(solver, solvers), linewidth=1.5, label=SOLVER_NAMES.get(solver, solver))
        t_end = max(t_end, float(x[-1]))
        if "reference_relative_error_bound" in sg:
            reference_bound = float(sg["reference_relative_error_bound"][row])
        drawn.append(solver)
    for level in levels:
        ax.axhline(1.0 - level, color="#BBBBBB", linewidth=0.6, linestyle=":", zorder=0)
        ax.text(t_end * 1.01, 1.0 - level, f"{100 * level:g}%", fontsize=6, color="#777777", va="center")
    ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    if reference_bound is not None:
        ax.axhspan(1e-12, reference_bound, color="#DDDDDD", alpha=0.6, zorder=0, linewidth=0)
    floor = min(1.0 - levels[-1], reference_bound or 1.0) * 0.1
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.5)
    ax.set_xlim(0.0, t_end)
    ax.set_xlabel("Solver time / T_N")
    ax.set_title(f"{benchmark_title(benchmark)}: {case_label(g, case).replace(chr(10), ', ')}, sample {sample}",
                 fontsize=9)
    ax.grid(True, which="major", color="#EEEEEE", linewidth=0.5)
    return drawn


def main():
    parser = argparse.ArgumentParser(description="Convergence curves (e/e0 against time / T_N) of a benchmark results file.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True, help="Output prefix: writes <prefix>.pdf and .png.")
    parser.add_argument("--problems", nargs="+", default=None, metavar="BENCHMARK/CASE",
                        help="Problems to draw, one panel each (default: the first case of each benchmark).")
    parser.add_argument("--sample", type=int, default=0, help="Sample index (default: %(default)s).")
    parser.add_argument("--solvers", default=None, help="Comma-separated solvers (default: the file's).")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    args = parser.parse_args()
    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        solvers = args.solvers.split(",") if args.solvers else [str(s) for s in f.attrs["solvers"]]
        problems = args.problems
        if not problems:
            first = {}
            for b, c, _ in results.iter_cases(f):
                first.setdefault(b, c)
            problems = [f"{b}/{c}" for b, c in first.items()]
        fig, axes = plt.subplots(1, len(problems), figsize=(4.6 * len(problems), 3.6), squeeze=False)
        drawn = []
        for ax, problem in zip(axes[0], problems):
            benchmark, _, case = problem.partition("/")
            if benchmark not in f or case not in f[benchmark]:
                raise SystemExit(f"{args.results} has no {problem}")
            drawn += [s for s in plot_problem(ax, f, benchmark, case, args.sample, solvers) if s not in drawn]
        axes[0][0].set_ylabel("Relative error e / e0")
        handles = [plt.Line2D([], [], color=solver_color(s, solvers), linewidth=1.5, label=SOLVER_NAMES.get(s, s))
                   for s in drawn]
        handles.append(matplotlib.patches.Patch(color="#DDDDDD", label="below the reference's accuracy"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        for ext in ("pdf", "png"):
            path = f"{args.output}.{ext}"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"Wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
