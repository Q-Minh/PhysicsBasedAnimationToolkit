"""
Convergence curves of a benchmark results file (``benchmark.py``): the relative error e/e0 (log scale)
against solver time over the problem's normalization time T_N (Newton's time to its highest level, or the
fixed budget), on a log time axis by default (``--xscale``), one curve per solver, one panel per chosen
problem. On a linear time axis, a solver much slower than the others stretches it until the fast solvers'
curves look alike; the log axis gives each decade of time the same width, and starts each curve at its first
logged iterate (t = 0 has no place on it). They show what the bar charts
(``benchmark_plot.py``) summarize per level: e.g. the VBD family's sub-linear tails, or a solver stalling
at a float32 floor. Each curve is the solve's whole log, up to the run's budget (``benchmark.py
--budget-factor``, 2 x T_N in the full run, ``configs/benchmark/trampolines.toml``), past the 1 x T_N that the
bar charts score by default (``benchmark_plot.py --budget-factor``), from the same results file.

- dotted horizontal lines: the benchmark's accuracy levels (e/e0 = 1 - level);
- dashed vertical line: t = T_N;
- grey band (dynamic problems): errors below the reference's relative accuracy, which the reference
  can't resolve.

``--all-samples`` aggregates a problem's samples instead of drawing one: per solver, a translucent band spans
the min and max of e/e0 over the samples at each time t / T_N (each sample on its own T_N), and a line their
median. A sample's error at time t is that of its last logged iterate at or before t (1 before its first),
held after its solve ends, i.e. the error it reached by t. The grey band is darker below every sample's
reference accuracy and lighter where only some samples' references can't resolve the error.

Usage:
    python -m python.tools.sim.benchmark_curves results.h5 -o out/figures/curves \
        [--problems static/Y1e6 dynamic_h0.01/Y1e7] [--sample 0 | --all-samples] [--solvers VBD,AAAVBD,...] \
        [--xscale log] [--dpi 200]

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
from .benchmark_plot import SOLVER_NAMES, benchmark_title, case_label, solver_color, use_paper_fonts  # noqa: E402


def sample_index(g: h5py.Group, sample: int) -> int:
    """Row of ``sample`` in a solver group's per-sample arrays."""
    rows = np.nonzero(np.asarray(g["sample"]) == sample)[0]
    if rows.size == 0:
        raise SystemExit(f"{g.name}: no sample {sample}")
    return int(rows[0])


def solve_curve(g: h5py.Group, solver: str, sample: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, e/e0) of one solve's log, x = t / its normalization (T_N or the fixed budget)."""
    sg = g[solver]
    row = sample_index(sg, sample)
    normalization = float(sg["normalization"][row]) if "normalization" in sg else float(g.attrs["budget"])
    log = sg[f"logs/{sample:06d}"]
    e, t = np.asarray(log["error"], dtype=np.float64), np.asarray(log["time"], dtype=np.float64)
    return t / normalization, results.relative_error(e)


def error_reached(x: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """e/e0 at each ``grid`` time: that of the last iterate at or before it (1 before the first, the last
    one's after the solve ends)."""
    i = np.searchsorted(x, grid, side="right") - 1
    return np.where(i >= 0, e[np.maximum(i, 0)], 1.0)


def plot_problem(
    ax, f: h5py.File, benchmark: str, case: str, sample: int | None, solvers: list[str], xscale: str = "log"
) -> list[str]:
    """One panel: e/e0 against t / T_N (``xscale`` "log" or "linear") for each solver of the case, for one
    ``sample``, or over all of them (None: min-max band and median). Returns the solvers drawn."""
    g = f[f"{benchmark}/{case}"]
    levels = results.benchmark_levels(f, benchmark)
    present = [s for s in solvers if s in g]
    samples = [sample] if sample is not None else [int(v) for v in np.asarray(g[present[0]]["sample"])]
    curves = {s: [solve_curve(g, s, k) for k in samples] for s in present}
    first = [x[1] for runs in curves.values() for x, _ in runs if x.size > 1]
    t_start = min([1.0] + first)  # the earliest first iterate, for the log axis
    t_end = max([1.0] + [float(x[-1]) for runs in curves.values() for x, _ in runs])
    for solver, runs in curves.items():
        color = solver_color(solver, solvers)
        if sample is not None:
            x, e = runs[0]
            if xscale == "log":
                x, e = x[1:], e[1:]
            ax.plot(x, e, color=color, linewidth=1.5, label=SOLVER_NAMES.get(solver, solver))
            # A dot at the first iterate: a solve whose first logged iterate already ends it (Newton at small h)
            # is a single point on the log axis
            ax.plot(x[:1], e[:1], "o", color=color, markersize=3)
            continue
        grid = np.geomspace(t_start, t_end, 400) if xscale == "log" else np.linspace(0.0, t_end, 400)
        reached = np.array([error_reached(x, e, grid) for x, e in runs])
        ax.fill_between(grid, reached.min(axis=0), reached.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(grid, np.median(reached, axis=0), color=color, linewidth=1.5, label=SOLVER_NAMES.get(solver, solver))
    bounds = None
    for solver in present:
        if "reference_relative_error_bound" in g[solver]:
            sg = g[solver]
            bounds = np.array([float(sg["reference_relative_error_bound"][sample_index(sg, k)]) for k in samples])
            break
    # The log axis ends a little past the last iterate, so a curve's last point isn't cut by the frame
    x_end = t_end * (1.15 if xscale == "log" else 1.0)
    for level in levels:
        ax.axhline(1.0 - level, color="#BBBBBB", linewidth=0.6, linestyle=":", zorder=0)
        ax.text(x_end * (1.04 if xscale == "log" else 1.01), 1.0 - level, f"{100 * level:g}%", fontsize=6, color="#777777", va="center")
    ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    if bounds is not None:
        # Darker below every sample's reference accuracy, lighter where only some can't resolve the error
        ax.axhspan(1e-12, bounds.min(), color="#DDDDDD", alpha=0.6, zorder=0, linewidth=0)
        if bounds.max() > bounds.min():
            ax.axhspan(bounds.min(), bounds.max(), color="#DDDDDD", alpha=0.3, zorder=0, linewidth=0)
    reference_bound = float(bounds.min()) if bounds is not None else None
    floor = min(1.0 - levels[-1], reference_bound or 1.0) * 0.1
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.5)
    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(0.7 * t_start, x_end)
    else:
        ax.set_xlim(0.0, x_end)
    ax.set_xlabel(r"$\dfrac{t}{T_N}$")
    which = f"sample {sample}" if sample is not None else f"{len(samples)} samples"
    ax.set_title(f"{benchmark_title(benchmark)}: {case_label(g, case).replace(chr(10), ', ')}, {which}", fontsize=9)
    ax.grid(True, which="major", color="#EEEEEE", linewidth=0.5)
    return present


def main():
    parser = argparse.ArgumentParser(description="Convergence curves (e/e0 against time / T_N) of a benchmark results file.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True, help="Output prefix: writes <prefix>.pdf and .png.")
    parser.add_argument("--problems", nargs="+", default=None, metavar="BENCHMARK/CASE",
                        help="Problems to draw, one panel each (default: the first case of each benchmark).")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--sample", type=int, default=0, help="Sample index (default: %(default)s).")
    which.add_argument("--all-samples", action="store_true", dest="all_samples",
                       help="Every sample: per solver, the min-max band and the median of e/e0 over the samples.")
    parser.add_argument("--solvers", default=None, help="Comma-separated solvers (default: the file's).")
    parser.add_argument("--xscale", choices=("log", "linear"), default="log",
                        help="Time axis scale (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Directory with the Linux Libertine OpenType files (default: PBAT_FONT_DIR, then the "
                        "installed fonts).")
    args = parser.parse_args()
    use_paper_fonts(args.font_dir)
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
            drawn += [s for s in plot_problem(ax, f, benchmark, case, None if args.all_samples else args.sample, solvers,
                                                     args.xscale) if s not in drawn]
        axes[0][0].set_ylabel(r"$\dfrac{e}{e_0}$", rotation=0, labelpad=10)  # upright: a fraction
        handles = [plt.Line2D([], [], color=solver_color(s, solvers), linewidth=1.5, label=SOLVER_NAMES.get(s, s))
                   for s in drawn]
        if args.all_samples:
            handles.append(matplotlib.patches.Patch(facecolor=(0.5, 0.5, 0.5, 0.15), edgecolor="none",
                                                    label="min–max (line: median)"))
        handles.append(matplotlib.patches.Patch(color="#DDDDDD", label="below reference accuracy"))
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
