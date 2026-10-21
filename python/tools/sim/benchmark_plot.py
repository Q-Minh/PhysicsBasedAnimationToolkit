"""
Figures of a convergence benchmark results file (``benchmark.py``): one 3D grouped bar chart per benchmark
(static, dynamic_h0.0001, dynamic_h0.01).

- x axis: the benchmark's cases (scenes), in the results file's order;
- depth axis: the benchmark's accuracy levels (error reductions 1 - e/e0), as many as it has;
- height: the mean incremental time of the level (from the previous level, from the start for the
  first) over the samples that reached it, each sample's increment divided by its normalization time:
  Newton's time T_N to its highest level on that problem (so Newton's bars of a sample sum to 1), or the
  case's fixed budget (``benchmark.py --budget``).

Each (case, level) cell has one bar per solver, colored by solver. A translucent box around each mean bar
spans the min and max normalized increments over the same samples. With ``--counts``, a bar whose level not
every sample reached is labeled with the fraction that did (e.g. 28/32), and the highest level always is,
also with 0/N, so a solver that stalled shows; by default there are no labels (visually noisy; maintainer),
and ``benchmark_iterations.py``'s tables give the counts. A level no sample reached has no bar, nor has a level every sample
that reached it jumped over (zero time). Dynamic references are float32 solves of limited accuracy
(``resolvable``, per sample and level): a cell whose level the reference can't resolve for most samples
(no strict majority) is left empty, since its errors would measure the reference, not the solvers, and in
the other cells the samples it can't resolve are left out of the statistics and of N. The highest level
labeled is then the highest drawn one. Bars are drawn back to front for the chosen camera, so nearer bars
cover farther ones.

Budget: the bars score each solve within ``--budget-factor F`` x T_N. The default is 1 (the paper's bars), or
the run's factor when that is smaller. A run at a larger factor (``benchmark.py --budget-factor``, e.g. 2, so
that ``benchmark_curves.py`` shows the convergence past T_N) is re-scored from its logs: levels crossed after
F x T_N count as unreached, and the increments, aggregates and n/N follow (``benchmark_results.rescore``).
F larger than the run's factor is refused: its logs end there. Cases with a fixed ``--budget`` are scored
as run.

Solver order within a cell: the solvers with taller bars stand further back, so shorter ones in front stay
visible. Per figure, solvers are sorted by the 75th percentile of their mean bar heights over the figure's
drawn cells (a percentile rather than the mean, so a single tall cell doesn't decide), tallest at the
back; the legend lists them in the same order, back to front. ``--solver-order`` overrides it.

Usage:
    python -m python.tools.sim.benchmark_plot results.h5 -o out/figures/benchmark \
        [--benchmarks static dynamic_h0.01] [--solvers VBD,AAAVBD,...] [--solver-order Newton,VBD,...] \
        [--budget-factor 1] [--elev 28] [--azim -62] [--dpi 200]

Writes ``<prefix>_<benchmark>.pdf`` and ``.png`` per benchmark.
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import benchmark_results as results  # noqa: E402

# Okabe-Ito colors (colorblind-safe), fixed per solver so a solver keeps its color in every figure
SOLVER_COLORS = {
    "VBD": "#0072B2",
    "AAAVBD": "#D55E00",
    "Chebyshev": "#009E73",
    "AndersonSolver": "#CC79A7",
    "Newton": "#E69F00",
}
SOLVER_NAMES = {"AndersonSolver": "Anderson"}
OTHER_COLORS = ["#56B4E9", "#F0E442", "#000000"]


def benchmark_title(benchmark: str) -> str:
    if benchmark == "static":
        return "Elastostatics"
    if benchmark.startswith("dynamic_h"):
        return f"Elastodynamics, h = {float(benchmark[len('dynamic_h'):]):g} s"
    return benchmark


def solver_color(solver: str, solvers: list[str]) -> str:
    if solver in SOLVER_COLORS:
        return SOLVER_COLORS[solver]
    others = [s for s in solvers if s not in SOLVER_COLORS]
    return OTHER_COLORS[others.index(solver) % len(OTHER_COLORS)]


def case_label(group: h5py.Group, case: str) -> str:
    label = str(group.attrs.get("label", group.attrs.get("scene", case)))
    if case != group.attrs.get("scene", case):  # one case per energy level of the scene
        label += f"\nE = {float(group.attrs['energy']):g} J"
    return label


def resolvable_samples(g: h5py.Group, solvers: list[str]) -> np.ndarray | None:
    """(S, L) bool: the levels each sample's reference resolves (any solver's ``resolvable``, the same for
    all); None for static cases, whose reference is exact."""
    for solver in solvers:
        if solver in g and "resolvable" in g[solver]:
            return np.asarray(g[solver]["resolvable"], dtype=bool)
    return None


def unresolvable_levels(g: h5py.Group, solvers: list[str], n_levels: int) -> np.ndarray:
    """(L,) bool: the levels the reference can't resolve for a strict majority of the samples (all False
    for static cases)."""
    resolvable = resolvable_samples(g, solvers)
    if resolvable is None:
        return np.zeros(n_levels, dtype=bool)
    return ~(2 * resolvable.sum(axis=0) > resolvable.shape[0])


def normalized_aggregate(
    g: h5py.Group, solver: str, rescore: tuple | None = None, counted: np.ndarray | None = None
) -> dict:
    """The solver's per-level statistics of normalized increments: ``normalized_increment`` (increment /
    T_N or the fixed budget), or for files without it, the increments over the case's budget. ``rescore``
    = (levels, factor) re-scores a T_N case at factor x T_N (:func:`benchmark_results.rescore`). ``counted``
    (S, L) bool restricts each level's statistics to those samples (e.g. :func:`resolvable_samples`)."""
    sg = g[solver]
    if rescore is not None and "budget" not in g.attrs and "normalization" in sg:
        a = results.rescore(sg, *rescore, counted=counted)
        return {s: a[s] for s in ("count", "mean", "min", "max")}
    if counted is not None:
        if "normalized_increment" in sg:
            values = np.asarray(sg["normalized_increment"])
        else:
            values = np.asarray(sg["increment"]) / float(g.attrs["budget"])
        a = results.aggregate(values, np.asarray(sg["reached"], dtype=bool) & counted)
        return {s: a[s] for s in ("count", "mean", "min", "max")}
    if "aggregate/normalized_increment" in sg:
        a = sg["aggregate/normalized_increment"]
        return {s: np.asarray(a[s]) for s in ("count", "mean", "min", "max")}
    a, budget = sg["aggregate/increment"], float(g.attrs["budget"])
    return {"count": np.asarray(a["count"]), **{s: np.asarray(a[s]) / budget for s in ("mean", "min", "max")}}


def drawn_heights(a: dict, unresolvable: np.ndarray) -> np.ndarray:
    """The mean bar heights a solver draws in a case: reached, not skipped by every sample, resolvable."""
    drawn = (a["count"] > 0) & (a["max"] > 0) & ~unresolvable
    return a["mean"][drawn]


def depth_order(cases, solvers: list[str], n_levels: int, rescore: tuple | None = None) -> list[str]:
    """Solvers back to front: descending 75th percentile of their drawn bar heights over the figure
    (0 for a solver without bars), ties in the given order."""
    key = {}
    for solver in solvers:
        heights = [
            drawn_heights(normalized_aggregate(g, solver, rescore, resolvable_samples(g, solvers)),
                          unresolvable_levels(g, solvers, n_levels))
            for _, g in cases
            if solver in g
        ]
        heights = np.concatenate(heights) if heights else np.zeros(0)
        key[solver] = float(np.percentile(heights, 75)) if heights.size else 0.0
    return sorted(solvers, key=lambda s: -key[s])


def budget_factor(f: h5py.File, factor: float | None) -> float:
    """The factor the bars score at: ``factor``, by default 1 or the run's factor when that is smaller.
    ValueError for one above the run's factor (its logs end there) or not positive."""
    run = float(f.attrs.get("budget_factor", 1.0))
    if factor is None:
        return min(1.0, run)
    if not factor > 0:
        raise ValueError(f"--budget-factor must be positive, got {factor:g}")
    if factor > run * (1 + 1e-9):
        raise ValueError(f"--budget-factor {factor:g} is above the run's {run:g}: its logs end at {run:g} x T_N")
    return factor


def plot_benchmark(
    f: h5py.File, benchmark: str, solvers: list[str], elev: float, azim: float, solver_order: list[str] | None = None,
    factor: float | None = None, counts: bool = False,
):
    """One figure: cases x levels x solvers bars of mean normalized increment, with min/max boxes.
    ``solver_order`` (back to front) overrides :func:`depth_order`. ``factor``: the budget factor the bars
    score at (:func:`budget_factor`); below the run's, the T_N cases are re-scored from their logs. ``counts``
    labels the bars with n/N (see the module docstring)."""
    levels = results.benchmark_levels(f, benchmark)
    factor = budget_factor(f, factor)
    run = float(f.attrs.get("budget_factor", 1.0))
    rescore = (levels, factor) if factor < run * (1 - 1e-9) else None
    cases = [(c, g) for b, c, g in results.iter_cases(f) if b == benchmark]
    solvers = [s for s in solvers if any(s in g for _, g in cases)]
    n_s, n_l = len(solvers), len(levels)
    # Back to front: the solvers with taller bars further from the camera
    if solver_order:
        solvers = [s for s in solver_order if s in solvers] + [s for s in solvers if s not in solver_order]
    else:
        solvers = depth_order(cases, solvers, n_l, rescore)
    slot = 0.84 / max(n_s, 1)  # bar footprint along x within a case
    bar_w, bar_d = 0.8 * slot, 0.4
    box_w, box_d = slot, 0.5  # min/max box footprint, around the mean bar
    x_range, y_range = (-0.6, len(cases) - 0.4), (-0.6, n_l - 0.4)
    aspect = (max(1.6, 1.1 * len(cases)), max(1.8, 0.3 * n_l), 1.0)
    a_el, a_az = np.radians(elev), np.radians(azim)
    toward_camera = np.array([np.cos(a_el) * np.cos(a_az), np.cos(a_el) * np.sin(a_az)])
    # Slots along x within a case: the first solver at the slot furthest from the camera
    x_sign = -1.0 if toward_camera[0] >= 0 else 1.0

    bars = []  # (x, y, z0, dz, color, label, is_box, width, depth)
    floating = []  # (x, y, z, label): labels of empty cells
    z_top = 0.0
    for i, (case, g) in enumerate(cases):
        unresolvable = unresolvable_levels(g, solvers, n_l)
        resolvable = resolvable_samples(g, solvers)
        top_level = int(np.nonzero(~unresolvable)[0][-1]) if (~unresolvable).any() else -1
        for k, solver in enumerate(solvers):
            if solver not in g:
                continue
            a = normalized_aggregate(g, solver, rescore, resolvable)
            n_samples = int(g[solver].attrs["n_samples"])
            counted = resolvable.sum(axis=0) if resolvable is not None else np.full(n_l, n_samples)
            count = a["count"]
            mean, lo, hi = a["mean"], a["min"], a["max"]
            color = solver_color(solver, solvers)
            xc = i - x_sign * (k - (n_s - 1) / 2) * slot
            for j in range(n_l):
                if unresolvable[j]:
                    continue
                top = j == top_level
                # The highest drawn level always says how many samples reached it, also 0/N (a stalled
                # solver); N counts the samples whose reference resolves the level
                n = int(counted[j])
                label = f"{count[j]}/{n}" if counts and (top or count[j] < n) else None
                if count[j] == 0 or not hi[j] > 0:
                    # Not reached, or skipped by every sample that reached it: an empty cell
                    if label:
                        floating.append((xc, j, 0.0, label))
                    continue
                bars.append((xc, j, lo[j], hi[j] - lo[j], color, label, True, box_w, box_d))
                bars.append((xc, j, 0.0, mean[j], color, None, False, bar_w, bar_d))
                z_top = max(z_top, hi[j])

    fig = plt.figure(figsize=(max(7.0, 1.6 * len(cases) + 4.0), 5.6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)
    # Back to front: position along the direction toward the camera, in displayed (box-aspect) units
    scale = np.array([aspect[0] / (x_range[1] - x_range[0]), aspect[1] / (y_range[1] - y_range[0])])
    def depth(x, y):
        return toward_camera @ (np.array([x, y]) * scale)

    bars.sort(key=lambda b: (depth(b[0], b[1]), not b[6]))
    labels = []
    for order, (x, y, z0, dz, color, label, is_box, w, d) in enumerate(bars):
        if is_box:
            ax.bar3d(x - w / 2, y - d / 2, z0, w, d, max(dz, 1e-9), color=color, alpha=0.18,
                     edgecolor=color, linewidth=0.3, shade=False, zorder=order)
            if label:
                labels.append((x, y, z0 + dz, label, order))
        else:
            ax.bar3d(x - w / 2, y - d / 2, 0.0, w, d, max(dz, 1e-9), color=color, alpha=1.0,
                     edgecolor="white", linewidth=0.2, shade=True, zorder=order)
    z_max = 1.08 * z_top if z_top > 0 else 1.0
    labels += [(x, y, z, label, len(bars)) for x, y, z, label in floating]
    for x, y, z, label, order in labels:
        ax.text(x, y, z + 0.015 * z_max, label, fontsize=6, ha="center", va="bottom", color="#333333",
                zorder=len(bars) + order)

    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([case_label(g, c) for c, g in cases], fontsize=8)
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f"{100 * v:g}%" for v in levels], fontsize=8)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(0.0, z_max)
    ax.set_xlabel("Scene", labelpad=10)
    ax.set_ylabel("Error reduction", labelpad=8)
    fixed = [("budget" in g.attrs) for _, g in cases]
    norm = "budget" if all(fixed) else "Newton's time T_N" if not any(fixed) else "T_N or budget"
    ax.set_zlabel(f"Mean incremental time / {norm}", labelpad=8)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_box_aspect(aspect, zoom=0.95)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    handles = [Patch(color=solver_color(s, solvers), label=SOLVER_NAMES.get(s, s)) for s in solvers]
    handles.append(Patch(facecolor=(0.5, 0.5, 0.5, 0.18), edgecolor="grey", label="min-max"))
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(benchmark_title(benchmark), x=0.02, ha="left", fontsize=11)
    within = "" if all(fixed) else f" within {factor:g} x T_N"
    unresolved = any(resolvable_samples(g, solvers) is not None for _, g in cases)
    reference = (
        "; no bar: not reached, or beyond the reference's accuracy for most samples (elsewhere, the samples it "
        "can't resolve are left out)" if unresolved else "; no bar: not reached"
    )
    count_text = "; n/N: samples that reached it (always shown at the highest level drawn)" if counts else ""
    fig.text(0.02, 0.02, f"Bars: mean over the samples that reached the level{within}{count_text}{reference}",
             fontsize=7, color="#555555")
    return fig


def main():
    parser = argparse.ArgumentParser(description="3D grouped bar charts of a convergence benchmark results file.")
    parser.add_argument("results", help="Results file (.h5) from python -m python.tools.sim.benchmark.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output prefix: writes <prefix>_<benchmark>.pdf and .png.")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmarks to plot (default: all in the file, e.g. static dynamic_h0.01).")
    parser.add_argument("--solvers", default=None,
                        help="Comma-separated solvers to draw (default: the file's).")
    parser.add_argument("--solver-order", default=None, dest="solver_order",
                        help="Comma-separated solvers, back to front within a cell (default: per figure, taller bars "
                        "further back; see the module docstring).")
    parser.add_argument("--budget-factor", type=float, default=None, dest="budget_factor",
                        help="Score the bars within this factor x Newton's T_N, at most the run's factor (a run at a "
                        "larger one is re-scored from its logs). Default: 1, or the run's factor when smaller.")
    parser.add_argument("--counts", action="store_true",
                        help="Label the bars with n/N, the samples that reached the level (see the module docstring).")
    parser.add_argument("--elev", type=float, default=28.0, help="Camera elevation (degrees).")
    parser.add_argument("--azim", type=float, default=-62.0, help="Camera azimuth (degrees).")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    args = parser.parse_args()
    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        solvers = args.solvers.split(",") if args.solvers else [str(s) for s in f.attrs["solvers"]]
        benchmarks = args.benchmarks or [str(b) for b in f.attrs["benchmarks"]]
        try:
            factor = budget_factor(f, args.budget_factor)
        except ValueError as e:
            raise SystemExit(str(e))
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        for benchmark in benchmarks:
            order = args.solver_order.split(",") if args.solver_order else None
            fig = plot_benchmark(f, benchmark, solvers, args.elev, args.azim, order, factor, args.counts)
            for ext in ("pdf", "png"):
                path = f"{args.output}_{benchmark}.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                print(f"Wrote {path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
