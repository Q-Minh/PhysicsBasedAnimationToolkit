"""
Figures of a convergence benchmark results file (``benchmark.py``): one 3D grouped bar chart per benchmark
(static, dynamic_h0.0001, dynamic_h0.01).

- x axis: the benchmark's cases (scenes), in the results file's order;
- depth axis: the accuracy levels (error reductions 1 - e/e0);
- height: the mean incremental time of the level (from the previous level, from the start for the
  first) over the samples that reached it, as a fraction of the case's time budget.

Each (case, level) cell has one bar per solver, colored by solver. A translucent box around each mean bar
spans the min and max increments over the same samples. A bar whose level not every sample reached is
labeled with the fraction that did (e.g. 28/32), and the highest level always is, also with 0/N, so a
solver that stalled shows. A level no sample reached has no bar, nor has a level every sample that
reached it jumped over (zero time). Bars are drawn back to front for the chosen camera, so nearer bars
cover farther ones.

Usage:
    python -m python.tools.sim.benchmark_plot results.h5 -o out/figures/benchmark \
        [--benchmarks static dynamic_h0.01] [--solvers VBD,AAAVBD,...] [--elev 28] [--azim -62] [--dpi 200]

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


def plot_benchmark(f: h5py.File, benchmark: str, solvers: list[str], elev: float, azim: float):
    """One figure: cases x levels x solvers bars of mean increment / budget, with min/max boxes."""
    levels = np.asarray(f.attrs["levels"], dtype=np.float64)
    cases = [(c, g) for b, c, g in results.iter_cases(f) if b == benchmark]
    solvers = [s for s in solvers if any(s in g for _, g in cases)]
    n_s, n_l = len(solvers), len(levels)
    slot = 0.84 / max(n_s, 1)  # bar footprint along x within a case
    bar_w, bar_d = 0.8 * slot, 0.4
    box_w, box_d = slot, 0.5  # min/max box footprint, around the mean bar
    x_range, y_range = (-0.6, len(cases) - 0.4), (-0.6, n_l - 0.4)
    aspect = (max(1.6, 1.1 * len(cases)), 1.8, 1.0)

    bars = []  # (x, y, z0, dz, color, label, is_box, width, depth)
    floating = []  # (x, y, z, label): labels of empty cells
    z_top = 0.0
    for i, (case, g) in enumerate(cases):
        budget = float(g.attrs["budget"])
        for k, solver in enumerate(solvers):
            if solver not in g:
                continue
            a = g[f"{solver}/aggregate/increment"]
            count, n = np.asarray(a["count"]), int(g[solver].attrs["n_samples"])
            mean, lo, hi = (np.asarray(a[s]) / budget for s in ("mean", "min", "max"))
            color = solver_color(solver, solvers)
            xc = i + (k - (n_s - 1) / 2) * slot
            for j in range(n_l):
                top = j == n_l - 1
                # The highest level always says how many samples reached it, also 0/N (a stalled solver)
                label = f"{count[j]}/{n}" if top or count[j] < n else None
                if count[j] == 0 or not hi[j] > 0:
                    # Not reached, or skipped by every sample that reached it: an empty cell
                    if top:
                        floating.append((xc, j, 0.0, label))
                    continue
                bars.append((xc, j, lo[j], hi[j] - lo[j], color, label, True, box_w, box_d))
                bars.append((xc, j, 0.0, mean[j], color, None, False, bar_w, bar_d))
                z_top = max(z_top, hi[j])

    fig = plt.figure(figsize=(max(7.0, 1.6 * len(cases) + 4.0), 5.6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)
    # Back to front: position along the direction toward the camera, in displayed (box-aspect) units
    a_el, a_az = np.radians(elev), np.radians(azim)
    toward_camera = np.array([np.cos(a_el) * np.cos(a_az), np.cos(a_el) * np.sin(a_az)])
    scale = np.array([aspect[0] / (x_range[1] - x_range[0]), aspect[1] / (y_range[1] - y_range[0])])
    bars.sort(key=lambda b: (toward_camera @ (np.array([b[0], b[1]]) * scale), not b[6]))
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
    ax.set_zlabel("Mean incremental time / budget", labelpad=8)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_box_aspect(aspect, zoom=0.95)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    handles = [Patch(color=solver_color(s, solvers), label=SOLVER_NAMES.get(s, s)) for s in solvers]
    handles.append(Patch(facecolor=(0.5, 0.5, 0.5, 0.18), edgecolor="grey", label="min-max"))
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(benchmark_title(benchmark), x=0.02, ha="left", fontsize=11)
    fig.text(0.02, 0.02, "Bars: mean over the samples that reached the level; n/N: samples that reached it "
             f"(always shown at {100 * levels[-1]:g}%)",
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
                        help="Comma-separated solvers, in bar order (default: the file's).")
    parser.add_argument("--elev", type=float, default=28.0, help="Camera elevation (degrees).")
    parser.add_argument("--azim", type=float, default=-62.0, help="Camera azimuth (degrees).")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    args = parser.parse_args()
    with h5py.File(args.results, "r") as f:
        results.check_format(f, args.results)
        solvers = args.solvers.split(",") if args.solvers else [str(s) for s in f.attrs["solvers"]]
        benchmarks = args.benchmarks or [str(b) for b in f.attrs["benchmarks"]]
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        for benchmark in benchmarks:
            fig = plot_benchmark(f, benchmark, solvers, args.elev, args.azim)
            for ext in ("pdf", "png"):
                path = f"{args.output}_{benchmark}.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                print(f"Wrote {path}")
            plt.close(fig)


if __name__ == "__main__":
    main()
