"""
Paper figures and tables of profiling results (``simulate.py --profile kernels --profile-out``), with the
per-step statistics of ``profiling/step_stats.py``.

Phases are grouped for the paper by :data:`PHASE_GROUPS` (scope name -> group, fixed colors); the rest are
"Other". Phase times are unions over streams per step (concurrent streams count once), which exclude
graph launch overhead and the gaps between kernels; production runs (``simulate.py --profile-out`` without a
backend: graph replay for the VBD family) give the true step times. Production runs are paired with kernels
runs by scene and solver. They are separate runs whose trajectories differ slightly (contact is chaotic and
GPU arithmetic order isn't deterministic), so they are compared as distributions, never step by step.

- ``timeline``: one row per kernels result (e.g. AAAVBD on each scene). Left: GPU time per step over the
  simulated time, stacked by phase group, with the paired production run's step time as a line, and below
  it the peak node speed (``profiling/trajectory.py``) on the same time axis. Right: one phase donut per
  slice of the steps by percentiles of their GPU time (default: the fastest 10%, P10-P90, the slowest 10%)
  or of their peak speed (``--slice-by speed``), with the slice's median step time in the center and its
  median iterations and early exits below.
- ``solvers``: one row per kernels result (e.g. every solver on one scene). Left: each phase group's share
  of the GPU time per step, with the mean kernel union per step at the bar's end. Middle: the step time distribution
  (box: quartiles, whiskers: P10-P90, tick: median) of the production run, or of the kernel union without
  one, and the kernel union's median as a diamond. Right: the fraction of solves that exited early (passed
  their convergence tests within the budget), with the median iterations per step.
- ``table``: a LaTeX ``table*`` (booktabs) with one row per kernels result: production step time (mean,
  median, P90), the kernel union's median and its ratio to production, the overlap of phases across
  streams, early exits, median iterations, time per iteration and the worst residual ratios.

Usage (from the repo root):

    python -m python.tools.sim.profiling.paper_figures timeline k_basket.json k_capsules.json \
        --production p_basket.json p_capsules.json -o data/figures/profiling/timeline \
        [--labels basket_of_fruit="Basket of fruit" ...] [--width 7] [--bin 1] [--slices 10,90] [--slice-by time]
    python -m python.tools.sim.profiling.paper_figures solvers k_*.json --production p_*.json \
        -o data/figures/profiling/solvers [--width 7]
    python -m python.tools.sim.profiling.paper_figures table k_*.json --production p_*.json \
        -o data/tables/profiling.tex [--labels ...]

Figures follow ``benchmark_plot.py``: Linux Biolinum (``--font-dir``, ``PBAT_FONT_DIR`` or installed), rendered
at the printed width (``--width``, inches) with 7-8 pt text, no titles (the LaTeX caption carries them; rows
and columns have headers in the figure), one legend on top, no grid lines, PDF only.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from . import results, step_stats  # noqa: E402
from ..benchmark_plot import SOLVER_NAMES, use_paper_fonts  # noqa: E402

# The paper's phase groups, in drawing order (bottom of the stack first): name -> (scope names, color). Scope
# names are innermost scopes (results.leaf); a scope in no group is drawn as OTHER. AAAVBD's acceleration runs
# inside its vertex-solve kernel (aaavbd.sweep), so it has no group: its cost shows in its time per sweep
# against VBD's. Colors: the profiling report's categorical palette, in its validated order.
PHASE_GROUPS = {
    "Vertex sweeps": (("vbd.sweep", "aaavbd.sweep"), "#2a78d6"),
    "Contact detection": (("contact.detect", "contact.filter", "contact.constraint_set"), "#eb6834"),
    "Contact update": (("contact.dual", "contact.penalty", "newton.adapt_penalty"), "#1baf7a"),
    "Residuals": (("newton.gradient", "newton.convergence"), "#eda100"),
    "Mixing": (("chebyshev.mix", "anderson.mix", "anderson.gram", "anderson.solve"), "#e87ba4"),
    "Newton CG": (("newton.linear_solve",), "#008300"),
    "Newton assembly, line search": (("newton.assemble", "newton.line_search", "newton.energy"), "#6250d6"),
}
OTHER = "Other"
OTHER_COLOR = "#a3a3a3"
INK = "#333333"  # text and neutral marks
FONT_SIZE = 7
PAD = 0.02  # inches around the saved figure


def phase_group(name: str) -> str:
    """The paper group of an innermost scope name."""
    for group, (scopes, _) in PHASE_GROUPS.items():
        if name in scopes:
            return group
    return OTHER


def group_color(group: str) -> str:
    return PHASE_GROUPS[group][1] if group in PHASE_GROUPS else OTHER_COLOR


def group_phases(phases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-step phase times (``step_stats.per_step``'s ``phases``) summed per group, in :data:`PHASE_GROUPS`
    order then :data:`OTHER`, leaving out groups without work."""
    out: dict[str, np.ndarray] = {}
    for name, ms in phases.items():
        g = phase_group(name)
        out[g] = out[g] + ms if g in out else np.array(ms, dtype=float)
    order = [*PHASE_GROUPS, OTHER]
    return {g: out[g] for g in order if g in out and np.any(out[g] > 0)}


def scene_key(result: dict) -> str:
    """A result's scene as a short key: the scene file's name without extension and group,
    e.g. ``basket_of_fruit`` for ``path/to/basket_of_fruit.h5:fem``."""
    scene = os.path.basename(str(result["metadata"].get("scene") or ""))
    return os.path.splitext(scene.split(":")[0])[0]


def solver_name(result: dict) -> str:
    solver = str(result["metadata"].get("solver", ""))
    return SOLVER_NAMES.get(solver, solver)


def pair_production(kernels: list[dict], production: list[dict]) -> list[dict | None]:
    """For each kernels result, the production result of the same scene and solver, or None."""
    by_key = {(scene_key(p), p["metadata"].get("solver")): p for p in production}
    return [by_key.get((scene_key(k), k["metadata"].get("solver"))) for k in kernels]


def step_seconds(result: dict, t: np.ndarray) -> np.ndarray:
    """Simulated time (s) at the start of each step, from the result's time step."""
    return t * float(result["metadata"].get("dt", 1.0))


def bin_steps(values: np.ndarray, n: int) -> np.ndarray:
    """Means over consecutive blocks of ``n`` steps (the last block may be shorter)."""
    values = np.asarray(values, dtype=float)
    if n <= 1 or values.size == 0:
        return values
    starts = np.arange(0, values.size, n)
    counts = np.diff(np.append(starts, values.size))
    return np.add.reduceat(values, starts) / counts


def slice_header(s: dict, edges: tuple[float, ...], what: str) -> str:
    """A slice's column header: ``Fastest 10%`` / ``Slowest 10%`` for the outer slices of step time
    (``Slowest`` / ``Fastest`` for speed), else its percentile range."""
    low, high = ("Fastest", "Slowest") if what == "time" else ("Slowest", "Fastest")
    if s["lo"] == 0.0:
        return f"{low} {edges[0]:g}%"
    if s["hi"] == 100.0:
        return f"{high} {100.0 - edges[-1]:g}%"
    return f"P{s['lo']:g}–P{s['hi']:g}"


def _fmt_ms(ms: float | None) -> str:
    if ms is None or not np.isfinite(ms):
        return "–"
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1000:.3g} s"


def _iterations_text(summary: dict, unit: str) -> str:
    it = summary["iterations"]
    return "" if it is None or not unit else f"{it:.0f} {unit}"


def _early_text(summary: dict) -> str:
    e = summary["early_exit"]
    return "" if e is None else f"{100 * e:.0f}% early exits"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZE, length=2, width=0.5, pad=1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)


def _legend(fig, handles, ncol=None):
    """One legend above everything drawn, centered on it."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drawn = Bbox.union([ax.get_tightbbox(renderer) for ax in fig.axes]).transformed(fig.transFigure.inverted())
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=((drawn.x0 + drawn.x1) / 2, drawn.y1),
               ncol=ncol or len(handles), fontsize=FONT_SIZE, frameon=False, borderaxespad=0.3, handlelength=1.2,
               columnspacing=1.2, handletextpad=0.4)


def _donut(ax, phases: dict[str, float], center: str):
    groups = [g for g, v in phases.items() if v > 0]
    ax.set_aspect("equal")
    if not groups:  # an empty slice (ties at a percentile fall into the lower slice)
        ax.add_patch(plt.Circle((0, 0), 0.81, fill=False, edgecolor=OTHER_COLOR, linewidth=0.5))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.text(0, 0, "no steps", ha="center", va="center", fontsize=FONT_SIZE, color=INK)
        return
    ax.pie([phases[g] for g in groups], colors=[group_color(g) for g in groups], startangle=90, counterclock=False,
           wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 0.6})
    ax.text(0, 0, center, ha="center", va="center", fontsize=FONT_SIZE, color=INK)


def plot_timeline(kernels: list[dict], production: list[dict | None], labels: list[str], width: float = 7.0,
                  bin_n: int = 1, edges: tuple[float, ...] = (10.0, 90.0), slice_by: str = "time"):
    """The ``timeline`` figure (see the module docstring): one row per kernels result."""
    key = {"time": "time_ms", "speed": "peak_speed"}[slice_by]
    tables = [step_stats.per_step(k) for k in kernels]
    n_slices = len(edges) + 1
    rows = len(kernels)
    has_speed = [bool(np.isfinite(t["peak_speed"]).any()) for t in tables]
    fig = plt.figure(figsize=(width, 1.75 * rows))
    grid = fig.add_gridspec(rows, 1 + n_slices, width_ratios=[4.2] + [1.0] * n_slices, hspace=0.55, wspace=0.12)
    present: list[str] = []
    drew_production = False
    for r, (result, table, prod, label) in enumerate(zip(kernels, tables, production, labels)):
        groups = group_phases(table["phases"])
        present += [g for g in groups if g not in present]
        x = bin_steps(step_seconds(result, table["t"]), bin_n)
        left = grid[r, 0].subgridspec(2, 1, height_ratios=[3.0, 1.0 if has_speed[r] else 0.001], hspace=0.12)
        area = fig.add_subplot(left[0])
        area.stackplot(x, *[bin_steps(v, bin_n) for v in groups.values()],
                       colors=[group_color(g) for g in groups], linewidth=0)
        if prod is not None:
            ptable = step_stats.per_step(prod)
            area.plot(bin_steps(step_seconds(prod, ptable["t"]), bin_n), bin_steps(ptable["time_ms"], bin_n),
                      color=INK, linewidth=0.6)
            drew_production = True
        area.set_xlim(x[0], x[-1])
        area.set_ylim(bottom=0)
        area.set_ylabel("GPU time (ms)", fontsize=FONT_SIZE)
        area.text(0, 1.04, label, transform=area.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold",
                  color=INK)
        _style(area)
        last = area
        if has_speed[r]:
            area.tick_params(labelbottom=False)
            strip = fig.add_subplot(left[1], sharex=area)
            strip.plot(x, bin_steps(table["peak_speed"], bin_n), color=INK, linewidth=0.6)
            strip.set_ylim(bottom=0)
            strip.set_ylabel("Speed\n(m/s)", fontsize=FONT_SIZE)
            _style(strip)
            last = strip
        if r == rows - 1:
            last.set_xlabel("Time (s)" if "dt" in result["metadata"] else "Time step", fontsize=FONT_SIZE)
        slices = step_stats.percentile_slices(table[key], edges)
        for c, s in enumerate(slices):
            ax = fig.add_subplot(grid[r, 1 + c])
            summary = step_stats.summarize_slice(table, s["mask"])
            grouped = {g: float(v[s["mask"]].mean()) if s["mask"].any() else 0.0 for g, v in groups.items()}
            _donut(ax, grouped, _fmt_ms(summary["time_ms"]))
            if r == 0:
                ax.set_title(slice_header(s, edges, slice_by), fontsize=FONT_SIZE, color=INK, pad=2)
            below = [t for t in (_iterations_text(summary, table["unit"]), _early_text(summary)) if t]
            ax.text(0.5, -0.04, "\n".join(below), transform=ax.transAxes, ha="center", va="top",
                    fontsize=FONT_SIZE - 1, color=INK)
    handles = [Patch(color=group_color(g), label=g) for g in present]
    if drew_production:
        handles.append(Line2D([], [], color=INK, linewidth=0.6, label="Step time (graph replay)"))
    _legend(fig, handles, ncol=min(len(handles), 6))
    return fig


def plot_solvers(kernels: list[dict], production: list[dict | None], width: float = 7.0):
    """The ``solvers`` figure (see the module docstring): one row per kernels result, top to bottom."""
    tables = [step_stats.per_step(k) for k in kernels]
    names = [solver_name(k) for k in kernels]
    n = len(kernels)
    fig, (share, dist, early) = plt.subplots(1, 3, figsize=(width, 0.32 * n + 0.75), sharey=True,
                                             gridspec_kw={"width_ratios": [2.4, 2.0, 1.4], "wspace": 0.3})
    y = np.arange(n)[::-1]
    present: list[str] = []
    for yi, table in zip(y, tables):
        groups = {g: float(v.mean()) for g, v in group_phases(table["phases"]).items()}
        present += [g for g in groups if g not in present]
        total = sum(groups.values())
        left = 0.0
        for g, ms in groups.items():
            w = 100.0 * ms / total if total > 0 else 0.0
            share.barh(yi, w, left=left, height=0.62, color=group_color(g), edgecolor="white", linewidth=0.6)
            left += w
        share.text(101.5, yi, _fmt_ms(float(np.nanmean(table["time_ms"]))), va="center", ha="left",
                   fontsize=FONT_SIZE - 1, color=INK)
    share.set_xlim(0, 100)
    share.set_xlabel("Share of GPU time per step (%)", fontsize=FONT_SIZE)
    share.set_yticks(y, names)

    any_production = any(p is not None for p in production)
    for yi, table, prod in zip(y, tables, production):
        times = step_stats.per_step(prod)["time_ms"] if prod is not None else table["time_ms"]
        d = step_stats.describe(times)
        if d is None:
            continue
        dist.plot([d["P10"], d["P90"]], [yi, yi], color=INK, linewidth=0.6)
        dist.barh(yi, d["Q3"] - d["Q1"], left=d["Q1"], height=0.5, color="#dcdcdc", edgecolor=INK, linewidth=0.5)
        dist.plot([d["median"]] * 2, [yi - 0.25, yi + 0.25], color=INK, linewidth=1.0)
        if prod is not None:
            u = step_stats.describe(table["time_ms"])
            if u is not None:
                dist.plot(u["median"], yi, marker="D", markersize=3, markerfacecolor="white",
                          markeredgecolor=INK, markeredgewidth=0.6, linestyle="none")
    dist.set_xscale("log")
    dist.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    dist.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    dist.xaxis.set_minor_formatter(NullFormatter())
    dist.set_xlabel("Step time (ms)" if any_production else "Step time, kernel union (ms)", fontsize=FONT_SIZE)

    for yi, table in zip(y, tables):
        s = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
        if s["early_exit"] is not None:
            early.barh(yi, 100.0 * s["early_exit"], height=0.5, color="#8c8c8c", linewidth=0)
        text = _iterations_text(s, table["unit"]) or "n/a"
        early.text(102, yi, text, va="center", ha="left", fontsize=FONT_SIZE - 1, color=INK)
    early.set_xlim(0, 100)
    early.set_xlabel("Early exits (%)", fontsize=FONT_SIZE)
    for ax in (share, dist, early):
        _style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.patch.set_visible(False)  # the bar-end labels reach into the next panel

    handles = [Patch(color=group_color(g), label=g) for g in present]
    if any_production:
        handles += [Patch(facecolor="#dcdcdc", edgecolor=INK, linewidth=0.5, label="Graph replay"),
                    Line2D([], [], marker="D", markersize=3, markerfacecolor="white", markeredgecolor=INK,
                           markeredgewidth=0.6, linestyle="none", label="Kernel union (median)")]
    _legend(fig, handles, ncol=min(len(handles), 6))
    return fig


def table_row(kernels: dict, production: dict | None) -> dict:
    """One row of the ``table``: step time statistics (production if given), the kernel union's median and
    its ratio to production's, the phases' overlap across streams, and the solve outcome statistics."""
    table = step_stats.per_step(kernels)
    s = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
    union = step_stats.describe(table["time_ms"])
    prod = step_stats.describe(step_stats.per_step(production)["time_ms"]) if production is not None else None
    ratio = None
    if union is not None and prod is not None:
        ratio = union["median"] / max(prod["median"], 1e-9)
    return {
        "scene": scene_key(kernels),
        "solver": solver_name(kernels),
        "production": prod,
        "union": union,
        "union_ratio": ratio,
        "overlap": step_stats.union_overlap(table),
        "early_exit": s["early_exit"],
        "iterations": s["iterations"],
        "unit": table["unit"],
        "ms_per_iteration": s["ms_per_iteration"],
        "gradient_ratio_max": s["gradient_ratio_max"],
        "feasibility_ratio_max": s["feasibility_ratio_max"],
    }


def _num(x, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}g}"


def latex_table(rows: list[dict], labels: dict[str, str]) -> str:
    """The ``table`` (booktabs, multirow): rows grouped by scene, numbers to 3 significant digits."""
    out = [r"\begin{table*}", r"\centering", r"\small",
           r"\caption{\textbf{GPU time per time step.} Step time of the production runs (CUDA graph replay "
           r"for the VBD family) and median of the profiled runs' kernel union over streams, which excludes "
           r"graph launch overhead and the gaps between kernels; overlap: time the phases run concurrently on "
           r"several streams; early exits: solves that passed their convergence tests within their budget; "
           r"iterations: VBD sweeps or Newton steps per time step (median); worst residual ratios to their "
           r"tolerances (gradient, feasibility).}",
           r"\label{tab:profiling}",
           r"\begin{tabular}{llrrrrrrrrrrr}", r"\toprule",
           r"& & \multicolumn{3}{c}{Step time (ms)} & \multicolumn{2}{c}{Kernel union} & & & & & "
           r"\multicolumn{2}{c}{Worst ratio} \\",
           r"\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){12-13}",
           r"Scene & Solver & mean & median & P90 & median (ms) & / step & Overlap & Early exits & Iterations & "
           r"ms / iter. & gradient & feasibility \\",
           r"\midrule"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["scene"]] = counts.get(row["scene"], 0) + 1
    previous = None
    for row in rows:
        if previous is not None and row["scene"] != previous:
            out.append(r"\midrule")
        label = labels.get(row["scene"], row["scene"].replace("_", r"\_"))
        scene = rf"\multirow{{{counts[row['scene']]}}}{{*}}{{{label}}}" if row["scene"] != previous else ""
        previous = row["scene"]
        p = row["production"] or {}
        pct = (lambda v: "--" if v is None else f"{100 * v:.0f}\\%")
        cells = [scene, row["solver"], _num(p.get("mean")), _num(p.get("median")), _num(p.get("P90")),
                 _num(row["union"]["median"] if row["union"] else None), _num(row["union_ratio"], 2),
                 pct(row["overlap"]), pct(row["early_exit"]),
                 "--" if row["iterations"] is None else f"{row['iterations']:.0f}",
                 _num(row["ms_per_iteration"]), _num(row["gradient_ratio_max"]),
                 _num(row["feasibility_ratio_max"])]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def _parse_labels(items: list[str]) -> dict[str, str]:
    labels = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--labels expects SCENE=LABEL, got {item!r}")
        labels[key] = value
    return labels


def main():
    parser = argparse.ArgumentParser(description="Paper figures and tables of profiling results.")
    parser.add_argument("kind", choices=["timeline", "solvers", "table"], help="What to write (see the docstring).")
    parser.add_argument("kernels", nargs="+", help="Kernels-backend results (.json), one per row.")
    parser.add_argument("--production", nargs="*", default=[],
                        help="Production results (.json), paired with the kernels results by scene and solver.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output: <output>.pdf for figures, the .tex file for the table.")
    parser.add_argument("--labels", nargs="*", default=[], metavar="SCENE=LABEL",
                        help="Row labels per scene key (the scene file's name without extension), e.g. "
                        "basket_of_fruit=\"Basket of fruit\" (default: the key).")
    parser.add_argument("--width", type=float, default=7.0,
                        help="Printed width in inches (7: TOG's text width, 3.33: a column); fonts at nominal size.")
    parser.add_argument("--bin", type=int, default=1, dest="bin_n",
                        help="timeline: average the time series over blocks of this many steps.")
    parser.add_argument("--slices", default="10,90",
                        help="timeline: percentile edges of the donut slices (default: 10,90).")
    parser.add_argument("--slice-by", choices=["time", "speed"], default="time", dest="slice_by",
                        help="timeline: slice the steps by GPU time or by peak node speed.")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Directory with the Linux Libertine OpenType files (default: PBAT_FONT_DIR, then the "
                        "installed fonts).")
    args = parser.parse_args()
    kernels = [results.load(p) for p in args.kernels]
    production = pair_production(kernels, [results.load(p) for p in args.production])
    labels = _parse_labels(args.labels)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.kind == "table":
        text = latex_table([table_row(k, p) for k, p in zip(kernels, production)], labels)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("% Needs \\usepackage{booktabs, multirow}\n\n" + text)
        print(f"Wrote {args.output}")
        return
    use_paper_fonts(args.font_dir)
    if args.kind == "timeline":
        edges = tuple(float(e) for e in args.slices.split(","))
        fig = plot_timeline(kernels, production, [labels.get(scene_key(k), scene_key(k)) for k in kernels],
                            args.width, args.bin_n, edges, args.slice_by)
    else:
        fig = plot_solvers(kernels, production, args.width)
    path = f"{args.output}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=PAD)
    plt.close(fig)
    print(f"Wrote {path}")


import unittest


class TestPaperFigures(unittest.TestCase):

    @staticmethod
    def _kernels(solver="AAAVBD", scene="path/to/basket_of_fruit.h5:fem", n=40):
        steps = []
        for t in range(10, 10 + n):
            contact = t >= 25
            steps.append({
                "t": t,
                "union_ms": {"solve/aaavbd.subproblem/aaavbd.sweep": 8.0 + 4.0 * contact,
                             "solve/solve.initialize/contact.detect": 0.5 + 1.5 * contact,
                             "solve/aaavbd.subproblem/contact.dual": 0.3,
                             "solve/aaavbd.subproblem/newton.gradient": 0.1,
                             "step.setup": 0.05},
                "calls": {"solve/aaavbd.subproblem/aaavbd.sweep": 100},
                "step_union_ms": 8.95 + 5.5 * contact,  # the phases' sum: no overlap
                "status": {"vbd": [{"substep": 0, "outer": 1 if t % 4 == 0 else 5, "sweeps": 20 if t % 4 == 0 else 100,
                                    "reason": "CONVERGED" if t % 4 == 0 else "GRADIENT_UNMET",
                                    "gradient_ratio": 0.5 if t % 4 == 0 else 3.0, "feasibility_ratio": 0.1,
                                    "max_outer": 5, "max_sweeps": 100}]},
                "peak_speed": 0.1 * t,
            })
        return {"source": "kernels", "metadata": {"scene": scene, "solver": solver, "dt": 0.01}, "steps": steps}

    @staticmethod
    def _production(kernels: dict):
        steps = [{"t": s["t"], "step_timeline_ms": s["step_union_ms"] * 1.08} for s in kernels["steps"]]
        return {"source": "production", "metadata": dict(kernels["metadata"]), "steps": steps}

    def test_groups(self):
        self.assertEqual(phase_group("aaavbd.sweep"), "Vertex sweeps")
        self.assertEqual(phase_group("contact.constraint_set"), "Contact detection")
        self.assertEqual(phase_group("step.setup"), OTHER)
        self.assertEqual(phase_group(results.UNSCOPED), OTHER)
        scopes = [s for scopes, _ in PHASE_GROUPS.values() for s in scopes]
        self.assertEqual(len(scopes), len(set(scopes)))  # every scope in one group at most
        table = step_stats.per_step(self._kernels())
        groups = group_phases(table["phases"])
        self.assertEqual(list(groups), ["Vertex sweeps", "Contact detection", "Contact update", "Residuals", OTHER])
        np.testing.assert_allclose(groups["Contact detection"][-1], 2.0)

    def test_pairing_and_bins(self):
        k = [self._kernels(), self._kernels("VBD"), self._kernels(scene="path/to/capsule_bowl.h5")]
        p = [self._production(k[2]), self._production(k[0])]
        self.assertEqual(scene_key(k[0]), "basket_of_fruit")
        paired = pair_production(k, p)
        self.assertIs(paired[0], p[1])
        self.assertIsNone(paired[1])
        self.assertIs(paired[2], p[0])
        np.testing.assert_allclose(bin_steps(np.arange(5.0), 2), [0.5, 2.5, 4.0])
        np.testing.assert_allclose(step_seconds(k[0], np.array([10, 20])), [0.1, 0.2])

    def test_table(self):
        k = self._kernels()
        row = table_row(k, self._production(k))
        self.assertAlmostEqual(row["union_ratio"], 1 / 1.08)
        self.assertEqual(row["early_exit"], 0.25)
        self.assertEqual((row["iterations"], row["unit"]), (100.0, "sweeps"))
        self.assertEqual(row["overlap"], 0.0)
        text = latex_table([row, table_row(self._kernels("VBD"), None)], {"basket_of_fruit": "Basket of fruit"})
        self.assertIn(r"\multirow{2}{*}{Basket of fruit}", text)
        self.assertIn("25\\%", text)
        self.assertEqual(text.count(r"\\"), 2 + 2)  # two header rows and one per result

    def test_figures_render(self):
        import tempfile

        k = [self._kernels(), self._kernels(scene="path/to/capsule_bowl.h5")]
        p = [self._production(k[0]), None]
        with tempfile.TemporaryDirectory() as d:
            for name, fig in (("timeline", plot_timeline(k, p, ["A", "B"], bin_n=2)),
                              ("speed", plot_timeline(k, p, ["A", "B"], slice_by="speed")),
                              ("solvers", plot_solvers([k[0], self._kernels("VBD")], p))):
                path = os.path.join(d, f"{name}.pdf")
                fig.savefig(path, bbox_inches="tight", pad_inches=PAD)
                plt.close(fig)
                self.assertGreater(os.path.getsize(path), 1000)
        self.assertEqual(slice_header({"lo": 0.0, "hi": 10.0}, (10.0, 90.0), "time"), "Fastest 10%")
        self.assertEqual(slice_header({"lo": 90.0, "hi": 100.0}, (10.0, 90.0), "speed"), "Fastest 10%")


if __name__ == "__main__":
    main()
