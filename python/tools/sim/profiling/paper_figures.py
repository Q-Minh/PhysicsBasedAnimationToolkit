"""
Paper figures and tables of profiling results (``simulate.py --profile kernels --profile-out``), with the
per-step statistics of ``profiling/step_stats.py``.

Phases are grouped for the paper by :data:`PHASE_GROUPS` (scope name -> group, fixed colors); the rest are
"Other". Phase times are unions over streams per step (concurrent streams count once), which exclude
graph launch overhead and the gaps between kernels; production runs (``simulate.py --profile-out`` without a
backend: graph replay for the VBD family) give the true step times. Production runs are paired with kernels
runs by scene and solver. They are separate runs whose trajectories differ slightly (contact is chaotic and
GPU arithmetic order isn't deterministic), so they are compared as distributions, never step by step.

- ``timeline``: one row per kernels result (e.g. AAAVBD on one scene): GPU time per step over the simulated
  time, stacked by phase group, with the paired production run's step time as a line, and below it the peak
  node speed (``profiling/trajectory.py``) on the same time axis. ``--snapshots TIME=IMAGE ...`` adds a row of
  simulation snapshots above (``profiling/snapshots.py`` renders them), numbered, with the numbers marked at
  their times on the plots (first row only).
- ``solvers``: one row per solver of one scene, in the benchmark figures' solver order. First column, from
  short fixed-budget kernels runs (no early exit, e.g. every solver resumed from the same checkpoint): each
  phase group's share of the GPU time, with the GPU time per sweep at the bar's end. The other columns, from
  full production runs (``--production``), show what each solver's budget buys: the distribution per step of
  the final gradient ratio and of the penetration (feasibility) ratio to their tolerances (box: quartiles,
  whiskers: P10-P90, dot: maximum; log axes, ratios below 1e-3 drawn at 1e-3, a dashed line at the
  tolerance), the fraction of early exits and, with ``--step-times``, the step time.
- ``table``: a LaTeX ``table*`` (booktabs) with one row per solver and scene: production step time (mean,
  median, P90), the kernel union's median and its ratio to production, the overlap of phases across
  streams (kernels runs), early exits, median iterations, time per iteration and the worst residual ratios
  (production runs when given, else kernels runs).
- ``slices``: Markdown, per kernels result: the steps grouped by percentiles of their GPU time
  (``--slices``, default the fastest 10%, P10-P90 and the slowest 10%) with each group's median step time,
  median iterations and early exits, for the paper's text.
- ``sweeps``: the ``solvers`` figure's first column for several scenes, one panel per scene.
- ``budget``: the ``solvers`` figure's other columns for several scenes, one block of rows per scene.
- ``scenes``: a LaTeX ``table*`` with one row per production result, grouped by scene: step time (mean,
  median, P90), median iterations (VBD sweeps, or Newton steps with the median of a step's most CG
  iterations of one linear solve), early exits, and the median and worst gradient and penetration ratios.

Usage (from the repo root):

    python -m python.tools.sim.profiling.paper_figures timeline k_capsules.json --production p_capsules.json \
        -o data/figures/profiling/timeline [--labels capsule_bowl="Capsules into bowl"] [--width 7] [--bin 5] \
        [--snapshots 0.9=snap_1.png 1.4=snap_2.png ...]
    python -m python.tools.sim.profiling.paper_figures solvers k_fixed_*.json --production p_*.json \
        -o data/figures/profiling/solvers [--step-times]
    python -m python.tools.sim.profiling.paper_figures table k_*.json --production p_*.json \
        -o data/tables/profiling.tex [--labels ...]
    python -m python.tools.sim.profiling.paper_figures slices k_*.json -o data/tables/slices.md [--slices 10,90]
    python -m python.tools.sim.profiling.paper_figures sweeps k_fixed_*.json -o data/figures/profiling/sweeps
    python -m python.tools.sim.profiling.paper_figures budget p_*.json -o data/figures/profiling/budget [--step-times]
    python -m python.tools.sim.profiling.paper_figures scenes p_*.json -o data/tables/scenes.tex [--labels ...]

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
from ..benchmark_plot import SOLVER_COLORS, SOLVER_NAMES, use_paper_fonts  # noqa: E402

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


def slice_header(s: dict, edges: tuple[float, ...]) -> str:
    """A slice of the steps by step time: ``Fastest 10%`` / ``Slowest 10%`` for the outer slices, else its
    percentile range."""
    if s["lo"] == 0.0:
        return f"Fastest {edges[0]:g}%"
    if s["hi"] == 100.0:
        return f"Slowest {100.0 - edges[-1]:g}%"
    return f"P{s['lo']:g}–P{s['hi']:g}"


def _fmt_ms(ms: float | None) -> str:
    if ms is None or not np.isfinite(ms):
        return "–"
    return f"{ms:.3g} ms" if ms < 1000 else f"{ms / 1000:.3g} s"


def _iterations_text(summary: dict, unit: str) -> str:
    it = summary["iterations"]
    return "" if it is None or not unit else f"{it:.0f} {unit}"


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


def _marker(ax, x: float, number: int, y: float = 1.0):
    """A snapshot's number in a small circle at data x and axes y."""
    ax.text(x, y, str(number), transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=FONT_SIZE - 1,
            fontweight="bold", color=INK, clip_on=False,
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": "white", "edgecolor": INK, "linewidth": 0.5})


def plot_timeline(kernels: list[dict], production: list[dict | None], labels: list[str], width: float = 7.0,
                  bin_n: int = 1, snapshots: list[tuple[float, str]] | None = None):
    """The ``timeline`` figure (see the module docstring): one row per kernels result, the first one under its
    ``snapshots`` (simulated time, image path), numbered at their times on the plots."""
    tables = [step_stats.per_step(k) for k in kernels]
    rows = len(kernels)
    has_speed = [bool(np.isfinite(t["peak_speed"]).any()) for t in tables]
    snapshots = sorted(snapshots or [])
    images = [plt.imread(path) for _, path in snapshots]
    image_height = width / len(images) * images[0].shape[0] / images[0].shape[1] if images else 0.0
    heights = []
    for speed in has_speed:
        heights += [1.5, 0.5 if speed else 0.001]
    fig = plt.figure(figsize=(width, image_height + sum(heights) + 0.45 * rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[max(image_height, 1e-3), sum(heights) + 0.45 * rows], hspace=0.12)
    if images:
        strip_of_images = outer[0].subgridspec(1, len(images), wspace=0.03)
        for i, image in enumerate(images):
            ax = fig.add_subplot(strip_of_images[i])
            ax.imshow(image)
            ax.set_axis_off()
            ax.text(0.04, 0.96, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    fontweight="bold", color=INK)
    grid = outer[1].subgridspec(2 * rows, 1, height_ratios=heights, hspace=0.15)
    present: list[str] = []
    drew_production = False
    for r, (result, table, prod, label) in enumerate(zip(kernels, tables, production, labels)):
        groups = group_phases(table["phases"])
        present += [g for g in groups if g not in present]
        x = bin_steps(step_seconds(result, table["t"]), bin_n)
        area = fig.add_subplot(grid[2 * r])
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
        if rows > 1 or not images:
            area.text(0, 1.04, label, transform=area.transAxes, ha="left", va="bottom", fontsize=8,
                      fontweight="bold", color=INK)
        _style(area)
        axes = [area]
        if has_speed[r]:
            area.tick_params(labelbottom=False)
            strip = fig.add_subplot(grid[2 * r + 1], sharex=area)
            strip.plot(x, bin_steps(table["peak_speed"], bin_n), color=INK, linewidth=0.6)
            strip.set_ylim(bottom=0)
            strip.set_ylabel("Speed\n(m/s)", fontsize=FONT_SIZE)
            _style(strip)
            axes.append(strip)
        if r == rows - 1:
            axes[-1].set_xlabel("Time (s)" if "dt" in result["metadata"] else "Time step", fontsize=FONT_SIZE)
        if r == 0:
            for i, (t, _) in enumerate(snapshots):
                for ax in axes:
                    ax.axvline(t, color=INK, linewidth=0.5, linestyle=(0, (2, 2)))
                _marker(area, t, i + 1)
    handles = [Patch(color=group_color(g), label=g) for g in present]
    if drew_production:
        handles.append(Line2D([], [], color=INK, linewidth=0.6, label="Step time (graph replay)"))
    _legend(fig, handles, ncol=min(len(handles), 6))
    return fig


def slices_text(kernels: list[dict], labels: dict[str, str], edges: tuple[float, ...] = (10.0, 90.0)) -> str:
    """Markdown: per kernels result, its steps grouped by percentiles of their GPU time (default: the fastest
    10%, P10-P90, the slowest 10%), with each group's median step time, median iterations and early exits."""
    out = []
    for k in kernels:
        table = step_stats.per_step(k)
        out += [f"## {labels.get(scene_key(k), scene_key(k))}, {solver_name(k)}", "",
                "| Steps | Count | Median step time | Median iterations | Early exits |", "| --- | --- | --- | --- | --- |"]
        for sl in step_stats.percentile_slices(table["time_ms"], edges):
            summary = step_stats.summarize_slice(table, sl["mask"])
            early = "-" if summary["early_exit"] is None else f"{100 * summary['early_exit']:.0f}%"
            out.append(f"| {slice_header(sl, edges)} | {summary['steps']} | {_fmt_ms(summary['time_ms'])} | "
                       f"{_iterations_text(summary, table['unit']) or '-'} | {early} |")
        out.append("")
    return "\n".join(out)


def solver_rank(result: dict) -> tuple[int, str]:
    """Sort key: the benchmark figures' solver order (VBD, AAAVBD, Chebyshev, Anderson, Newton), then others."""
    solver = str(result["metadata"].get("solver", ""))
    order = list(SOLVER_COLORS)
    return (order.index(solver) if solver in order else len(order), solver)


def solver_rows(kernels: list[dict], production: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """(kernels, production) pairs by scene and solver, either side possibly None, sorted by scene (order of
    first appearance) and :func:`solver_rank`."""
    rows: dict[tuple, list] = {}
    for i, group in enumerate((kernels, production)):
        for r in group:
            rows.setdefault((scene_key(r), r["metadata"].get("solver")), [None, None])[i] = r
    scenes = list(dict.fromkeys(k for k, _ in rows))
    pairs = [tuple(v) for v in rows.values()]
    return sorted(pairs, key=lambda kp: (scenes.index(scene_key(kp[0] or kp[1])), solver_rank(kp[0] or kp[1])))


def _sweeps_panel(ax, runs: list[dict | None], y: np.ndarray) -> list[str]:
    """Bars of each phase group's share of :func:`ms_per_sweep` per run (None: no bar) at rows ``y``, with the
    total GPU time per sweep at the bars' ends. Returns the groups drawn."""
    present: list[str] = []
    for yi, r in zip(y, runs):
        if r is None:
            continue
        per_sweep = ms_per_sweep(r)
        total = sum(per_sweep.values())
        left = 0.0
        for g, ms in per_sweep.items():
            present += [g] if g not in present else []
            share = 100.0 * ms / total if total > 0 else 0.0
            ax.barh(yi, share, left=left, height=0.62, color=group_color(g), edgecolor="white", linewidth=0.6)
            left += share
        ax.text(102, yi, f"{total:.3g}", va="center", ha="left", fontsize=FONT_SIZE - 1, color=INK)
    ax.text(102, len(y) - 0.45, "ms/sweep", va="bottom", ha="left", fontsize=FONT_SIZE - 1, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of GPU time (%)", fontsize=FONT_SIZE)
    return present


BUDGET_COLUMNS = ("gradient", "feasibility", "early", "time")


def _budget_panel(ax, column: str, tables: list[dict | None], y: np.ndarray):
    """One :data:`BUDGET_COLUMNS` quantity per row from production runs' per-step tables (None: nothing)."""
    for yi, table in zip(y, tables):
        if table is None:
            continue
        if column == "gradient":
            _box(ax, yi, table["gradient_ratio"], RATIO_FLOOR)
        elif column == "feasibility":
            _box(ax, yi, table["feasibility_ratio"], RATIO_FLOOR)
        elif column == "early":
            summary = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
            if summary["early_exit"] is not None:
                early = 100.0 * summary["early_exit"]
                ax.barh(yi, early, height=0.5, color="#8c8c8c", linewidth=0)
                ax.text(early + 3, yi, f"{early:.0f}%", va="center", ha="left", fontsize=FONT_SIZE - 1, color=INK)
        else:
            _box(ax, yi, table["time_ms"])
    if column in ("gradient", "feasibility"):
        _log_axis(ax)
        ax.axvline(1.0, color=INK, linewidth=0.5, linestyle=(0, (2, 2)))
        ax.set_xlabel("Gradient ratio per step" if column == "gradient" else "Penetration ratio per step",
                      fontsize=FONT_SIZE)
    elif column == "early":
        ax.set_xlim(0, 100)
        ax.set_xlabel("Early exits (%)", fontsize=FONT_SIZE)
    else:
        _log_axis(ax)
        ax.set_xlabel("Step time (ms)", fontsize=FONT_SIZE)


def _row_axes(ax, y: np.ndarray, names: list[str] | None):
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_yticks(y, names if names is not None else [])
    _style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.patch.set_visible(False)  # bar-end labels may reach into the next panel


def _box_handles() -> list:
    return [Patch(facecolor="#dcdcdc", edgecolor=INK, linewidth=0.5, label="Quartiles"),
            Line2D([], [], color=INK, linewidth=0.6, label="P10\u2013P90"),
            Line2D([], [], color=INK, marker="o", markersize=2, linestyle="none", label="Maximum"),
            Line2D([], [], color=INK, linewidth=0.5, linestyle=(0, (2, 2)), label="Tolerance")]


def plot_solvers(kernels: list[dict], production: list[dict], width: float = 7.0, step_times: bool = True):
    """The ``solvers`` figure (see the module docstring): one row per solver of one scene, the first column
    from fixed-budget kernels runs, the others from full production runs."""
    rows = solver_rows(kernels, production)
    y = np.arange(len(rows))[::-1]
    columns = BUDGET_COLUMNS if step_times else BUDGET_COLUMNS[:-1]
    ratios = [2.0] + [{"early": 1.0}.get(c, 1.5) for c in columns]
    fig, axes = plt.subplots(1, 1 + len(columns), figsize=(width, 0.3 * len(rows) + 0.85),
                             gridspec_kw={"width_ratios": ratios, "wspace": 0.28})
    names = [solver_name(k or p) for k, p in rows]
    present = _sweeps_panel(axes[0], [k for k, _ in rows], y)
    _row_axes(axes[0], y, names)
    tables = [step_stats.per_step(p) if p is not None else None for _, p in rows]
    for ax, column in zip(axes[1:], columns):
        _budget_panel(ax, column, tables, y)
        _row_axes(ax, y, None)
    _legend(fig, [Patch(color=group_color(g), label=g) for g in present] + _box_handles(), ncol=6)
    return fig


def table_row(kernels: dict | None, production: dict | None) -> dict:
    """One row of the ``table``: step time statistics (production), the kernel union's median and its ratio
    to production's, the phases' overlap across streams (kernels), and the solve outcome statistics and time
    per iteration (production if given, else kernels). Either run may be None."""
    ktable = step_stats.per_step(kernels) if kernels is not None else None
    ptable = step_stats.per_step(production) if production is not None else None
    table = ptable if ptable is not None else ktable
    s = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
    union = step_stats.describe(ktable["time_ms"]) if ktable is not None else None
    prod = step_stats.describe(ptable["time_ms"]) if ptable is not None else None
    ratio = None
    if union is not None and prod is not None:
        ratio = union["median"] / max(prod["median"], 1e-9)
    return {
        "scene": scene_key(kernels or production),
        "solver": solver_name(kernels or production),
        "production": prod,
        "union": union,
        "union_ratio": ratio,
        "overlap": step_stats.union_overlap(ktable) if ktable is not None else None,
        "early_exit": s["early_exit"],
        "iterations": s["iterations"],
        "unit": table["unit"],
        "ms_per_iteration": s["ms_per_iteration"],
        "gradient_ratio_max": s["gradient_ratio_max"],
        "feasibility_ratio_max": s["feasibility_ratio_max"],
    }


def _num(x, digits: int = 3) -> str:
    """``digits`` significant digits, without an exponent at or above 10^digits (1943, not 1.94e+03)."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.0f}" if abs(x) >= 10**digits else f"{x:.{digits}g}"


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


RATIO_FLOOR = 1e-3  # ratios below it (e.g. no penetration at all) are drawn at it, on log axes


def by_scene(runs: list[dict]) -> dict[str, list[dict]]:
    """Results grouped by :func:`scene_key`, in order of first appearance."""
    out: dict[str, list[dict]] = {}
    for r in runs:
        out.setdefault(scene_key(r), []).append(r)
    return out


def ms_per_sweep(result: dict) -> dict[str, float]:
    """Each phase group's GPU time per iteration (VBD sweep): its total over the steps divided by the
    steps' total iterations."""
    table = step_stats.per_step(result)
    iterations = float(np.nansum(table["iterations"]))
    if iterations <= 0:
        return {}
    return {g: float(v.sum()) / iterations for g, v in group_phases(table["phases"]).items()}


def _box(ax, y: float, values: np.ndarray, floor: float | None = None, height: float = 0.5):
    """A horizontal box: quartiles, P10-P90 whiskers, a median tick and the maximum as a dot (values below
    ``floor`` drawn at it)."""
    v = np.asarray(values, dtype=float)
    if floor is not None:
        v = np.maximum(v, floor)
    d = step_stats.describe(v)
    if d is None:
        return
    ax.plot([d["P10"], d["P90"]], [y, y], color=INK, linewidth=0.6)
    ax.barh(y, d["Q3"] - d["Q1"], left=d["Q1"], height=height, color="#dcdcdc", edgecolor=INK, linewidth=0.5)
    ax.plot([d["median"]] * 2, [y - height / 2, y + height / 2], color=INK, linewidth=1.0)
    ax.plot(d["max"], y, marker="o", markersize=2, color=INK, linestyle="none")


def _log_axis(ax):
    """Log x axis with plain-number ticks: 1-2-5 within a narrow range, decades over a wide one (call after
    drawing, so the limits are known)."""
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0) if hi / lo < 300 else (1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_sweeps(runs: list[dict], labels: dict[str, str], width: float = 7.0):
    """The ``sweeps`` figure (see the module docstring): one panel per scene, one stacked bar per solver."""
    scenes = {k: sorted(v, key=solver_rank) for k, v in by_scene(runs).items()}
    n_rows = max(len(v) for v in scenes.values())
    fig, axes = plt.subplots(1, len(scenes), figsize=(width, 0.32 * n_rows + 0.7), squeeze=False,
                             gridspec_kw={"wspace": 0.35})
    present: list[str] = []
    for ax, (scene, rows) in zip(axes[0], scenes.items()):
        y = np.arange(len(rows))[::-1]
        present += [g for g in _sweeps_panel(ax, rows, y) if g not in present]
        _row_axes(ax, y, [solver_name(r) for r in rows])
        ax.text(0, 1.04, labels.get(scene, scene), transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                fontweight="bold", color=INK)
    _legend(fig, [Patch(color=group_color(g), label=g) for g in present], ncol=6)
    return fig


def plot_budget(runs: list[dict], labels: dict[str, str], width: float = 7.0, step_times: bool = False):
    """The ``budget`` figure (see the module docstring): one block of rows per scene (a row per solver),
    a column per quantity, all from the steps' solve statuses (and step times)."""
    scenes = {k: sorted(v, key=solver_rank) for k, v in by_scene(runs).items()}
    columns = BUDGET_COLUMNS if step_times else BUDGET_COLUMNS[:-1]
    n_rows = [len(v) for v in scenes.values()]
    fig = plt.figure(figsize=(width, 0.3 * sum(n_rows) + 0.55 * len(scenes) + 0.3))
    grid = fig.add_gridspec(len(scenes), len(columns), height_ratios=n_rows, hspace=0.9, wspace=0.3,
                            width_ratios=[{"early": 1.0}.get(c, 1.6) for c in columns])
    for s, (scene, rows) in enumerate(scenes.items()):
        y = np.arange(len(rows))[::-1]
        tables = [step_stats.per_step(r) for r in rows]
        axes = [fig.add_subplot(grid[s, c]) for c in range(len(columns))]
        for ax, column in zip(axes, columns):
            _budget_panel(ax, column, tables, y)
            _row_axes(ax, y, [solver_name(r) for r in rows] if ax is axes[0] else None)
        axes[0].text(0, 1.06, labels.get(scene, scene), transform=axes[0].transAxes, ha="left", va="bottom",
                     fontsize=8, fontweight="bold", color=INK)
    _legend(fig, _box_handles())
    return fig


def scenes_row(production: dict) -> dict:
    """One row of the ``scenes`` table: a production run's step times, iterations, early exits and residual
    ratios (median and worst), from its steps' solve statuses."""
    table = step_stats.per_step(production)
    status = [s.get("status") or {} for s in production["steps"]]
    cg = [max((r["max_cg_iters"] for r in st["newton"]), default=0) for st in status if "newton" in st]

    def stat(key, name):
        d = step_stats.describe(table[key])
        return d[name] if d else None

    summary = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
    return {
        "scene": scene_key(production),
        "solver": solver_name(production),
        "steps": int(table["t"].size),
        "time": step_stats.describe(table["time_ms"]),
        "iterations": stat("iterations", "median"),
        "unit": table["unit"],
        "max_cg": float(np.median(cg)) if cg else None,
        "early_exit": summary["early_exit"],
        "gradient_median": stat("gradient_ratio", "median"),
        "gradient_max": stat("gradient_ratio", "max"),
        "feasibility_median": stat("feasibility_ratio", "median"),
        "feasibility_max": stat("feasibility_ratio", "max"),
    }


def latex_scenes(rows: list[dict], labels: dict[str, str]) -> str:
    """The ``scenes`` table (booktabs, multirow): rows grouped by scene, numbers to 3 significant digits."""
    out = [r"\begin{table*}", r"\centering", r"\small",
           r"\caption{\textbf{Expected efficiency per time step.} Production runs (CUDA graph replay for the VBD "
           r"family; Newton runs eagerly with host synchronization) over each scene's trajectory. Iterations: "
           r"median VBD sweeps, or Newton steps per time step, with the median of the most conjugate gradient "
           r"iterations of one linear solve in parentheses; early exits: time steps whose solve passed its "
           r"convergence tests within the budget; residual ratios to their tolerances per time step (median and "
           r"worst; at most 1 passes): the gradient and the penetration into the contact margin.}",
           r"\label{tab:scenes}",
           r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r"& & \multicolumn{3}{c}{Step time (ms)} & & & \multicolumn{2}{c}{Gradient ratio} & "
           r"\multicolumn{2}{c}{Penetration ratio} \\",
           r"\cmidrule(lr){3-5} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
           r"Scene & Solver & mean & median & P90 & Iterations & Early exits & median & worst & median & worst \\",
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
        t = row["time"] or {}
        iterations = "--" if row["iterations"] is None else f"{row['iterations']:.0f}"
        if row["max_cg"] is not None:
            iterations += f" ({row['max_cg']:.0f})"
        early = "--" if row["early_exit"] is None else f"{100 * row['early_exit']:.0f}\\%"
        cells = [scene, row["solver"], _num(t.get("mean")), _num(t.get("median")), _num(t.get("P90")), iterations,
                 early, _num(row["gradient_median"]), _num(row["gradient_max"]), _num(row["feasibility_median"]),
                 _num(row["feasibility_max"])]
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
    parser.add_argument("kind", choices=["timeline", "solvers", "table", "slices", "sweeps", "budget", "scenes"],
                        help="What to write (see the docstring).")
    parser.add_argument("kernels", nargs="+",
                        help="Results (.json), one per row: kernels-backend runs (timeline, solvers, table, sweeps) "
                        "or production runs (budget, scenes).")
    parser.add_argument("--production", nargs="*", default=[],
                        help="Production results (.json), paired with the kernels results by scene and solver.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output: <output>.pdf for figures, the .tex or .md file for tables.")
    parser.add_argument("--labels", nargs="*", default=[], metavar="SCENE=LABEL",
                        help="Row labels per scene key (the scene file's name without extension), e.g. "
                        "basket_of_fruit=\"Basket of fruit\" (default: the key).")
    parser.add_argument("--width", type=float, default=7.0,
                        help="Printed width in inches (7: TOG's text width, 3.33: a column); fonts at nominal size.")
    parser.add_argument("--bin", type=int, default=1, dest="bin_n",
                        help="timeline: average the time series over blocks of this many steps.")
    parser.add_argument("--snapshots", nargs="*", default=[], metavar="TIME=IMAGE",
                        help="timeline: snapshot images and their simulated times (s), e.g. 0.9=snap_1.png.")
    parser.add_argument("--slices", default="10,90",
                        help="slices: percentile edges of the step-time slices (default: 10,90).")
    parser.add_argument("--step-times", action="store_true", dest="step_times",
                        help="solvers, budget: add a column with the step time distribution.")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Directory with the Linux Libertine OpenType files (default: PBAT_FONT_DIR, then the "
                        "installed fonts).")
    args = parser.parse_args()
    kernels = [results.load(p) for p in args.kernels]
    production_runs = [results.load(p) for p in args.production]
    production = pair_production(kernels, production_runs)
    labels = _parse_labels(args.labels)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.kind in ("table", "scenes", "slices"):
        if args.kind == "table":
            text = latex_table([table_row(k, p) for k, p in solver_rows(kernels, production_runs)], labels)
        elif args.kind == "scenes":
            text = latex_scenes([scenes_row(k) for k in kernels], labels)
        else:
            text = slices_text(kernels, labels, tuple(float(e) for e in args.slices.split(",")))
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text if args.kind == "slices" else "% Needs \\usepackage{booktabs, multirow}\n\n" + text)
        print(f"Wrote {args.output}")
        return
    use_paper_fonts(args.font_dir)
    if args.kind == "timeline":
        snapshots = []
        for item in args.snapshots:
            time, sep, path = item.partition("=")
            if not sep:
                raise SystemExit(f"--snapshots expects TIME=IMAGE, got {item!r}")
            snapshots.append((float(time), path))
        fig = plot_timeline(kernels, production, [labels.get(scene_key(k), scene_key(k)) for k in kernels],
                            args.width, args.bin_n, snapshots)
    elif args.kind == "solvers":
        fig = plot_solvers(kernels, production_runs, args.width, args.step_times)
    elif args.kind == "sweeps":
        fig = plot_sweeps(kernels, labels, args.width)
    else:
        fig = plot_budget(kernels, labels, args.width, args.step_times)
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
        steps = [{"t": s["t"], "step_timeline_ms": s["step_union_ms"] * 1.08, "status": s["status"]}
                 for s in kernels["steps"]]
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

    def test_sweeps_budget_scenes(self):
        import tempfile

        k = self._kernels()
        per_sweep = ms_per_sweep(k)
        # 15 steps without contact, 25 with; 3200 sweeps (every 4th step exits early at 20)
        self.assertAlmostEqual(per_sweep["Vertex sweeps"], (15 * 8.0 + 25 * 12.0) / 3200)
        self.assertAlmostEqual(per_sweep["Contact detection"], (15 * 0.5 + 25 * 2.0) / 3200)
        newton = self._production(k)
        newton["metadata"]["solver"] = "Newton"
        for i, s in enumerate(newton["steps"]):
            s["status"] = {"newton": [{"substep": 0, "outer": 3, "converged": i % 2 == 0, "newton_steps": 6,
                                       "max_cg_iters": 100 + i, "gradient_ratio": 0.5, "feasibility_ratio": 0.0}]}
        row = scenes_row(newton)
        self.assertEqual((row["solver"], row["iterations"], row["early_exit"]), ("Newton", 6.0, 0.5))
        self.assertAlmostEqual(row["max_cg"], 100 + 19.5)
        vbd = self._production(self._kernels("VBD"))
        rows = solver_rows([k], [self._production(k), vbd])
        self.assertEqual([(a is None, solver_name(a or b)) for a, b in rows], [(True, "VBD"), (False, "AAAVBD")])
        only_production = table_row(None, vbd)
        self.assertIsNone(only_production["union"])
        self.assertIsNone(only_production["overlap"])
        self.assertAlmostEqual(only_production["production"]["max"], 14.45 * 1.08)
        aaavbd = scenes_row(self._production(k))
        self.assertEqual((aaavbd["early_exit"], aaavbd["max_cg"]), (0.25, None))
        text = latex_scenes([row, aaavbd], {})
        self.assertIn("6 (120)", text)
        self.assertIn(r"\multirow{2}{*}{basket\_of\_fruit}", text)
        with tempfile.TemporaryDirectory() as d:
            runs = [k, self._kernels("VBD"), self._kernels(scene="path/to/capsule_bowl.h5")]
            for name, fig in (("sweeps", plot_sweeps(runs, {})), ("budget", plot_budget(runs, {}, step_times=True))):
                path = os.path.join(d, f"{name}.pdf")
                fig.savefig(path, bbox_inches="tight", pad_inches=PAD)
                plt.close(fig)
                self.assertGreater(os.path.getsize(path), 1000)

    def test_figures_render(self):
        import tempfile

        k = [self._kernels(), self._kernels(scene="path/to/capsule_bowl.h5")]
        p = [self._production(k[0]), None]
        with tempfile.TemporaryDirectory() as d:
            image = os.path.join(d, "snap.png")
            plt.imsave(image, np.ones((30, 40, 3)))
            for name, fig in (("timeline", plot_timeline(k, p, ["A", "B"], bin_n=2)),
                              ("snapshots", plot_timeline(k[:1], p[:1], ["A"], snapshots=[(0.3, image), (0.2, image)])),
                              ("solvers", plot_solvers([self._kernels("VBD"), k[0]], [p[0]], step_times=True))):
                path = os.path.join(d, f"{name}.pdf")
                fig.savefig(path, bbox_inches="tight", pad_inches=PAD)
                plt.close(fig)
                self.assertGreater(os.path.getsize(path), 1000)
        self.assertEqual(slice_header({"lo": 0.0, "hi": 10.0}, (10.0, 90.0)), "Fastest 10%")
        self.assertEqual(slice_header({"lo": 90.0, "hi": 100.0}, (10.0, 90.0)), "Slowest 10%")
        text = slices_text(k[:1], {"basket_of_fruit": "Basket"})
        self.assertIn("## Basket, AAAVBD", text)
        self.assertIn("| Fastest 10% | 15 | 8.95 ms | 100 sweeps | 27% |", text)  # ties: every no-contact step


if __name__ == "__main__":
    main()
