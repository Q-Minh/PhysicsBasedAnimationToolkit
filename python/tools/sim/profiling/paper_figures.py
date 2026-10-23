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
  time, stacked by phase group (groups under 3% of the row's GPU time fold into a top "Other" band, which has
  a legend entry only if it reaches 3% itself), with the step time of the paired production run and of the
  scene's other production runs (e.g. VBD) as lines in their solvers' colors. Below it, on the same time axis,
  the production runs' final gradient ratio to its tolerance per step (log axis, its rolling median over
  ``--smooth`` steps, a dashed line at 1), over the peak node speed (``profiling/trajectory.py``) on a right
  axis: the paired run's as a light fill (else the kernels run's), the other runs' as dotted lines, each with
  its own legend entry. Without solve statuses the strip shows the speed only. ``--snapshots TIME=IMAGE ...``
  adds a row of simulation snapshots above (``profiling/snapshots.py`` renders them), numbered, with the
  numbers marked at their times on the plots (first row only).
- ``solvers``: one row per solver of one scene, in the benchmark figures' solver order. First column, from
  short fixed-budget kernels runs (no early exit, e.g. every solver resumed from the same checkpoint): each
  phase group's share of the GPU time, with the GPU time per sweep at the bar's end. Groups shared by every
  solver and negligible (:data:`SOLVERS_OMITTED`) are left out without renormalizing, so the bars end short of
  100%. The other columns, from full production runs (``--production``), show what each solver's budget buys:
  the distribution per step of the final gradient ratio and of the penetration (feasibility) ratio to their
  tolerances (box: quartiles, whiskers: P10-P90, dot: maximum; log axes, ratios below 1e-3 drawn at 1e-3, a
  dashed line at the tolerance), the sweeps per step (linear axis) and, with ``--step-times``, the step time.
  Sweep counts are quantized by the convergence test, which runs every ``Solver.Convergence.check_interval``
  sweeps and after each subproblem (20 sweeps on CapsulesIntoBowl, so every count is a multiple of 20). Early
  exits aren't shown: their fraction doesn't say how soon a solver exits.
- ``slices``: Markdown, per kernels result: the steps grouped by percentiles of their GPU time
  (``--slices``, default the fastest 10%, P10-P90 and the slowest 10%) with each group's median step time,
  median iterations and early exits, for the paper's text.
- ``sweeps``: the ``solvers`` figure's first column for several scenes, one panel per scene.
- ``budget``: the ``solvers`` figure's other columns for several scenes, one block of rows per scene.
- ``scenes``: the paper's LaTeX ``table*`` of production runs (graph replay for the VBD family, eager for
  Newton): two lines per scene (its nodes, tetrahedra and time step h), a column per solver (VBD, AAAVBD,
  Chebyshev, Anderson, Newton; -- without a run): the mean step time in ms with the P90 in parentheses, then
  the median iterations per step (VBD sweeps, Newton steps) and the fraction of early exits.

Usage (from the repo root):

    python -m python.tools.sim.profiling.paper_figures timeline k_capsules.json --production p_capsules.json \
        p_capsules_vbd.json -o data/figures/profiling/timeline [--labels capsule_bowl="Capsules into bowl"] \
        [--width 7] [--bin 5] [--smooth 9] [--snapshots 0.9=snap_1.png 1.4=snap_2.png ...]
    python -m python.tools.sim.profiling.paper_figures solvers k_fixed_*.json --production p_*.json \
        -o data/figures/profiling/solvers [--step-times]
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
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from . import results, step_stats  # noqa: E402
from ..benchmark_plot import SOLVER_COLORS, SOLVER_NAMES, use_paper_fonts  # noqa: E402

# The paper's phase groups, in drawing order (bottom of the stack first): name -> (scope names, color). Scope
# names are innermost scopes (results.leaf); a scope in no group is drawn as OTHER. AAAVBD's acceleration runs
# inside its vertex-solve kernel (aaavbd.sweep), so it has no group: its cost shows in its time per sweep
# against VBD's. Anderson's history updates run directly in its subproblem scope (anderson.subproblem's self
# time), so they count as Mixing. Colors: muted olive, khaki, slate and brown tones, none sharing a hue with a solver's color
# (benchmark_plot.SOLVER_COLORS), so phase fills never read as the solver lines drawn over them. Sweeps, detection
# and Other are at least 14 apart in OKLab (x100) under normal vision and simulated color vision deficiencies
# (Machado et al. 2009), every pair at least 4.9.
PHASE_GROUPS = {
    "Vertex sweeps": (("vbd.sweep", "aaavbd.sweep"), "#77764f"),
    "Contact detection": (("contact.detect", "contact.filter", "contact.constraint_set"), "#ded5b3"),
    "Contact update": (("contact.dual", "contact.penalty", "newton.adapt_penalty"), "#3f3f30"),
    "Residuals": (("newton.gradient", "newton.convergence"), "#6f8094"),
    "Mixing": (("chebyshev.mix", "anderson.subproblem", "anderson.mix", "anderson.gram", "anderson.solve"),
               "#9c7f66"),
    "Newton CG": (("newton.linear_solve",), "#5c6b70"),
    "Newton assembly, line search": (("newton.assemble", "newton.line_search", "newton.energy"), "#f0ede4"),
}
OTHER = "Other"
OTHER_COLOR = "#a49e94"
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


def fold_small_groups(groups: dict[str, np.ndarray], share: float) -> dict[str, np.ndarray]:
    """``groups`` (:func:`group_phases`) with those under ``share`` of the total time summed into :data:`OTHER`,
    drawn last (on top of the stack)."""
    total = max(sum(float(v.sum()) for v in groups.values()), 1e-12)
    out = {g: v for g, v in groups.items() if g != OTHER and v.sum() >= share * total}
    rest = [v for g, v in groups.items() if g not in out]
    if rest:
        out[OTHER] = np.sum(rest, axis=0)
    return out


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


def other_production(kernels: list[dict], production: list[dict]) -> list[list[dict]]:
    """For each kernels result, the production results of the same scene and other solvers, in the benchmark
    figures' solver order."""
    return [sorted((p for p in production if scene_key(p) == scene_key(k)
                    and p["metadata"].get("solver") != k["metadata"].get("solver")), key=solver_rank)
            for k in kernels]


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


SMOOTH_STEPS = 9  # timeline: the rolling median's window over the per-step residual ratios
# timeline: phase groups under this share of a row's GPU time fold into Other, which gets a legend entry only if
# it reaches it too (thinner bands are under ~1 pt at print size)
TIMELINE_MIN_SHARE = 0.03
SPEED_FILL = "#d9d9d9"  # timeline: the peak speed behind the residual ratios
HALO = [patheffects.withStroke(linewidth=1.8, foreground="white")]  # lines drawn over the phase bands


def rolling_median(values: np.ndarray, n: int) -> np.ndarray:
    """Centered rolling median over ``n`` values (rounded up to odd), ignoring NaNs, with shorter windows at the
    ends; NaN where a window has no value."""
    v = np.asarray(values, dtype=float)
    if n <= 1 or v.size == 0:
        return v
    half = n // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(v, half, constant_values=np.nan), 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        return np.nanmedian(windows, axis=1)


def solver_color(result: dict) -> str:
    return SOLVER_COLORS.get(result["metadata"].get("solver"), INK)


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


SPEED_DOTS = (0, (1, 1.5))  # timeline: the other production runs' peak speeds


def _residual_strip(strip, x: np.ndarray, kernels: dict, kernels_speed: np.ndarray, runs: list[dict],
                    smooth: int) -> list:
    """The timeline's lower strip: each production run's final gradient ratio per step (its rolling median) on a
    log axis, over the peak speed on a right axis (see the module docstring); ``x`` and ``kernels_speed`` are
    the ``kernels`` run's step times (s) and peak speeds, drawn if the runs have none. Returns the speeds'
    legend handles."""
    speed = strip.twinx()
    handles = []
    medians = [np.array([1.0])]  # the y range covers the tolerance line
    for i, run in enumerate(runs):
        table = step_stats.per_step(run)
        xr = step_seconds(run, table["t"])
        if i == 0 and np.isfinite(table["peak_speed"]).any():
            speed.fill_between(xr, table["peak_speed"], color=SPEED_FILL, alpha=0.6, linewidth=0)
            handles.append(Patch(color=SPEED_FILL, alpha=0.6, label=f"{solver_name(run)} peak speed"))
        elif i > 0 and np.isfinite(table["peak_speed"]).any():
            speed.plot(xr, table["peak_speed"], color=solver_color(run), linewidth=0.5, linestyle=SPEED_DOTS)
            handles.append(Line2D([], [], color=solver_color(run), linewidth=0.5, linestyle=SPEED_DOTS,
                                  label=f"{solver_name(run)} peak speed"))
        ratio = np.maximum(table["gradient_ratio"], RATIO_FLOOR)
        medians.append(rolling_median(ratio, smooth))
        strip.plot(xr, medians[-1], color=solver_color(run), linewidth=0.8)
    if not any(isinstance(h, Patch) for h in handles) and np.isfinite(kernels_speed).any():
        speed.fill_between(x, kernels_speed, color=SPEED_FILL, alpha=0.6, linewidth=0)
        handles.insert(0, Patch(color=SPEED_FILL, alpha=0.6, label=f"{solver_name(kernels)} peak speed"))
    strip.axhline(1.0, color=INK, linewidth=0.5, linestyle=(0, (3, 2)))
    strip.set_yscale("log")
    values = np.concatenate(medians)
    values = values[np.isfinite(values) & (values > 0)]
    strip.set_ylim(values.min() / 1.25, values.max() * 1.25)  # the curves with a margin, not clipped at the edges
    _log_axis(strip, "y")
    strip.set_ylabel("Residual /\ntolerance", fontsize=FONT_SIZE)
    speed.set_ylim(bottom=0)
    speed.set_ylabel("Peak speed\n(m/s)", fontsize=FONT_SIZE, color="#7f7f7f")
    # The ratios over the speed fill: the left axes in front, without their background
    strip.set_zorder(speed.get_zorder() + 1)
    strip.patch.set_visible(False)
    _style(strip)
    for side in ("top", "left", "bottom"):
        speed.spines[side].set_visible(False)
    speed.spines["right"].set_linewidth(0.5)
    speed.tick_params(labelsize=FONT_SIZE, length=2, width=0.5, pad=1.5, colors="#7f7f7f")
    return handles


def plot_timeline(kernels: list[dict], production: list[dict | None], labels: list[str], width: float = 7.0,
                  bin_n: int = 1, snapshots: list[tuple[float, str]] | None = None,
                  others: list[list[dict]] | None = None, smooth: int = SMOOTH_STEPS):
    """The ``timeline`` figure (see the module docstring): one row per kernels result, the first one under its
    ``snapshots`` (simulated time, image path), numbered at their times on the plots. ``others`` are each row's
    production runs of other solvers (:func:`other_production`), drawn over the paired ``production`` run."""
    tables = [step_stats.per_step(k) for k in kernels]
    rows = len(kernels)
    others = others or [[] for _ in kernels]
    runs_per_row = [([prod] if prod is not None else []) + list(more) for prod, more in zip(production, others)]
    has_ratio = [any(np.isfinite(step_stats.per_step(run)["gradient_ratio"]).any() for run in runs)
                 for runs in runs_per_row]
    has_speed = [bool(np.isfinite(t["peak_speed"]).any()) for t in tables]
    snapshots = sorted(snapshots or [])
    images = [plt.imread(path) for _, path in snapshots]
    image_height = width / len(images) * images[0].shape[0] / images[0].shape[1] if images else 0.0
    heights = []
    for ratio, speed in zip(has_ratio, has_speed):
        heights += [1.5, 0.9 if ratio else 0.5 if speed else 0.001]
    fig = plt.figure(figsize=(width, image_height + sum(heights) + 0.45 * rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[max(image_height, 1e-3), sum(heights) + 0.45 * rows], hspace=0.12)
    if images:
        strip_of_images = outer[0].subgridspec(1, len(images), wspace=0.03)
        for i, image in enumerate(images):
            ax = fig.add_subplot(strip_of_images[i])
            ax.imshow(image, interpolation="none")  # embedded in the PDF at its own resolution, not resampled
            ax.set_axis_off()
            ax.text(0.04, 0.96, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    fontweight="bold", color=INK)
    grid = outer[1].subgridspec(2 * rows, 1, height_ratios=heights, hspace=0.15)
    present: list[str] = []  # phase groups with a legend entry
    solvers: dict[str, str] = {}  # drawn production runs: solver name -> color
    speeds: dict[str, object] = {}  # peak speeds' legend handles by label
    for r, (result, table, runs, label) in enumerate(zip(kernels, tables, runs_per_row, labels)):
        groups = fold_small_groups(group_phases(table["phases"]), TIMELINE_MIN_SHARE)
        total = max(sum(float(v.sum()) for v in groups.values()), 1e-12)
        present += [g for g in groups if g not in present and groups[g].sum() >= TIMELINE_MIN_SHARE * total]
        x = bin_steps(step_seconds(result, table["t"]), bin_n)
        area = fig.add_subplot(grid[2 * r])
        area.stackplot(x, *[bin_steps(v, bin_n) for v in groups.values()],
                       colors=[group_color(g) for g in groups], linewidth=0)
        for run in runs:
            ptable = step_stats.per_step(run)
            area.plot(bin_steps(step_seconds(run, ptable["t"]), bin_n), bin_steps(ptable["time_ms"], bin_n),
                      color=solver_color(run), linewidth=0.6, path_effects=HALO)
            solvers.setdefault(solver_name(run), solver_color(run))
        area.set_xlim(x[0], x[-1])
        area.set_ylim(bottom=0)
        area.set_ylabel("GPU time (ms)", fontsize=FONT_SIZE)
        if rows > 1 or not images:
            area.text(0, 1.04, label, transform=area.transAxes, ha="left", va="bottom", fontsize=8,
                      fontweight="bold", color=INK)
        _style(area)
        axes = [area]
        if has_ratio[r] or has_speed[r]:
            area.tick_params(labelbottom=False)
            strip = fig.add_subplot(grid[2 * r + 1], sharex=area)
            if has_ratio[r]:
                for h in _residual_strip(strip, step_seconds(result, table["t"]), result, table["peak_speed"], runs,
                                         smooth):
                    speeds.setdefault(h.get_label(), h)
            else:
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
    handles = [Patch(color=group_color(g), label=g) for g in [*PHASE_GROUPS, OTHER] if g in present]
    handles += [Line2D([], [], color=color, linewidth=0.8, label=name) for name, color in solvers.items()]
    handles += list(speeds.values())
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


def _sweeps_panel(ax, runs: list[dict | None], y: np.ndarray, omit: tuple[str, ...] = ()) -> list[str]:
    """Bars of each phase group's share of :func:`ms_per_sweep` per run (None: no bar) at rows ``y``, with the
    total GPU time per sweep at the bars' ends. Groups in ``omit`` aren't drawn, but still count in the total.
    Returns the groups drawn."""
    present: list[str] = []
    for yi, r in zip(y, runs):
        if r is None:
            continue
        per_sweep = ms_per_sweep(r)
        total = sum(per_sweep.values())
        left = 0.0
        for g, ms in per_sweep.items():
            if g in omit:
                continue
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
    """One :data:`BUDGET_COLUMNS` quantity, or ``"sweeps"`` (iterations per step), per row from production runs'
    per-step tables (None: nothing)."""
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
        elif column == "sweeps":
            _box(ax, yi, table["iterations"])
        else:
            _box(ax, yi, table["time_ms"])
    if column in ("gradient", "feasibility"):
        _log_axis(ax)
        ax.axvline(1.0, color=INK, linewidth=0.5, linestyle=(0, (2, 2)))
        ax.set_xlabel("Residual ratio per step" if column == "gradient" else "Penetration ratio per step",
                      fontsize=FONT_SIZE)
    elif column == "early":
        ax.set_xlim(0, 100)
        ax.set_xlabel("Early exits (%)", fontsize=FONT_SIZE)
    elif column == "sweeps":
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.set_xlabel("Sweeps per step", fontsize=FONT_SIZE)
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


# Phase groups the solvers figure leaves out: in the VBD family, the same code in every solver and under 1% of its
# GPU time (CapsulesIntoBowl, fixed budget: Contact update 0.6%; Residuals 0.3%, 0.6% for VBD, whose only sweep
# block per subproblem ends with an inner test that repeats the outer one; Other 0.2%). Bars aren't renormalized,
# so they end short of 100% by the omitted share. Newton's Contact update and Residuals hold its own scopes
# (newton.adapt_penalty, newton.convergence): check them before plotting a Newton row.
SOLVERS_OMITTED = ("Contact update", "Residuals", OTHER)


def plot_solvers(kernels: list[dict], production: list[dict], width: float = 7.0, step_times: bool = True):
    """The ``solvers`` figure (see the module docstring): one row per solver of one scene, the first column
    from fixed-budget kernels runs, the others from full production runs."""
    rows = solver_rows(kernels, production)
    y = np.arange(len(rows))[::-1]
    # No early exits: their fraction reads as speed, which it isn't (a solver can exit as often as another,
    # but after fewer sweeps). Sweeps per step instead, then the time they cost
    columns = ["gradient", "feasibility", "sweeps"] + (["time"] if step_times else [])
    ratios = [2.4] + [1.5] * len(columns)
    fig, axes = plt.subplots(1, 1 + len(columns), figsize=(width, 0.3 * len(rows) + 0.85),
                             gridspec_kw={"width_ratios": ratios, "wspace": 0.28})
    names = [solver_name(k or p) for k, p in rows]
    present = _sweeps_panel(axes[0], [k for k, _ in rows], y, SOLVERS_OMITTED)
    _row_axes(axes[0], y, names)
    tables = [step_stats.per_step(p) if p is not None else None for _, p in rows]
    for ax, column in zip(axes[1:], columns):
        _budget_panel(ax, column, tables, y)
        _row_axes(ax, y, None)
    _legend(fig, [Patch(color=group_color(g), label=g) for g in present] + _box_handles())
    return fig


def _num(x, digits: int = 3) -> str:
    """``digits`` significant digits, without an exponent at or above 10^digits (1943, not 1.94e+03)."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.0f}" if abs(x) >= 10**digits else f"{x:.{digits}g}"


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


def _log_axis(ax, which: str = "x"):
    """Log x (or y) axis with plain-number ticks: 1-2-5 within a narrow range, decades over a wide one (call
    after drawing, so the limits are known)."""
    getattr(ax, f"set_{which}scale")("log")
    lo, hi = getattr(ax, f"get_{which}lim")()
    axis = getattr(ax, f"{which}axis")
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0) if hi / lo < 300 else (1.0,)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


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


SCENE_SOLVERS = ("VBD", "AAAVBD", "Chebyshev", "AndersonSolver", "Newton")  # the table's columns


def scene_cell(production: dict) -> dict:
    """One solver's cell of the ``scenes`` table, from its production run: the step time's mean and P90 (ms),
    the median iterations per step (VBD sweeps, Newton steps) and the fraction of early exits."""
    table = step_stats.per_step(production)
    time = step_stats.describe(table["time_ms"])
    iterations = step_stats.describe(table["iterations"])
    summary = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
    return {
        "mean": time["mean"] if time else None,
        "P90": time["P90"] if time else None,
        "iterations": iterations["median"] if iterations else None,
        "early_exit": summary["early_exit"],
    }


def _compact(count: float) -> str:
    """A count in at most 3 significant digits, with a k from 10^4 and an M from 10^6: 3269, 54.5k, 1.81M."""
    if count >= 1e6:
        return f"{count / 1e6:.3g}M"
    if count >= 1e4:
        return f"{count / 1e3:.3g}k"
    return f"{count:.0f}"


def latex_scenes(runs: list[dict], labels: dict[str, str]) -> str:
    """The ``scenes`` table (booktabs): two lines per scene, a column per solver of :data:`SCENE_SOLVERS`.
    The scene's nodes, tetrahedra and time step h come from its runs' metadata; per solver, the first line is
    the mean step time (P90) in ms, the second the median iterations per step and the early exits."""
    out = [r"\begin{table*}", r"\centering", r"\small",
           r"\caption{\textbf{GPU time per time step.} Mean time per time step of each solver on the scenes of the "
           r"paper's results and video (production runs: CUDA graph replay for the VBD family, eager for Newton), "
           r"with the 90th percentile in parentheses, and below, the median iterations per time step (VBD sweeps; "
           r"Newton steps) and the fraction of time steps that pass their convergence test within the budget.}",
           r"\label{tab:scenes}",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrr ccccc}", r"\toprule",
           r"& & & & \multicolumn{5}{c}{Time per step, ms (P90) \;/\; iterations per step, early exits} \\",
           r"\cmidrule(lr){5-9}",
           r"Scene & Nodes & Tets & $h$ (ms) & " + " & ".join(SOLVER_NAMES.get(v, v) for v in SCENE_SOLVERS) + r" \\",
           r"\midrule"]
    for scene, rows in by_scene(runs).items():
        by_solver = {r["metadata"].get("solver"): r for r in rows}
        meta = rows[0]["metadata"]
        h = 1e3 * float(meta.get("dt", float("nan"))) / max(int(meta.get("substeps", 1)), 1)
        first = [labels.get(scene, scene.replace("_", r"\_")),
                 _compact(meta["n_nodes"]) if "n_nodes" in meta else "--",
                 _compact(meta["n_elements"]) if "n_elements" in meta else "--", _num(h)]
        second = ["", "", "", ""]
        for solver in SCENE_SOLVERS:
            if solver not in by_solver:
                first.append("--")
                second.append("--")
                continue
            c = scene_cell(by_solver[solver])
            first.append(f"{_num(c['mean'])} ({_num(c['P90'])})")
            iterations = "--" if c["iterations"] is None else f"{c['iterations']:.0f}"
            early = "--" if c["early_exit"] is None else f"{100 * c['early_exit']:.0f}\\%"
            second.append(f"{iterations}, {early}")
        out += [" & ".join(first) + r" \\", " & ".join(second) + r" \\"]
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
    parser.add_argument("kind", choices=["timeline", "solvers", "slices", "sweeps", "budget", "scenes"],
                        help="What to write (see the docstring).")
    parser.add_argument("kernels", nargs="+",
                        help="Results (.json), one per row: kernels-backend runs (timeline, solvers, slices, sweeps) "
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
    parser.add_argument("--smooth", type=int, default=SMOOTH_STEPS,
                        help="timeline: steps in the rolling median over the residual ratios.")
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
    if args.kind in ("scenes", "slices"):
        if args.kind == "scenes":
            text = "% Needs \\usepackage{booktabs}\n\n" + latex_scenes(kernels, labels)
        else:
            text = slices_text(kernels, labels, tuple(float(e) for e in args.slices.split(",")))
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
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
                            args.width, args.bin_n, snapshots, other_production(kernels, production_runs),
                            args.smooth)
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
        metadata = {"scene": scene, "solver": solver, "dt": 0.01, "substeps": 1, "n_nodes": 54502,
                    "n_elements": 231338}
        return {"source": "kernels", "metadata": metadata, "steps": steps}

    @staticmethod
    def _production(kernels: dict):
        steps = [{"t": s["t"], "step_timeline_ms": s["step_union_ms"] * 1.08, "status": s["status"]}
                 for s in kernels["steps"]]
        return {"source": "production", "metadata": dict(kernels["metadata"]), "steps": steps}

    def test_rolling_median(self):
        v = np.array([1.0, 9.0, 2.0, np.nan, 3.0, 100.0, 4.0])
        self.assertEqual(rolling_median(v, 3).tolist(), [5.0, 2.0, 5.5, 2.5, 51.5, 4.0, 52.0])
        self.assertEqual(rolling_median(v, 1).tolist()[:3], [1.0, 9.0, 2.0])

    def test_groups(self):
        self.assertEqual(phase_group("aaavbd.sweep"), "Vertex sweeps")
        self.assertEqual(phase_group("contact.constraint_set"), "Contact detection")
        self.assertEqual(phase_group("anderson.subproblem"), "Mixing")  # its history updates
        self.assertEqual(phase_group("step.setup"), OTHER)
        self.assertEqual(phase_group(results.UNSCOPED), OTHER)
        scopes = [s for scopes, _ in PHASE_GROUPS.values() for s in scopes]
        self.assertEqual(len(scopes), len(set(scopes)))  # every scope in one group at most
        table = step_stats.per_step(self._kernels())
        groups = group_phases(table["phases"])
        self.assertEqual(list(groups), ["Vertex sweeps", "Contact detection", "Contact update", "Residuals", OTHER])
        np.testing.assert_allclose(groups["Contact detection"][-1], 2.0)
        small = {"Vertex sweeps": np.array([90.0, 90.0]), "Residuals": np.array([2.0, 2.0]),
                 OTHER: np.array([1.0, 1.0]), "Contact update": np.array([0.5, 0.5])}
        folded = fold_small_groups(small, 0.03)
        self.assertEqual(list(folded), ["Vertex sweeps", OTHER])
        np.testing.assert_allclose(folded[OTHER], [3.5, 3.5])
        self.assertEqual(list(fold_small_groups(small, 0.01)), ["Vertex sweeps", "Residuals", OTHER])

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

    def test_scenes_table(self):
        k = self._kernels()
        newton = self._production(k)
        newton["metadata"]["solver"] = "Newton"
        for i, step in enumerate(newton["steps"]):
            step["status"] = {"newton": [{"substep": 0, "outer": 3, "converged": i % 2 == 0, "newton_steps": 6,
                                          "gradient_ratio": 0.5, "feasibility_ratio": 0.0}]}
        cell = scene_cell(self._production(k))
        # Production times: the union x 1.08, 15 steps at 8.95 ms and 25 at 14.45 ms
        self.assertAlmostEqual(cell["mean"], (15 * 8.95 + 25 * 14.45) / 40 * 1.08)
        self.assertEqual((cell["iterations"], cell["early_exit"]), (100.0, 0.25))
        capsules = self._production(self._kernels("VBD", scene="path/to/capsule_bowl.h5"))
        text = latex_scenes([self._production(k), newton, capsules], {"basket_of_fruit": "Basket of fruit"})
        self.assertIn(r"Basket of fruit & 54.5k & 231k & 10 & -- & 13.4 (15.6) & -- & -- & 13.4 (15.6) \\", text)
        self.assertIn(r" &  &  &  & -- & 100, 25\% & -- & -- & 6, 50\% \\", text)
        self.assertIn(r"capsule\_bowl & 54.5k & 231k & 10 & 13.4 (15.6) & -- & -- & -- & -- \\", text)
        self.assertEqual((_compact(3269), _compact(1809486)), ("3269", "1.81M"))

    def test_sweeps_budget_scenes(self):
        import tempfile

        k = self._kernels()
        per_sweep = ms_per_sweep(k)
        # 15 steps without contact, 25 with; 3200 sweeps (every 4th step exits early at 20)
        self.assertAlmostEqual(per_sweep["Vertex sweeps"], (15 * 8.0 + 25 * 12.0) / 3200)
        self.assertAlmostEqual(per_sweep["Contact detection"], (15 * 0.5 + 25 * 2.0) / 3200)
        vbd = self._production(self._kernels("VBD"))
        rows = solver_rows([k], [self._production(k), vbd])
        self.assertEqual([(a is None, solver_name(a or b)) for a, b in rows], [(True, "VBD"), (False, "AAAVBD")])
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
        vbd = self._production(self._kernels("VBD"))
        self.assertEqual([[solver_name(o) for o in row] for row in other_production(k, [p[0], vbd])],
                         [["VBD"], []])
        with tempfile.TemporaryDirectory() as d:
            image = os.path.join(d, "snap.png")
            plt.imsave(image, np.ones((30, 40, 3)))
            timeline = plot_timeline(k, p, ["A", "B"], bin_n=2, others=[[vbd], []])
            # Contact update and residuals fold into Other; the speed fill falls back to the kernels run's
            self.assertEqual([t.get_text() for t in timeline.legends[0].get_texts()],
                             ["Vertex sweeps", "Contact detection", OTHER, "AAAVBD", "VBD", "AAAVBD peak speed"])
            solvers = plot_solvers([self._kernels("VBD"), k[0]], [p[0]], step_times=True)
            # Contact update, residuals and Other left out; no early-exit panel
            self.assertEqual([t.get_text() for t in solvers.legends[0].get_texts()][:3],
                             ["Vertex sweeps", "Contact detection", "Quartiles"])
            self.assertEqual([ax.get_xlabel() for ax in solvers.axes[1:]],
                             ["Residual ratio per step", "Penetration ratio per step", "Sweeps per step", "Step time (ms)"])
            for name, fig in (("timeline", timeline),
                              ("snapshots", plot_timeline(k[:1], p[:1], ["A"], snapshots=[(0.3, image), (0.2, image)])),
                              ("solvers", solvers)):
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
