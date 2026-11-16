"""
Paper figures and tables of profiling results (``simulate.py --profile kernels --profile-out``), with the
per-step statistics of ``profiling/step_stats.py``.

Phases are grouped for the paper by :data:`PHASE_GROUPS` (scope name -> group, fixed colors); the rest are
"Other". Phase times are unions over streams per step (concurrent streams count once), which exclude
graph launch overhead and the gaps between kernels; production runs (``simulate.py --profile-out`` without a
backend: graph replay for the VBD family) give the true step times. Production runs are paired with kernels
runs by scene and solver. They are separate runs whose trajectories differ slightly (contact is chaotic and
GPU arithmetic order isn't deterministic), so they are compared as distributions, never step by step.

- ``timeline``: one row per kernels result (e.g. AAAVBD on one scene): the GPU time per step of every production
  run over the simulated time, a line in its solver's color (the paired one and the scene's others, e.g. VBD), in a
  plot :data:`TIMELINE_COST_HEIGHT` tall. No phase bands (maintainer, 2026-11-15): the ``solvers`` figure shows the
  phase shares. Behind the curves, on a right axis, the overall motion as an indicator of what happens in the scene
  (maintainer, 2026-11-15; until then behind the residual strip): the free nodes' mass-weighted RMS speed
  sqrt(2 K / M) (``profiling/trajectory.py``'s ``rms_speed``), the paired run's as a light fill (else the kernels
  run's), with ``--rms-speed all`` also the other runs' as dotted lines, each with its own legend entry (default
  ``paired``: the paired run's only). With ``--vbd-ratio`` (off by default: the maintainer found it too busy) and a
  VBD production run, a thin strip under it shows every other production run's step time over VBD's at the same
  step, the paired run's included (:func:`vbd_time_ratio`; VBD is the baseline, so the strip reads as each solver's
  overhead or saving per step; no VBD curve, a dashed line at 1, the same rolling median; log axis when the curves
  span more than :data:`VBD_RATIO_LOG_SPAN`, else linear; steps either run lacks are gaps, steps from either run's
  detected blow-up on are left out); the cost plot then keeps only the paired run's step-time line. Below them, on
  the same time axis, the production runs' smoothed stationarity ratio per step (:func:`ratio_table`; log axis, its
  rolling median over ``--smooth`` steps, a dashed line at 1; a step more than :data:`SPIKE_FACTOR` times its run's
  median there, and at least :data:`SPIKE_MIN_RATIO` (spikes far from convergence only), gets a small marker in the run's color at its value, at the top edge with the value written beside
  it where it's above the range, which stays the median curves'; maintainer, 2026-11-16; a run that blew up stops
  where it began to diverge, :func:`divergence_time` as in the ``solvers`` figure, with a cross in its color and
  "Anderson diverged (1.08 s)" beside it, so the other runs set the range); runs without solve statuses have no
  such strip. Last,
  when any run recorded a penetration (a scene with contact), a thin strip of each production run's feasibility
  (penetration) ratio per step (:data:`FEASIBILITY_STRIP`, the worst over the step's substeps; the same rolling
  median, solver colors, log axis and dashed line at 1, values below :data:`RATIO_FLOOR` drawn at it, cut where a
  run that blew up began to diverge, with a cross there). It has its own strip rather than sharing the residual's: penetration ratios span other decades
  (0 without contact, up to ~100 at impacts), so a shared log axis would flatten the residual curves, and eight
  curves in one strip don't read. Each strip adds its own height to the row. ``--snapshots TIME=IMAGE ...`` adds a
  row of simulation snapshots above (``profiling/snapshots.py`` renders them; any size and aspect, all at one height,
  each as wide as its aspect needs), numbered, with the numbers marked at their times on the plots (first row only). Steps a result lacks, the warm-up steps at the seams of a run merged
  from legs (``profiling/legs.py``), are gaps in every curve (with ``--bin``, the bins that hold them), not
  interpolated.
- ``solvers``: one row per solver of one scene, in the benchmark figures' solver order. First column, from
  short fixed-budget kernels runs (no early exit, e.g. every solver resumed from the same checkpoint): each
  phase group's share of the GPU time, with the GPU time per sweep at the bar's end. Groups shared by every
  solver and negligible (:data:`SOLVERS_OMITTED`) are left out without renormalizing, so the bars end short of
  100%. The other columns, from full production runs (``--production``), show what each solver's budget buys:
  the distribution per step of the smoothed stationarity ratio (:func:`ratio_table`) and of the penetration
  (feasibility) ratio to its tolerance (box: quartiles, whiskers: P10-P90, dot: maximum; log axes, ratios below
  1e-3 drawn at 1e-3, a dashed line at 1, where each test passes; a wide log axis labels only every k-th decade,
  the smallest k at which its labels don't overlap, :func:`_fit_decade_labels`), the sweeps per step (linear axis)
  and, with ``--step-times``, the step time. A run that blew up has its two ratio boxes cut where it began to
  diverge (:func:`divergence_time`, :func:`diverged_table`), marked by a cross at the last value drawn and named in
  the legend ("Anderson: diverged at 1.08 s"), so the other runs set those axes; its sweeps and step times keep
  every step. One-solve spikes aren't cut: only a detected blow-up (:func:`blowup_time`) is.
  Sweep counts are quantized by the convergence test, which runs every ``Solver.Convergence.check_interval``
  sweeps and after each subproblem (20 sweeps on CapsulesIntoBowl, so every count is a multiple of 20). Early
  exits aren't shown: their fraction doesn't say how soon a solver exits.
- The residual ratio of the ``timeline`` strips and the ``solvers`` and ``budget`` boxes is each step's smoothed
  stationarity ratio, the smoothed test's own (``status_ratio.smoothed_test_ratio``, the worst over the step's
  substeps; the ratio the maintainer trusts, as in the ``scenes`` table): the decisive ratio follows the plain test,
  whose float32 noise misranks solvers (FourBunnies.Trampoline250kgPerM3.60Hz, medians AAAVBD 4.18 / Anderson 3.24
  decisive, 0.24 / 2.23 smoothed). Runs without smoothed ratios (recorded before them, or without smoothing) fall
  back to the decisive ratio (``status_ratio.stationarity_test_ratio``), marked "(decisive)" in the legend or row
  label. Steps from a detected blow-up on (:func:`blowup_time`) are left out.
- ``slices``: Markdown, per kernels result: the steps grouped by percentiles of their GPU time
  (``--slices``, default the fastest 10%, P10-P90 and the slowest 10%) with each group's median step time,
  median iterations and early exits, for the paper's text.
- ``sweeps``: the ``solvers`` figure's first column for several scenes, one panel per scene.
- ``budget``: the ``solvers`` figure's other columns for several scenes, one block of rows per scene.
- ``scenes``: the paper's LaTeX ``table*`` of production runs (graph replay for the VBD family, eager for
  Newton): four or five lines per scene (its nodes, tetrahedra and time step h), a column per solver (VBD, AAAVBD,
  Chebyshev, Anderson, Newton; -- without a run): the mean step time in ms with the P90 in parentheses, then
  the median iterations per step (VBD sweeps, Newton steps) and the fraction of early exits, then the residual: the
  median smoothed stationarity ratio over the run's solves with its minimum and maximum in brackets
  (``status_ratio.smoothed_test_ratio``, the smoothed test's own ratio, the reliable one by the maintainer; no plain
  or decisive ratio; -- for runs recorded without smoothing), then contact: the
  largest penetration (feasibility) ratio over the run's solves (substeps), from the same solve statuses, with the
  number of solves above 1 (contact constraints unmet at the end of the solve) in parentheses. A run that blew up
  counts only its steps before the blow-up in these two lines (:func:`before_blowup`, at the outcome mark's
  :func:`blowup_time`); a run stopped early without blowing up counts all its steps; non-finite ratios are left out
  either way. A fifth line, written for the scenes where any
  cell has one, gives each run's outcome: a short note from ``--outcomes`` (TOML: a table per scene key, optionally
  qualified by the frame rate, e.g. ``["capsule_bowl.30Hz"]``, with a note per solver; judgments made by watching
  the runs, not computed), else an automatic mark, ``blew up at t = X s`` where its positions went non-finite or
  very far (a non-finite residual or speed, or a node faster than :data:`BLOWUP_SPEED`), else ``stopped at t = X s``
  for a run whose output ends before the scene's longest run. Scene labels wrap over the scene's lines
  (``multirow``) and outcome notes within their cell, so the table fits a two-column paper's ``table*``.
- ``scene_setup`` and ``simulation``: the paper's two LaTeX ``table*`` of the scenes' setup (maintainer, 2026-11-16:
  one table was too wide), a line per scene, from production runs at equal sweeps (Newton's included) plus
  ``--equal-time`` runs, every value from the runs' own metadata (:func:`setup_rows`, which refuses a scene whose
  runs disagree). ``scene_setup``: the mesh's nodes and tetrahedra (as the ``scenes`` table), Young's modulus,
  Poisson's ratio and density, ranges over the tetrahedra of the scene file the runs name (:func:`scene_material`),
  E and nu checked against each run's simulated Lame parameters (:func:`run_material`, :func:`check_material`). ``simulation``: the frame step h (each solve's is h / substeps), the budgets as
  substeps x outer iterations x iterations per subproblem (VBD sweeps for the VBD family at equal sweeps, the
  equal-time runs' sweeps per subproblem, Newton steps with Newton's CG cap and relative tolerance from its output,
  :func:`newton_cg_params`; ``--newton-config`` reads a scene config's for a Newton run that hasn't finished,
  marked) and the convergence tolerances, each group's quantities as sub-columns under one spanning header
  (:data:`SIMULATION_GROUPS`), every quantity a math-mode variable through a macro declared once at the top of the
  .tex (:data:`SIMULATION_SYMBOLS`, ``\providecommand``: rename one in a line); wrapped scene labels and a 3 pt
  column separation keep it narrow. ``--notes`` (:func:`load_setup_notes`) adds footnotes per cell,
  explained in the caption; BDF2 scenes are marked. Readable copies go next to each .tex, for reading without
  LaTeX: ``.md`` and a ``.pdf`` drawn by Matplotlib (:func:`markdown_simulation`, :func:`plot_simulation`, and the
  scene setup's counterparts), with the same cells, caption and footnotes.
- ``lead``: AAAVBD's lead over the other VBD-family solvers against the time step, one panel per scene: each
  solver's median smoothed stationarity ratio over its solves (``status_ratio.smoothed_test_ratio`` only, as the
  other figures; runs without smoothed ratios give no point), one line per solver, divided by AAAVBD's at the same
  scene and solver step h = dt / substeps (log x, ticks at the steps run), over the same window (log y, a dashed
  line at 1, above it AAAVBD is more converged). The positional runs are compared over their scene's common span (circles: from the latest first
  recorded step, production runs skip their warm-up, to the end of the shortest campaign); ``--windowed`` runs over
  their scene's ``--window`` (squares; e.g. the real-time screen's phase-2 windows). Either is narrowed to what both
  runs recorded and to before either blew up (crossed: compared up to it). The points are also printed as Markdown.

Usage (from the repo root):

    python -m python.tools.sim.profiling.paper_figures timeline k_capsules.json --production p_capsules.json \
        p_capsules_vbd.json -o data/figures/profiling/timeline [--labels capsule_bowl="Capsules into bowl"] \
        [--width 7] [--bin 5] [--smooth 9] [--rms-speed paired|all] [--vbd-ratio] \
        [--snapshots 0.9=snap_1.png 1.4=snap_2.png ...]
    python -m python.tools.sim.profiling.paper_figures solvers k_fixed_*.json --production p_*.json \
        -o data/figures/profiling/solvers [--step-times]
    python -m python.tools.sim.profiling.paper_figures slices k_*.json -o data/tables/slices.md [--slices 10,90]
    python -m python.tools.sim.profiling.paper_figures sweeps k_fixed_*.json -o data/figures/profiling/sweeps
    python -m python.tools.sim.profiling.paper_figures budget p_*.json -o data/figures/profiling/budget [--step-times]
    python -m python.tools.sim.profiling.paper_figures scenes p_*.json -o data/tables/scenes.tex [--labels ...] \
        [--outcomes outcomes.toml]
    python -m python.tools.sim.profiling.paper_figures lead p_10ms_*.json p_60Hz_*.json -o data/figures/lead \
        [--windowed w_*.json --window armadillo-pull=0:3 ...] [--labels ...] [--width 7]

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
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from . import results, step_stats  # noqa: E402
from .. import status_ratio  # noqa: E402
from ..figure_style import SOLVER_COLORS, SOLVER_NAMES, use_paper_fonts  # noqa: E402

# The paper's phase groups, in drawing order (bottom of the stack first): name -> (scope names, color). Scope
# names are innermost scopes (results.leaf); a scope in no group is drawn as OTHER. AAAVBD's acceleration runs
# inside its vertex-solve kernel (aaavbd.sweep), so it has no group: its cost shows in its time per sweep
# against VBD's. Anderson's history updates run directly in its subproblem scope (anderson.subproblem's self
# time), so they count as Mixing. Colors: muted olive, khaki, slate and brown tones, none sharing a hue with a solver's color
# (figure_style.SOLVER_COLORS), so phase fills never read as the solver lines drawn over them. Sweeps, detection
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
SNAPSHOT_GAP = 0.03  # timeline: the gap between snapshots, a fraction of their mean width
# timeline: a step whose stationarity ratio is more than this factor times its run's rolling median at that step,
# and at least SPIKE_MIN_RATIO, gets a marker on the residual strip (a one-solve spike the median hides:
# ArmadilloPull.60Hz AAAVBD's 2.6e4 at its release, step 61)
SPIKE_FACTOR = 10.0
# ... and only spikes far from convergence (maintainer, 2026-11-16: "only flag steps way above the ratio of 1"):
# without it, Trampoline.StiffSphere.60Hz AAAVBD got six markers at ~0.2, below the tolerance line
SPIKE_MIN_RATIO = 10.0
# The residual ratio the timeline and solvers figures plot: the stationarity test's decisive ratio, at most 1 iff
# the test passed (status_ratio.stationarity_test_ratio, step_stats' test_ratio; the gradient ratio for runs
# recorded before its other ratios existed, and for GPU Newton). The gradient ratio RMS_M(q) alone can be below 1
# on a step that failed the test's floor clause (CapsulesIntoBowl VBD, 58fd2f99: median 0.23, while 389 of 591
# steps failed).
TEST_RATIO = "test_ratio"
# The ratio the timeline strips and the solvers and budget boxes plot (ratio_table): the smoothed test's own, else
# TEST_RATIO for runs without smoothed ratios
STRIP_RATIO = "strip_ratio"
STRIP_RATIO_LABEL = "Stationarity ratio"
FEASIBILITY_STRIP = "feasibility_strip"  # ratio_table: the timeline's per-step feasibility ratio, cut at a blow-up
DECISIVE_MARK = " (decisive)"  # legend or row label suffix of runs plotted with the decisive ratio instead


def ratio_table(result: dict) -> dict:
    """:func:`step_stats.per_step` with :data:`STRIP_RATIO`, each step's smoothed stationarity ratio (the worst over
    its substeps, ``status_ratio.smoothed_test_ratio``), else, for runs without any, :data:`TEST_RATIO`, and
    :data:`FEASIBILITY_STRIP`, its feasibility (penetration) ratio (the worst over its substeps); both NaN from a
    detected blow-up on (:func:`blowup_time`). ``smoothed`` says which; ``decisive_fallback`` is true for a run
    plotted with the decisive ratio (it has some, and no smoothed ones)."""
    table = step_stats.per_step(result)
    smoothed = []
    for step in result["steps"]:
        status = step.get("status") or {}
        values = np.array([status_ratio.smoothed_test_ratio(r) for r in status.get("vbd", status.get("newton", []))],
                          dtype=float)
        values = values[np.isfinite(values)]
        smoothed.append(values.max() if values.size else np.nan)
    smoothed = np.array(smoothed, dtype=float)
    has_smoothed = bool(np.isfinite(smoothed).any())
    ratio = smoothed if has_smoothed else np.array(table[TEST_RATIO], dtype=float)
    feasibility = np.array(table["feasibility_ratio"], dtype=float)
    blowup = blowup_time(result)
    if blowup is not None:
        dt = float(result["metadata"].get("dt", 1.0))
        after = (table["t"] + 1) * dt >= blowup - 0.5 * dt
        ratio = np.where(after, np.nan, ratio)
        feasibility = np.where(after, np.nan, feasibility)
    return {**table, STRIP_RATIO: ratio, FEASIBILITY_STRIP: feasibility, "smoothed": has_smoothed,
            "decisive_fallback": not has_smoothed and bool(np.isfinite(ratio).any())}


def _ratio_label(name: str, table: dict) -> str:
    return name + (DECISIVE_MARK if table["decisive_fallback"] else "")
# timeline: the height (in) of a row's cost plot, the production runs' step times (maintainer, 2026-11-15: no phase
# bands, the solvers figure shows the phase shares, so the curves need less height)
TIMELINE_COST_HEIGHT = 0.8
SPEED = "rms_speed"  # timeline: the per-step speed behind the residual ratios (profiling/trajectory.py)
SPEED_NAME = "RMS speed"
SPEED_FILL = "#d9d9d9"  # timeline: the paired run's speed behind the residual ratios
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


SPEED_DOTS = (0, (1, 1.5))  # timeline: the other production runs' speeds (``--rms-speed all``)
RMS_SPEED_MODES = ("paired", "all")  # timeline: whose RMS speed the strip draws (the paired run's, or every run's)


def _speed_axis(ax, x: np.ndarray, kernels: dict, kernels_speed: np.ndarray, runs: list[dict],
                rms_speed: str = "paired") -> list:
    """The :data:`SPEED` behind the timeline's cost plot ``ax``, on a right axis, an indicator of the scene's motion
    over time (maintainer, 2026-11-15; see the module docstring): the paired run's (``runs[0]``) as a light fill, and
    with ``rms_speed`` ``"all"`` the other runs' as dotted lines; ``x`` and ``kernels_speed`` are the ``kernels`` run's
    step times (s) and speeds, drawn if the paired run has none. Returns the speeds' legend handles."""
    speed = ax.twinx()
    handles = []
    for i, run in enumerate(runs):
        table = step_stats.on_step_grid(step_stats.per_step(run))
        if not np.isfinite(table[SPEED]).any():
            continue
        xr = step_seconds(run, table["t"])
        if i == 0:
            speed.fill_between(xr, table[SPEED], color=SPEED_FILL, alpha=0.6, linewidth=0)
            handles.append(Patch(color=SPEED_FILL, alpha=0.6, label=f"{solver_name(run)} {SPEED_NAME}"))
        elif rms_speed == "all":
            speed.plot(xr, table[SPEED], color=solver_color(run), linewidth=0.5, linestyle=SPEED_DOTS)
            handles.append(Line2D([], [], color=solver_color(run), linewidth=0.5, linestyle=SPEED_DOTS,
                                  label=f"{solver_name(run)} {SPEED_NAME}"))
    if not any(isinstance(h, Patch) for h in handles) and np.isfinite(kernels_speed).any():
        speed.fill_between(x, kernels_speed, color=SPEED_FILL, alpha=0.6, linewidth=0)
        handles.insert(0, Patch(color=SPEED_FILL, alpha=0.6, label=f"{solver_name(kernels)} {SPEED_NAME}"))
    speed.set_ylim(bottom=0)
    speed.set_ylabel(f"{SPEED_NAME}\n(m/s)", fontsize=FONT_SIZE, color="#7f7f7f")
    # The step times over the speed fill: the left axes in front, without their background
    ax.set_zorder(speed.get_zorder() + 1)
    ax.patch.set_visible(False)
    for side in ("top", "left", "bottom"):
        speed.spines[side].set_visible(False)
    speed.spines["right"].set_linewidth(0.5)
    speed.tick_params(labelsize=FONT_SIZE, length=2, width=0.5, pad=1.5, colors="#7f7f7f")
    return handles


def _residual_strip(strip, runs: list[dict], smooth: int):
    """The timeline's residual strip: each production run's smoothed stationarity ratio per step
    (:data:`STRIP_RATIO`, its rolling median of ``smooth`` steps, gaps kept) on a log axis, with a dashed line at 1,
    and markers on the steps above :data:`SPIKE_FACTOR` times that median and at least :data:`SPIKE_MIN_RATIO`
    (:func:`_spike_markers`)."""
    # TODO: show every convergence metric over time, the feasibility (penetration) ratio beside the stationarity
    #  ratio (and the unsmoothed beside the smoothed one, both recorded since fd5eaa56). A per-step residual can't
    #  carry a trajectory's past error: on BasketOfFruit, Chebyshev's smoothed ratio is 2x AAAVBD's while its net
    #  overstretches (0.5-1.25 s, 133 infeasible steps, a fruit falls through) but lower after its net recovers
    #  (1.5-2 s, 19 vs 28), so the stationarity ratio alone misreads a late window (maintainer, 2026-11-09;
    #  data/profiling/investigations/BasketOfFruit-residual-misranking).
    medians = [np.array([1.0])]  # the y range covers the tolerance line
    spikes = []  # (run, times, ratios) of the steps above SPIKE_FACTOR times their rolling median
    cuts = []  # (run, time, value) where a run's curve stops at its divergence
    for run in runs:
        table = step_stats.on_step_grid(diverged_table(run))
        ratio = np.maximum(table[STRIP_RATIO], RATIO_FLOOR)
        medians.append(_cut_curve(run, table, np.where(table["measured"], rolling_median(ratio, smooth), np.nan),
                                  cuts))  # gaps stay gaps
        x = step_seconds(run, table["t"])
        strip.plot(x, medians[-1], color=solver_color(run), linewidth=0.8)
        with np.errstate(invalid="ignore"):
            spike = (np.isfinite(ratio) & np.isfinite(medians[-1]) & (ratio > SPIKE_FACTOR * medians[-1])
                     & (ratio >= SPIKE_MIN_RATIO))
        if spike.any():
            spikes.append((run, x[spike], ratio[spike]))
    strip.axhline(1.0, color=INK, linewidth=0.5, linestyle=(0, (3, 2)))
    strip.set_yscale("log")
    values = np.concatenate(medians)
    values = values[np.isfinite(values) & (values > 0)]
    strip.set_ylim(values.min() / 1.25, values.max() * 1.25)  # the curves with a margin, not clipped at the edges
    _spike_markers(strip, spikes)
    _cut_markers(strip, cuts, label=True)
    _log_axis(strip, "y")
    strip.set_ylabel("Stationarity\nratio", fontsize=FONT_SIZE)
    _style(strip)


def _spike_label(value: float) -> str:
    """A clipped spike's value: three significant digits, a compact exponent beyond 1000 ("2.6e4")."""
    if value < 1e3:
        return f"{value:.3g}"
    e = int(np.floor(np.log10(value)))
    return f"{value / 10.0**e:.2g}e{e}"


def _spike_markers(strip, spikes: list[tuple[dict, np.ndarray, np.ndarray]]):
    """The residual strip's spikes (:data:`SPIKE_FACTOR`): a small marker in the run's color at each step's ratio,
    within the y range the median curves set; above it, at the top edge, labeled with the value. The range stays the
    curves'."""
    lo, hi = strip.get_ylim()
    for run, x, ratio in spikes:
        inside = ratio <= hi
        strip.plot(x[inside], ratio[inside], marker="o", markersize=2.0, color=solver_color(run), linestyle="none")
        strip.plot(x[~inside], np.full((~inside).sum(), hi), marker="o", markersize=2.0, color=solver_color(run),
                   linestyle="none", clip_on=False)
        for xi, value in zip(x[~inside], ratio[~inside]):
            strip.annotate(_spike_label(value), (xi, hi), xytext=(2.0, -0.5), textcoords="offset points", ha="left",
                           va="top", fontsize=FONT_SIZE - 1, color=solver_color(run))
    strip.set_ylim(lo, hi)


def _cut_curve(run: dict, table: dict, curve: np.ndarray, cuts: list) -> np.ndarray:
    """A timeline strip's rolling-median ``curve`` of a run cut where it began to diverge (:func:`diverged_table`;
    the median of the steps before would otherwise run half a window past it), with (run, time, last value) appended
    to ``cuts``; unchanged for a run that didn't blow up."""
    start = table.get("diverged")
    if start is None:
        return curve
    x = step_seconds(run, table["t"])
    curve = np.where(x >= start - 0.5 * float(run["metadata"].get("dt", 1.0)), np.nan, curve)
    drawn = np.flatnonzero(np.isfinite(curve))
    if drawn.size:
        cuts.append((run, start, float(curve[drawn[-1]])))
    return curve


def _cut_markers(ax, cuts: list, label: bool):
    """A cross in the run's color where its curve stops at its divergence (:func:`_cut_curve`), with "Anderson
    diverged (1.08 s)" beside it if ``label``; the y range is left as the drawn curves set it."""
    lo, hi = ax.get_ylim()
    for run, start, value in cuts:
        value = min(max(value, lo), hi)
        ax.plot(start, value, **{**CUT_MARKER, "color": solver_color(run), "markersize": 4.0})
        if label:
            ax.annotate(f"{solver_name(run)} diverged ({start:.2f} s)", (start, value), xytext=(3.0, 0.0),
                        textcoords="offset points", ha="left", va="center", fontsize=FONT_SIZE - 1,
                        color=solver_color(run))
    ax.set_ylim(lo, hi)


BASELINE = "VBD"  # timeline: the production run every other run's step time is divided by
# timeline: the step-time ratio strip gets a log axis when its curves span more than this factor (max / min, with 1),
# a linear one otherwise (where a log axis would show a single tick or two)
VBD_RATIO_LOG_SPAN = 3.0


def vbd_time_ratio(run: dict, baseline: dict) -> tuple[np.ndarray, np.ndarray]:
    """``run``'s production step time over ``baseline``'s at the same step (simulated time, s; ratio), on ``run``'s
    step grid (:func:`step_stats.on_step_grid`): NaN where either run lacks the step (warm-up gaps, legs) and from
    either run's detected blow-up on (:func:`blowup_time`)."""
    table = step_stats.on_step_grid(step_stats.per_step(run))
    ref = step_stats.per_step(baseline)
    ref_ms = dict(zip(ref["t"].tolist(), ref["time_ms"].tolist()))
    t = table["t"]
    base = np.array([ref_ms.get(int(step), np.nan) for step in t], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(table["measured"] & (base > 0), table["time_ms"] / base, np.nan)
    dt = float(run["metadata"].get("dt", 1.0))
    blowups = [b for b in map(blowup_time, (run, baseline)) if b is not None]
    if blowups:
        ratio = np.where((t + 1) * dt >= min(blowups) - 0.5 * dt, np.nan, ratio)
    return step_seconds(run, t), ratio


def _vbd_ratio_strip(ax, runs: list[dict], baseline: dict, smooth: int):
    """The timeline's step-time strip: each run's :func:`vbd_time_ratio` against ``baseline`` (rolling median of
    ``smooth`` steps, gaps kept), in its solver's color, with a dashed line at 1."""
    curves = [np.array([1.0])]
    for run in runs:
        x, ratio = vbd_time_ratio(run, baseline)
        if not np.isfinite(ratio).any():
            continue
        curves.append(np.where(np.isfinite(ratio), rolling_median(ratio, smooth), np.nan))
        ax.plot(x, curves[-1], color=solver_color(run), linewidth=0.8)
    ax.axhline(1.0, color=INK, linewidth=0.5, linestyle=(0, (3, 2)))
    values = np.concatenate(curves)
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = values.min(), values.max()
    if hi / lo > VBD_RATIO_LOG_SPAN:
        ax.set_ylim(lo / 1.25, hi * 1.25)
        _log_axis(ax, "y")
    else:
        pad = 0.1 * max(hi - lo, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    ax.set_ylabel(f"Step time /\n{SOLVER_NAMES.get(BASELINE, BASELINE)}'s", fontsize=FONT_SIZE)
    _style(ax)


def _feasibility_strip(ax, runs: list[dict], smooth: int):
    """The timeline's feasibility strip: each run's per-step feasibility (penetration) ratio to its tolerance
    (:data:`FEASIBILITY_STRIP`, the worst over the step's substeps; its rolling median of ``smooth`` steps, gaps kept;
    values below :data:`RATIO_FLOOR`, e.g. no penetration, drawn at it) on a log axis, with a dashed line at 1."""
    curves = [np.array([1.0])]
    cuts = []
    for run in runs:
        table = step_stats.on_step_grid(diverged_table(run))
        ratio = np.maximum(table[FEASIBILITY_STRIP], RATIO_FLOOR)
        curves.append(_cut_curve(run, table, np.where(table["measured"] & np.isfinite(ratio),
                                                      rolling_median(ratio, smooth), np.nan), cuts))
        ax.plot(step_seconds(run, table["t"]), curves[-1], color=solver_color(run), linewidth=0.8)
    ax.axhline(1.0, color=INK, linewidth=0.5, linestyle=(0, (3, 2)))
    values = np.concatenate(curves)
    values = values[np.isfinite(values) & (values > 0)]
    ax.set_ylim(values.min() / 1.25, values.max() * 1.25)
    _cut_markers(ax, cuts, label=False)
    _log_axis(ax, "y")
    ax.set_ylabel("Penetration\nratio", fontsize=FONT_SIZE)
    _style(ax)


def has_contact(run: dict) -> bool:
    """Whether a run recorded any penetration ratio above :data:`RATIO_FLOOR` (scenes without contact record 0)."""
    feasibility = ratio_table(run)[FEASIBILITY_STRIP]
    return bool((feasibility[np.isfinite(feasibility)] > RATIO_FLOOR).any())


def plot_timeline(kernels: list[dict], production: list[dict | None], labels: list[str], width: float = 7.0,
                  bin_n: int = 1, snapshots: list[tuple[float, str]] | None = None,
                  others: list[list[dict]] | None = None, smooth: int = SMOOTH_STEPS, rms_speed: str = "paired",
                  vbd_ratio: bool = False):
    """The ``timeline`` figure (see the module docstring): one row per kernels result, the first one under its
    ``snapshots`` (simulated time, image path), numbered at their times on the plots. ``others`` are each row's
    production runs of other solvers (:func:`other_production`), drawn over the paired ``production`` run;
    ``rms_speed`` (:data:`RMS_SPEED_MODES`) says whose RMS speed the cost plot draws; ``vbd_ratio`` adds the
    step-time-over-VBD strip (``--vbd-ratio``; off by default, the maintainer found it too busy), which then replaces
    the other runs' step-time lines on the cost plot."""
    tables = [step_stats.on_step_grid(step_stats.per_step(k)) for k in kernels]
    rows = len(kernels)
    others = others or [[] for _ in kernels]
    runs_per_row = [([prod] if prod is not None else []) + list(more) for prod, more in zip(production, others)]
    has_ratio = [any(np.isfinite(ratio_table(run)[STRIP_RATIO]).any() for run in runs) for runs in runs_per_row]
    has_speed = [bool(np.isfinite(t[SPEED]).any()) or any(np.isfinite(step_stats.per_step(run)[SPEED]).any()
                                                         for run in runs)
                 for t, runs in zip(tables, runs_per_row)]
    baselines = [next((run for run in runs if run["metadata"].get("solver") == BASELINE), None) if vbd_ratio
                 else None for runs in runs_per_row]
    has_feasibility = [any(has_contact(run) for run in runs) for runs in runs_per_row]
    snapshots = sorted(snapshots or [])
    images = [plt.imread(path) for _, path in snapshots]
    # Any size and aspect: one row height, each image as wide as its aspect needs (the gaps between them included)
    aspects = [image.shape[1] / image.shape[0] for image in images]
    image_height = width / (sum(aspects) + SNAPSHOT_GAP * (len(images) - 1) * np.mean(aspects)) if images else 0.0
    heights = []
    for ratio, baseline, feasibility in zip(has_ratio, baselines, has_feasibility):
        strips = [0.45 if baseline is not None else 1e-4, 0.65 if ratio else 1e-4,
                  0.45 if feasibility else 1e-4]  # 1e-4: an absent strip
        heights += [TIMELINE_COST_HEIGHT] + strips
    fig = plt.figure(figsize=(width, image_height + sum(heights) + 0.45 * rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[max(image_height, 1e-3), sum(heights) + 0.45 * rows], hspace=0.12)
    if images:
        strip_of_images = outer[0].subgridspec(1, len(images), wspace=SNAPSHOT_GAP, width_ratios=aspects)
        for i, image in enumerate(images):
            ax = fig.add_subplot(strip_of_images[i])
            ax.imshow(image, interpolation="none")  # embedded in the PDF at its own resolution, not resampled
            ax.set_axis_off()
            ax.text(0.04, 0.96, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    fontweight="bold", color=INK)
    grid = outer[1].subgridspec(4 * rows, 1, height_ratios=heights, hspace=0.15)
    solvers: dict[str, str] = {}  # drawn production runs: solver name -> color
    speeds: dict[str, object] = {}  # speeds' legend handles by label
    for r, (result, table, runs, label) in enumerate(zip(kernels, tables, runs_per_row, labels)):
        x = bin_steps(step_seconds(result, table["t"]), bin_n)
        area = fig.add_subplot(grid[4 * r])
        for run in runs:
            solvers.setdefault(_ratio_label(solver_name(run), ratio_table(run)), solver_color(run))
            # With a baseline, the other runs' step times show only as ratios to it, in the strip below
            if baselines[r] is not None and run is not production[r]:
                continue
            ptable = step_stats.on_step_grid(step_stats.per_step(run))
            area.plot(bin_steps(step_seconds(run, ptable["t"]), bin_n), bin_steps(ptable["time_ms"], bin_n),
                      color=solver_color(run), linewidth=0.6, path_effects=HALO)
        area.set_xlim(x[0], x[-1])
        area.set_ylim(bottom=0)
        area.set_ylabel("GPU time (ms)", fontsize=FONT_SIZE)
        if rows > 1 or not images:
            area.text(0, 1.04, label, transform=area.transAxes, ha="left", va="bottom", fontsize=8,
                      fontweight="bold", color=INK)
        _style(area)
        axes = [area]
        if baselines[r] is not None:
            area.tick_params(labelbottom=False)
            relative = fig.add_subplot(grid[4 * r + 1], sharex=area)
            _vbd_ratio_strip(relative, [run for run in runs if run is not baselines[r]], baselines[r], smooth)
            axes.append(relative)
        if has_ratio[r]:
            axes[-1].tick_params(labelbottom=False)
            strip = fig.add_subplot(grid[4 * r + 2], sharex=area)
            _residual_strip(strip, runs, smooth)
            axes.append(strip)
        if has_feasibility[r]:
            axes[-1].tick_params(labelbottom=False)
            penetration = fig.add_subplot(grid[4 * r + 3], sharex=area)
            _feasibility_strip(penetration, runs, smooth)
            axes.append(penetration)
        if r == rows - 1:
            axes[-1].set_xlabel("Time (s)" if "dt" in result["metadata"] else "Time step", fontsize=FONT_SIZE)
        if r == 0:
            for i, (t, _) in enumerate(snapshots):
                for ax in axes:
                    ax.axvline(t, color=INK, linewidth=0.5, linestyle=(0, (2, 2)))
                _marker(area, t, i + 1)
        if has_speed[r]:  # last, so the row's other axes keep their order in fig.axes
            for h in _speed_axis(area, step_seconds(result, table["t"]), result, table[SPEED], runs, rms_speed):
                speeds.setdefault(h.get_label(), h)
    handles = [Line2D([], [], color=color, linewidth=0.8, label=name) for name, color in solvers.items()]
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
            _box(ax, yi, table[STRIP_RATIO], RATIO_FLOOR)
            _cut_marker(ax, yi, table, STRIP_RATIO)
        elif column == "feasibility":
            _box(ax, yi, table["feasibility_ratio"], RATIO_FLOOR)
            _cut_marker(ax, yi, table, "feasibility_ratio")
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
        # The tolerance line first, so the ticks are chosen for the final limits
        ax.axvline(1.0, color=INK, linewidth=0.5, linestyle=(0, (2, 2)))
        _log_axis(ax, dense=NARROW_LOG_RANGE, fit_labels=True)
        ax.set_xlabel(STRIP_RATIO_LABEL if column == "gradient" else "Penetration ratio",
                      fontsize=FONT_SIZE)
    elif column == "early":
        ax.set_xlim(0, 100)
        ax.set_xlabel("Early exits (%)", fontsize=FONT_SIZE)
    elif column == "sweeps":
        ax.set_xlim(left=0)
        # A few round ticks at any budget (every 50 sweeps crowded a 750-sweep axis)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
        ax.set_xlabel("Sweeps", fontsize=FONT_SIZE)
    else:
        _log_axis(ax, dense=NARROW_LOG_RANGE, fit_labels=True)
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
    tables = [diverged_table(p) if p is not None else None for _, p in rows]
    names = [solver_name(k or p) if t is None else _ratio_label(solver_name(k or p), t)
             for (k, p), t in zip(rows, tables)]
    present = _sweeps_panel(axes[0], [k for k, _ in rows], y, SOLVERS_OMITTED)
    _row_axes(axes[0], y, names)
    for ax, column in zip(axes[1:], columns):
        _budget_panel(ax, column, tables, y)
        _row_axes(ax, y, None)
    _legend(fig, [Patch(color=group_color(g), label=g) for g in present] + _box_handles()
            + _cut_handles([(p, t) for (_, p), t in zip(rows, tables)]))
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


# solvers and budget figures: the narrow panels' log axes get 1-2-5 ticks only within this ratio of their limits
# (about 1.5 decades), decades over a wider range, so their labels don't overlap
NARROW_LOG_RANGE = 30.0


def _log_axis(ax, which: str = "x", dense: float = 300.0, fit_labels: bool = False):
    """Log x (or y) axis with plain-number ticks: 1-2-5 within a range under ``dense`` (hi / lo), decades over a
    wider one, and a few round linear ticks where fewer than two of those fall in range (call after drawing
    everything, so the limits are final). With ``fit_labels``, a decade axis labels only as many decades as fit
    (:func:`_fit_decade_labels`)."""
    getattr(ax, f"set_{which}scale")("log")
    lo, hi = getattr(ax, f"get_{which}lim")()
    axis = getattr(ax, f"{which}axis")
    subs = (1.0, 2.0, 5.0) if hi / lo < dense else (1.0,)
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    if sum(lo <= m * 10.0**e <= hi for e in decades for m in subs) >= 2:
        # numticks: LogLocator's default can thin a multi-decade 1-2-5 axis to no ticks at all
        axis.set_major_locator(LogLocator(subs=subs, numticks=100))
    else:  # LogLocator would fall back to dense linear ticks (4000, 4250, ... on a 4-5.6 s step time axis)
        axis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())
    if fit_labels and subs == (1.0,):
        _fit_decade_labels(ax, which, [e for e in decades if lo <= 10.0**e <= hi])


def _decade_label(v: float, _=None) -> str:
    """A decade tick's label: plain from 0.001 to 1000, else a power of ten (10^4, not 10000 or 1e+04)."""
    e = int(round(np.log10(v)))
    return f"{v:g}" if -3 <= e <= 3 else f"$10^{{{e}}}$"


# Minimum gap (points) between two tick labels of a decade axis (:func:`_fit_decade_labels`)
TICK_LABEL_GAP = 1.5


def _fit_decade_labels(ax, which: str, exponents: list[int]):
    """Labels every k-th decade of a log axis (the exponents divisible by k, so 1 keeps its label where in range),
    for the smallest k at which no two labels overlap or come closer than :data:`TICK_LABEL_GAP` at the figure's
    size; the other decades keep unlabeled ticks. Wide stationarity spans crowded the narrow solvers panels
    (BirthdayBanner.60Hz: 1 to 10^6 in under an inch, 2026-11-16)."""
    axis = getattr(ax, f"{which}axis")
    fig = ax.figure
    gap = TICK_LABEL_GAP * fig.dpi / 72.0
    k = 1
    while True:
        major = [10.0**e for e in exponents if e % k == 0]
        axis.set_major_locator(FixedLocator(major))
        axis.set_minor_locator(FixedLocator([10.0**e for e in exponents if e % k != 0]))
        axis.set_major_formatter(FuncFormatter(_decade_label))
        axis.set_minor_formatter(NullFormatter())
        fig.canvas.draw()
        boxes = sorted((t.get_window_extent() for t in axis.get_majorticklabels() if t.get_visible() and t.get_text()),
                       key=lambda b: b.x0 if which == "x" else b.y0)
        if which == "x":
            crowded = any(b.x0 < a.x1 + gap for a, b in zip(boxes, boxes[1:]))
        else:
            crowded = any(b.y0 < a.y1 + gap for a, b in zip(boxes, boxes[1:]))
        # Stop before a k that would leave fewer than two labels
        if not crowded or sum(e % (k + 1) == 0 for e in exponents) < 2:
            return
        k += 1


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
    cut: list[tuple[dict, dict]] = []
    for s, (scene, rows) in enumerate(scenes.items()):
        y = np.arange(len(rows))[::-1]
        tables = [diverged_table(r) for r in rows]
        cut += [(r, t) for r, t in zip(rows, tables)]
        axes = [fig.add_subplot(grid[s, c]) for c in range(len(columns))]
        for ax, column in zip(axes, columns):
            _budget_panel(ax, column, tables, y)
            _row_axes(ax, y, [_ratio_label(solver_name(r), t) for r, t in zip(rows, tables)] if ax is axes[0]
                      else None)
        axes[0].text(0, 1.06, labels.get(scene, scene), transform=axes[0].transAxes, ha="left", va="bottom",
                     fontsize=8, fontweight="bold", color=INK)
    _legend(fig, _box_handles() + _cut_handles(cut))
    return fig


SCENE_SOLVERS = ("VBD", "AAAVBD", "Chebyshev", "AndersonSolver", "Newton")  # the table's columns


def solve_rows(result: dict) -> list[dict]:
    """Every solve status row of a result (one per substep: the VBD family's ``vbd`` rows or GPU Newton's ``newton``
    rows), in step order, each with its step's ``t`` added."""
    rows = []
    for step in result["steps"]:
        status = step.get("status") or {}
        for row in status.get("vbd", status.get("newton", [])):
            rows.append({**row, "t": step["t"]})
    return rows


def contact_summary(result: dict) -> dict | None:
    """The run's contact: the largest feasibility (penetration) ratio to its tolerance over its solves, ``max``, and
    the number of solves above 1, ``unmet`` (contact constraints unmet at the end of the solve); None without
    recorded feasibility ratios. Non-finite ratios (after a blow-up) are left out of the maximum."""
    values = np.array([r["feasibility_ratio"] for r in solve_rows(result) if r.get("feasibility_ratio") is not None],
                      dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {"max": float(values.max()), "unmet": int((values > 1.0).sum())}


def residual_summary(result: dict) -> dict | None:
    """The run's residual: the median, minimum and maximum over its solves of the smoothed stationarity test's own
    ratio (:func:`status_ratio.smoothed_test_ratio`); None without smoothed ratios (recorded without smoothing, or
    before the field existed). Non-finite ratios (after a blow-up) are left out."""
    values = np.array([status_ratio.smoothed_test_ratio(r) for r in solve_rows(result)], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {"median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())}


def scene_cell(production: dict) -> dict:
    """One solver's cell of the ``scenes`` table, from its production run: the step time's mean and P90 (ms),
    the median iterations per step (VBD sweeps, Newton steps), the fraction of early exits, the residual
    (:func:`residual_summary`, None without smoothed ratios) and the contact (:func:`contact_summary`, None without
    feasibility ratios), the last two over the steps before a blow-up (:func:`before_blowup`)."""
    table = step_stats.per_step(production)
    time = step_stats.describe(table["time_ms"])
    iterations = step_stats.describe(table["iterations"])
    summary = step_stats.summarize_slice(table, np.ones(table["t"].size, dtype=bool))
    return {
        "mean": time["mean"] if time else None,
        "P90": time["P90"] if time else None,
        "iterations": iterations["median"] if iterations else None,
        "early_exit": summary["early_exit"],
        "residual": residual_summary(before_blowup(production)),
        "contact": contact_summary(before_blowup(production)),
    }


def _compact(count: float) -> str:
    """A count in at most 3 significant digits, with a k from 10^4 and an M from 10^6: 3269, 54.5k, 1.81M."""
    if count >= 1e6:
        return f"{count / 1e6:.3g}M"
    if count >= 1e4:
        return f"{count / 1e3:.3g}k"
    return f"{count:.0f}"


# scenes: a node's peak speed (``profiling/trajectory.py``) above this means the run blew up. Physical motion in the
# paper's scenes stays far below it (fastest at 7ce4082b: 58 m/s, ArmadilloPull's snap-back; free fall from rest
# reaches 98 m/s in 10 s), while the blown-up runs reach it within a few steps of diverging (BirthdayBanner.60Hz
# Anderson 2.2e3 m/s at 1.83 s, then 2e5; Trampoline.StiffSphere.60Hz BDF1 VBD 3.9e3 m/s at its watchdog stop)
BLOWUP_SPEED = 1e3


def run_end(result: dict) -> float:
    """Simulated time (s) at the end of the result's last step."""
    return float(result["steps"][-1]["t"] + 1) * float(result["metadata"].get("dt", 1.0))


def blowup_time(result: dict) -> float | None:
    """Simulated time (s) at the end of the first step whose positions went non-finite or very far, or None: a
    non-finite RMS or peak speed, a peak speed above :data:`BLOWUP_SPEED`, or a solve's non-finite
    ``gradient_ratio`` (its residual is computed from the positions)."""
    for step in result["steps"]:
        speeds = [step[k] for k in ("peak_speed", "rms_speed") if k in step]
        status = step.get("status") or {}
        ratios = [r.get("gradient_ratio") for r in status.get("vbd", status.get("newton", [])) if "gradient_ratio" in r]
        if (any(v is None or not np.isfinite(v) for v in speeds + ratios)
                or step.get("peak_speed", 0.0) > BLOWUP_SPEED):
            return float(step["t"] + 1) * float(result["metadata"].get("dt", 1.0))
    return None


def before_blowup(result: dict) -> dict:
    """The result with only its steps before its :func:`blowup_time` (the first blown-up step left out too), or the
    result itself if it didn't blow up: the ``scenes`` table's residual and contact statistics stop at a blow-up,
    after which they measure nothing (BirthdayBanner.60Hz Anderson's smoothed ratio reached 2.8e16 before turning
    NaN)."""
    blowup = blowup_time(result)
    if blowup is None:
        return result
    dt = float(result["metadata"].get("dt", 1.0))
    return {**result, "steps": [s for s in result["steps"] if (s["t"] + 1) * dt < blowup - 0.5 * dt]}


# solvers and budget figures: a run that blew up is cut where it began to diverge (:func:`divergence_time`): from
# the last stretch before its blow-up in which its stationarity ratio's rolling median stays above this factor
# times its median over the first half of its steps before the blow-up
DIVERGENCE_FACTOR = 10.0
CUT_MARKER = dict(marker="x", markersize=3.5, markeredgewidth=0.8, color=INK, linestyle="none")


def divergence_time(result: dict) -> float | None:
    """Simulated time (s) at the start of the step where a run that blew up (:func:`blowup_time`) began to diverge,
    or None for a run that didn't blow up (a one-solve spike, e.g. ArmadilloPull.60Hz AAAVBD's 2.6e4 at its release,
    or Celebration.60Hz Newton's two at steps 32-33, isn't a blow-up, and neither is a run that ended without one).
    The start of the last stretch before the blow-up in which the rolling median (:data:`SMOOTH_STEPS`) of its
    :data:`STRIP_RATIO` stays above :data:`DIVERGENCE_FACTOR` times its median over the first half of its steps
    before the blow-up; the blow-up itself where the rolling median is below that just before it. A detected blow-up
    trails the divergence: BirthdayBanner.60Hz Anderson's ratio rose from ~80 at 1.0 s to 1e3 at 1.12 s and 1e5-1e6
    after, while its nodes passed :data:`BLOWUP_SPEED` only at 1.83 s (7ce4082b; equal sweep: diverged at 1.08 s,
    equal time at 1.13 s)."""
    blowup = blowup_time(result)
    if blowup is None:
        return None
    table = ratio_table(result)  # NaN from the blow-up on
    dt = float(result["metadata"].get("dt", 1.0))
    ratio = np.asarray(table[STRIP_RATIO], dtype=float)
    keep = np.isfinite(ratio) & ((table["t"] + 1) * dt < blowup - 0.5 * dt)
    ratio, steps = ratio[keep], np.asarray(table["t"])[keep]
    if ratio.size < 2:
        return blowup
    baseline = float(np.median(ratio[: ratio.size // 2]))
    above = rolling_median(ratio, SMOOTH_STEPS) > DIVERGENCE_FACTOR * baseline
    if not above[-1]:
        return blowup
    i = ratio.size - 1
    while i > 0 and above[i - 1]:
        i -= 1
    return float(steps[i]) * dt


def diverged_table(result: dict) -> dict:
    """:func:`ratio_table` for the solvers and budget figures: a run that blew up has its stationarity and
    penetration ratios (``feasibility_ratio`` and :data:`FEASIBILITY_STRIP`) NaN from where it began to diverge
    (:func:`divergence_time`) on, so the other runs set the axes; ``diverged`` is that time (None: no cut), and
    ``cut_at`` each cut column's last value drawn, where the figures mark the cut. Sweeps and step times keep every
    step (the work the run did)."""
    table = ratio_table(result)
    start = divergence_time(result)
    table = {**table, "diverged": start, "cut_at": {}}
    if start is None:
        return table
    dt = float(result["metadata"].get("dt", 1.0))
    after = table["t"] * dt >= start - 0.5 * dt
    for key in (STRIP_RATIO, "feasibility_ratio", FEASIBILITY_STRIP):
        values = np.where(after, np.nan, np.asarray(table[key], dtype=float))
        table[key] = values
        drawn = values[np.isfinite(values)]
        table["cut_at"][key] = float(drawn[-1]) if drawn.size else None
    return table


def _cut_marker(ax, y: float, table: dict, key: str):
    """The cut's mark on a row (:func:`diverged_table`): a cross at the last value drawn (at least
    :data:`RATIO_FLOOR`)."""
    value = (table.get("cut_at") or {}).get(key)
    if value is not None:
        ax.plot(max(value, RATIO_FLOOR), y, **CUT_MARKER)


def _cut_handles(runs: list[tuple[dict | None, dict | None]]) -> list:
    """Legend entries of the runs cut at their divergence (:func:`diverged_table`): "Anderson: diverged at 1.08 s"."""
    return [Line2D([], [], label=f"{solver_name(r)}: diverged at {t['diverged']:.2f} s", **CUT_MARKER)
            for r, t in runs if r is not None and t is not None and t.get("diverged") is not None]


def automatic_outcome(result: dict, scene_end: float) -> str | None:
    """The table's automatic outcome mark of a run: ``blew up at t = X s`` at its :func:`blowup_time` (where its
    residual and contact statistics stop, :func:`before_blowup`), else ``stopped at t = X s`` if its output ends
    before the scene's (``scene_end``, the longest run's end; e.g. a watchdog's stop), else None."""
    blowup = blowup_time(result)
    if blowup is not None:
        return f"blew up at t = {blowup:.3g} s"
    end = run_end(result)
    if end < scene_end - 0.5 * float(result["metadata"].get("dt", 1.0)):
        return f"stopped at t = {end:.3g} s"
    return None


def load_outcomes(path: str) -> dict[str, dict[str, str]]:
    """The ``scenes`` table's outcome annotations (``--outcomes``, TOML): one table per scene, keyed like the table's
    rows by the scene file's key (:func:`scene_key`), optionally qualified by the frame rate 1/dt for scenes run at
    several (``"capsule_bowl.30Hz"``), with a short note per solver (its ``solver`` metadata, e.g. ``AndersonSolver``,
    or its display name). Unquoted dotted table names (``[capsule_bowl.30Hz]``) are read as the quoted ones."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: dict[str, dict[str, str]] = {}

    def walk(prefix: str, table: dict):
        notes = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if notes:
            out[prefix] = {k: str(v) for k, v in notes.items()}
        for k, v in table.items():
            if isinstance(v, dict):
                walk(f"{prefix}.{k}" if prefix else k, v)

    walk("", data)
    known = set(SCENE_SOLVERS) | {SOLVER_NAMES.get(s, s) for s in SCENE_SOLVERS}
    for scene, notes in out.items():
        for solver in set(notes) - known:
            warnings.warn(f"{path}: [{scene}] has an unknown solver {solver!r}")
    return out


def frame_rate(result: dict) -> str:
    """The result's frame rate 1/dt as an outcome key qualifier, e.g. ``30Hz``."""
    return f"{round(1.0 / float(result['metadata']['dt']))}Hz" if result["metadata"].get("dt") else ""


def outcome_note(outcomes: dict[str, dict[str, str]], result: dict) -> str | None:
    """A run's annotation in ``outcomes`` (:func:`load_outcomes`): the scene's table qualified by the run's frame rate
    first, then the plain one; its solver's note by ``solver`` metadata, then display name. None if not annotated
    (an empty note counts as none)."""
    solver = str(result["metadata"].get("solver", ""))
    for key in (f"{scene_key(result)}.{frame_rate(result)}", scene_key(result)):
        notes = outcomes.get(key)
        if notes is not None:
            note = notes.get(solver, notes.get(SOLVER_NAMES.get(solver, solver)))
            return note or None
    return None


# LaTeX for the characters an outcome note may hold (the rest are written as they are)
LATEX_TEXT = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
              "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "−": "$-$", "–": "--",
              "—": "---", "×": r"$\times$"}
SCENE_LABEL_WIDTH = "2.6cm"  # scenes: the scene column, its label wrapped over the scene's lines
OUTCOME_WIDTH = "1.9cm"  # scenes: an outcome cell, wrapped


def _latex_text(text: str) -> str:
    return "".join(LATEX_TEXT.get(c, c) for c in text)


def latex_scenes(runs: list[dict], labels: dict[str, str], outcomes: dict[str, dict[str, str]] | None = None) -> str:
    """The ``scenes`` table (booktabs, multirow): four or five lines per scene, a column per solver of
    :data:`SCENE_SOLVERS`. The scene's nodes, tetrahedra and time step h come from its runs' metadata; per solver,
    the first line is the mean step time (P90) in ms, the second the median iterations per step and the early exits,
    the third the median smoothed stationarity ratio [min, max] over the solves (:func:`residual_summary`), the
    fourth the largest penetration ratio over the solves and the number of solves above 1
    (:func:`contact_summary`), the fifth, written where any of the scene's cells has one, the outcome: the run's note
    in ``outcomes`` (:func:`outcome_note`), else its :func:`automatic_outcome` (against the scene's longest run)."""
    outcomes = outcomes or {}
    body = []
    any_outcome = False
    for scene, rows in by_scene(runs).items():
        by_solver = {r["metadata"].get("solver"): r for r in rows}
        meta = rows[0]["metadata"]
        h = 1e3 * float(meta.get("dt", float("nan"))) / max(int(meta.get("substeps", 1)), 1)
        scene_end = max(run_end(r) for r in rows)
        first = ["", _compact(meta["n_nodes"]) if "n_nodes" in meta else "--",
                 _compact(meta["n_elements"]) if "n_elements" in meta else "--", _num(h)]
        second = ["", "", "", ""]
        residual = ["", "", "", ""]
        contact_line = ["", "", "", ""]
        outcome_line = ["", "", "", ""]
        for solver in SCENE_SOLVERS:
            if solver not in by_solver:
                for line in (first, second, residual, contact_line, outcome_line):
                    line.append("--")
                continue
            c = scene_cell(by_solver[solver])
            first.append(f"{_num(c['mean'])} ({_num(c['P90'])})")
            iterations = "--" if c["iterations"] is None else f"{c['iterations']:.0f}"
            early = "--" if c["early_exit"] is None else f"{100 * c['early_exit']:.0f}\\%"
            second.append(f"{iterations}, {early}")
            r = c["residual"]
            residual.append("--" if r is None else f"{_num(r['median'], 2)} [{_num(r['min'], 2)}, {_num(r['max'], 2)}]")
            contact = c["contact"]
            contact_line.append("--" if contact is None else f"{_num(contact['max'], 2)} ({contact['unmet']})")
            note = outcome_note(outcomes, by_solver[solver]) or automatic_outcome(by_solver[solver], scene_end)
            outcome_line.append(rf"\parbox[t]{{{OUTCOME_WIDTH}}}{{\centering\scriptsize {_latex_text(note)}}}"
                          if note else "")
        lines = [first, second, residual, contact_line]
        if any(cell not in ("", "--") for cell in outcome_line[4:]):
            lines.append(outcome_line)
            any_outcome = True
        # The label spans the scene's lines, wrapped, so a long one doesn't widen the table
        label = labels.get(scene, scene.replace("_", r"\_"))
        first[0] = rf"\multirow[t]{{{len(lines)}}}{{{SCENE_LABEL_WIDTH}}}{{\raggedright {label}}}"
        body += ([r"\addlinespace"] if body else []) + [" & ".join(line) + r" \\" for line in lines]
    outcome_caption = (r" Last, the outcome of each run by eye, against AAAVBD's where it differs; ``blew up at $t$'' "
                       r"marks a run that blew up, whose stationarity and penetration ratios stop there, and ``stopped "
                       r"at $t$'' a run stopped early, whose numbers cover the steps up to the stop." if any_outcome
                       else "")
    out = [r"\begin{table*}", r"\centering", r"\small",
           r"\caption{\textbf{GPU time per time step.} Mean time per time step of each solver on the scenes of the "
           r"paper's results and video (production runs: CUDA graph replay for the VBD family, eager for Newton), "
           r"with the 90th percentile in parentheses, and below, the median iterations per time step (VBD sweeps; "
           r"Newton steps) and the fraction of time steps that pass their convergence test within the budget; third, "
           r"the median over the run's solves (substeps) of the stationarity ratio, the residual of the "
           r"stationarity test after smoothing over each node's neighbours over its tolerance, at most 1 where that "
           r"test passes, with its minimum and maximum in brackets; fourth, "
           r"the largest penetration ratio over the run's solves (substeps), the deepest penetration into the contact "
           r"margin over its tolerance, at most 1 where every contact constraint is met at the end of the solve, with "
           r"the number of solves above 1 in parentheses. A run that blew up counts only its solves before the "
           r"blow-up in these two lines." + outcome_caption + r" Newton "
           r"and the VBD-family solvers handle contact differently, so their trajectories are not expected to match: "
           r"the Newton column reports its cost on the same scene, not the same outcome.}",
           r"\label{tab:scenes}",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrr ccccc}", r"\toprule",
           r"& & & & \multicolumn{5}{c}{Time per step, ms (P90) \;/\; iterations per step, early exits \;/\; } \\",
           r"& & & & \multicolumn{5}{c}{stationarity ratio, median [min, max] \;/\; } \\",
           r"& & & & \multicolumn{5}{c}{max.\ penetration ratio (solves $>1$)"
           + (r" \;/\; outcome" if any_outcome else "") + r"} \\",
           r"\cmidrule(lr){5-9}",
           r"Scene & Nodes & Tets & $h$ (ms) & " + " & ".join(SOLVER_NAMES.get(v, v) for v in SCENE_SOLVERS) + r" \\",
           r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


VBD_FAMILY = ("VBD", "AAAVBD", "Chebyshev", "AndersonSolver")
EQUAL_TIME_SOLVERS = ("VBD", "Chebyshev", "AndersonSolver")  # simulation: the equal-time column's order
# scene_setup, simulation: the footnote columns of --notes (see load_setup_notes)
SETUP_NOTE_COLUMNS = ("scene", "time_step", "material", "budget", "equal_time", "newton", "tolerances")


def scene_material(path: str) -> dict | None:
    """Young's modulus, Poisson's ratio and density ranges, ``{"E": (min, max), "nu": (min, max), "rho": (min,
    max)}`` in Pa, 1 and kg/m^3, over every tetrahedron of a CPU editor scene file (``state.load_scene``'s input:
    ``fem_tet_elastic_bodies/<i>/<body>/Ye``, ``nue`` and ``rhoe``; ``nu`` None in a file without ``nue``), or None
    for any other file (a Dump, ``path:fem``). Colliders aren't tetrahedra, so they don't count."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "fem_tet_elastic_bodies" not in f:
                return None
            young, nu, rho = [], [], []
            for body in f["fem_tet_elastic_bodies"].values():
                for group in body.values():
                    young.append(np.asarray(group["Ye"], dtype=float))
                    rho.append(np.asarray(group["rhoe"], dtype=float))
                    if "nue" in group:
                        nu.append(np.asarray(group["nue"], dtype=float))
    except OSError:
        return None
    if not young:
        return None
    nu = np.concatenate(nu) if len(nu) == len(young) else None  # every body's groups have nue
    young, rho = np.concatenate(young), np.concatenate(rho)
    return {"E": (float(young.min()), float(young.max())),
            "nu": None if nu is None else (float(nu.min()), float(nu.max())),
            "rho": (float(rho.min()), float(rho.max()))}


def run_material(path: str) -> dict | None:
    """Young's modulus and Poisson's ratio ranges, ``{"E": (min, max), "nu": (min, max)}``, that a run simulated:
    from its output's (.h5) per-element Lame parameters ``FemElastoDynamics/lame_mu`` and ``lame_lambda``
    (E = mu (3 lambda + 2 mu) / (lambda + mu), nu = lambda / (2 (lambda + mu))), or None without them."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "FemElastoDynamics/lame_mu" not in f or "FemElastoDynamics/lame_lambda" not in f:
                return None
            mu = np.asarray(f["FemElastoDynamics/lame_mu"], dtype=float)
            lam = np.asarray(f["FemElastoDynamics/lame_lambda"], dtype=float)
    except OSError:
        return None
    young, nu = mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))
    return {"E": (float(young.min()), float(young.max())), "nu": (float(nu.min()), float(nu.max()))}


MATERIAL_RTOL = 1e-4  # check_material: the runs' Lame parameters are float32


def check_material(scene: str, material: dict, simulated: dict, rtol: float = MATERIAL_RTOL) -> None:
    """Refuse (ValueError) a run whose simulated E or nu range (:func:`run_material`) differs from its scene file's
    (:func:`scene_material`) by more than ``rtol`` relative to the scene's value (floored at 1e-12)."""
    for key in ("E", "nu"):
        if material.get(key) is None:
            continue
        for want, got in zip(material[key], simulated[key]):
            if abs(got - want) > rtol * max(abs(want), 1e-12):
                raise ValueError(f"{scene}: the run simulated {key} in {simulated[key]}, its scene file has "
                                 f"{material[key]}")


def newton_cg_params(path: str) -> dict | None:
    """GPU Newton's CG settings per Newton step, from a Newton run's output (.h5, the profile's sibling):
    ``params/Solver/Newton/LinearSolver``'s cap ``n_max_iters`` as ``cg`` and tolerances ``rtol``, ``atol`` as
    ``cg_rtol``, ``cg_atol`` (Warp's CG stops at a residual norm of max(atol, rtol |g|), g the step's right-hand side,
    the gradient); None without one."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            attrs = f["params/Solver/Newton/LinearSolver"].attrs
            return {"cg": int(attrs["n_max_iters"]), "cg_rtol": float(attrs["rtol"]), "cg_atol": float(attrs["atol"])}
    except (OSError, KeyError):
        return None


def config_newton_budget(path: str) -> dict:
    """A scene config's (TOML, every parameter written out: ``configs/scenes``) Newton budget, for a scene whose
    Newton run hasn't finished: substeps, outer iterations, Newton steps per subproblem, the CG cap and tolerances
    (:func:`newton_cg_params`)."""
    import tomllib

    with open(path, "rb") as f:
        config = tomllib.load(f)
    newton = config["Solver"]["Newton"]
    cg = newton["LinearSolver"]
    return {"substeps": int(config["Integration"]["substeps"]), "outer": int(newton["n_max_iters"]),
            "inner": int(newton["newton"]["n_max_iters"]), "cg": int(cg["n_max_iters"]), "cg_rtol": float(cg["rtol"]),
            "cg_atol": float(cg["atol"])}


def load_setup_notes(path: str) -> dict[str, dict[str, str]]:
    """The ``setup`` table's footnotes (``--notes``, TOML): one table per scene key (:func:`scene_key`), a note per
    column of :data:`SETUP_NOTE_COLUMNS`, e.g. ``[celebration] tolerances = "4x the shared rule's"``."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    out = {str(scene): {str(k): str(v) for k, v in notes.items()} for scene, notes in data.items()}
    for scene, notes in out.items():
        for column in set(notes) - set(SETUP_NOTE_COLUMNS):
            warnings.warn(f"{path}: [{scene}] has an unknown column {column!r}")
    return out


def _vbd_params(result: dict) -> dict:
    params = result["metadata"].get("solver_params") or {}
    return params.get("vbd_params", params)


def _one(values: dict, what: str, scene: str):
    """The value every run of a scene shares, else ValueError naming the runs."""
    distinct = set(values.values())
    if len(distinct) != 1:
        raise ValueError(f"{scene}: the runs differ in {what}: {values}")
    return distinct.pop()


def setup_rows(runs: list[dict], equal_time: list[dict] | None = None, materials: dict | None = None,
               cg_params: dict | None = None, newton_configs: dict | None = None) -> list[dict]:
    """The ``setup`` table's data, one dict per scene (:func:`by_scene` order of ``runs``): frame step ``dt``,
    ``substeps``, ``bdf``, the mesh's ``nodes`` and ``tets`` (``n_nodes``, ``n_elements``), ``material``
    (``materials[scene]``, :func:`scene_material`), the VBD family's equal-sweep
    ``budget`` (substeps, outer iterations, sweeps per subproblem), the ``equal_time`` runs' sweeps per subproblem by
    solver, ``newton`` (substeps, outer iterations, Newton steps, CG cap and tolerances from ``cg_params[scene]``,
    :func:`newton_cg_params`, ``from_config``
    when it comes from ``newton_configs[scene]``, :func:`config_newton_budget`, for a scene without a Newton run) and
    the convergence ``tolerances`` (rstol, astol, ftol, smoothing passes, check interval). Every value is what the
    runs ran with (their metadata). Raises ValueError where a scene's runs differ in the time step, substeps, BDF
    order, tolerances or mesh size, or its VBD-family runs (or equal-time runs) in anything but their sweeps."""
    materials, cg_params, newton_configs = materials or {}, cg_params or {}, newton_configs or {}
    extra = by_scene(equal_time or [])
    rows = []
    for scene, scene_runs in by_scene(runs).items():
        every = scene_runs + extra.get(scene, [])

        def shared(key, what, of=every):
            return _one({f"{solver_name(r)}{' (equal time)' if r in extra.get(scene, []) else ''}": key(r)
                         for r in of}, what, scene)

        dt = shared(lambda r: round(float(r["metadata"]["dt"]), 9), "dt")
        mesh = shared(lambda r: (r["metadata"].get("n_nodes"), r["metadata"].get("n_elements")), "mesh size")
        substeps = shared(lambda r: int(r["metadata"].get("substeps", 1)), "substeps")
        bdf = shared(lambda r: int(r["metadata"].get("bdf_order", 1)), "BDF order")
        tolerances = shared(lambda r: tuple((r["metadata"].get("convergence_params") or {}).get(k) for k in
                                            ("rstol", "astol", "ftol", "smoothing_passes", "check_interval")),
                            "convergence tolerances")
        family = [r for r in scene_runs if r["metadata"].get("solver") in VBD_FAMILY]
        budget = shared(lambda r: (int(r["metadata"].get("substeps", 1)), int(_vbd_params(r)["n_max_iters"]),
                                   int(_vbd_params(r)["n_subproblem_max_iters"])), "equal-sweep budgets",
                        family) if family else None
        sweeps = {}
        if extra.get(scene):
            shared(lambda r: (int(r["metadata"].get("substeps", 1)), int(_vbd_params(r)["n_max_iters"])),
                   "equal-time substeps or outer iterations", extra[scene])
            sweeps = {r["metadata"]["solver"]: int(_vbd_params(r)["n_subproblem_max_iters"]) for r in extra[scene]}
        newton = None
        for r in scene_runs:
            if r["metadata"].get("solver") == "Newton":
                params = r["metadata"].get("solver_params") or {}
                newton = {"substeps": substeps, "outer": int(params["n_max_iters"]),
                          "inner": int(params["newton"]["n_max_iters"]), "cg": None, "cg_rtol": None,
                          "cg_atol": None, **(cg_params.get(scene) or {}), "from_config": False}
        if newton is None and scene in newton_configs:
            newton = {**newton_configs[scene], "from_config": True}
        rows.append({"scene": scene, "dt": dt, "substeps": substeps, "bdf": bdf, "nodes": mesh[0], "tets": mesh[1],
                     "material": materials.get(scene),
                     "budget": budget, "equal_time": sweeps, "newton": newton,
                     "tolerances": dict(zip(("rstol", "astol", "ftol", "smoothing", "check_interval"), tolerances))})
    return rows


def _step_fraction(h: float) -> str:
    """A time step as 1/n s where n is a whole number (1/60, 1/300), else in ms."""
    n = 1.0 / h
    return f"1/{round(n)}" if abs(n - round(n)) < 1e-3 * n else f"{_num(1e3 * h)} ms"


_SUPERSCRIPT = dict(zip("0123456789-,abcdefghijklmnoprstuvwxyz",
                        "⁰¹²³⁴⁵⁶⁷⁸⁹⁻,ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"))
# scene_setup, simulation: how the tables' three renderings (LaTeX, Markdown, the PDF drawn by Matplotlib) write a product, a range,
# a power of ten, a footnote mark, a note's text and a scene label
SETUP_STYLES = {
    "latex": {"times": r" $\times$ ", "dash": "--", "power": lambda e: f"$10^{{{e}}}$",
              "mark": lambda m: rf"$^{{\mathrm{{{m}}}}}$", "text": lambda t: _latex_text(t),
              "label": lambda s: s.replace("_", r"\_"), "none": "--"},
    "markdown": {"times": " × ", "dash": "–", "power": lambda e: "10" + "".join(_SUPERSCRIPT[c] for c in str(e)),
                 "mark": lambda m: "".join(_SUPERSCRIPT.get(c, c) for c in m), "text": str, "label": str,
                 "none": "–"},
    "mpl": {"times": " × ", "dash": "–", "power": lambda e: f"$10^{{{e}}}$", "mark": lambda m: rf"$^{{\mathrm{{{m}}}}}$",
            "text": lambda t: t.replace("$", r"\$"), "label": lambda s: s.replace("$", r"\$"), "none": "–"},
}


def _si(lo: float, hi: float, unit: str, dash: str = "--") -> str:
    """A value or range with the SI prefix of its larger end (0.1--100 MPa)."""
    for scale, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")):
        if hi >= scale:
            break
    a, b = _num(lo / scale), _num(hi / scale)
    return f"{a} {prefix}{unit}" if a == b else f"{a}{dash}{b} {prefix}{unit}"


def _power(v: float, style: str = "latex") -> str:
    """A tolerance: a power of ten as 10^e in the style's notation, else three significant digits."""
    e = np.log10(v) if v > 0 else np.nan
    return SETUP_STYLES[style]["power"](int(round(e))) if np.isfinite(e) and abs(e - round(e)) < 1e-6 else _num(v)


def _note_marker(style: str, notes: dict[str, dict[str, str]] | None):
    """A cell writer that appends footnote marks (``notes`` per scene and column, plus an automatic note), and the
    marks it gave, note text -> letter, in reading order."""
    st, notes, marks = SETUP_STYLES[style], notes or {}, {}

    def cell(value: str, scene: str, column: str, automatic: str | None = None) -> str:
        letters = []
        for text in (automatic, notes.get(scene, {}).get(column)):
            if text:
                if text not in marks:
                    marks[text] = "abcdefghijklmnopqrstuvwxyz"[len(marks)]
                letters.append(marks[text])
        return value + (st["mark"](",".join(letters)) if letters else "")

    return cell, marks


def _scene_setup_table(rows: list[dict], labels: dict[str, str], notes: dict[str, dict[str, str]] | None,
                       style: str) -> dict:
    """The ``scene_setup`` table's cells in one of :data:`SETUP_STYLES`: ``body`` (per scene: its label, the mesh's
    nodes and tetrahedra in the ``scenes`` table's format, E, nu and rho) and the footnotes of its ``material`` notes
    as (mark, text)."""
    st = SETUP_STYLES[style]
    cell, marks = _note_marker(style, notes)
    body = []
    for r in rows:
        s, material = r["scene"], r["material"]
        young = st["none"] if material is None else _si(*material["E"], "Pa", st["dash"])
        span = lambda v: (st["none"] if v is None else _num(v[0]) if v[0] == v[1]  # noqa: E731
                          else f"{_num(v[0])}{st['dash']}{_num(v[1])}")
        nu, rho = (st["none"], st["none"]) if material is None else (span(material.get("nu")), span(material["rho"]))
        size = [st["none"] if v is None else _compact(v) for v in (r["nodes"], r["tets"])]
        body.append([st["label"](labels.get(s, s)), *size, cell(young, s, "material"), nu, rho])
    return {"body": body, "notes": [(m, text) for text, m in marks.items()]}


def _simulation_table(rows: list[dict], labels: dict[str, str], notes: dict[str, dict[str, str]] | None,
                      style: str) -> dict:
    """The ``simulation`` table's cells in one of :data:`SETUP_STYLES`: ``body`` (per scene: h; the equal-sweep
    budget's s, o, n; the equal-time sweeps of VBD, Chebyshev and Anderson; Newton's s, o, k; its CG cap and relative
    tolerance; the three convergence tolerances; and the smoothing passes where they differ: one cell per
    sub-column, :data:`SIMULATION_GROUPS`), the footnotes as (mark, text) in reading order, ``atol``
    (Newton's CG absolute tolerance where every scene shares it, else None: each cell lists it), ``common``
    (settings every scene shares, for the caption) and ``with_k``."""
    st = SETUP_STYLES[style]
    cell, marks = _note_marker(style, notes)
    atols = {r["newton"]["cg_atol"] for r in rows if r["newton"] and r["newton"].get("cg_atol") is not None}
    one_atol = len(atols) <= 1
    smoothing = {r["tolerances"]["smoothing"] for r in rows}
    intervals = {r["tolerances"]["check_interval"] for r in rows}
    with_k = len(smoothing) > 1
    body = []
    for r in rows:
        s = r["scene"]
        label = cell(st["label"](labels.get(s, s)), s, "scene")
        h = cell(_step_fraction(r["dt"]), s, "time_step", "BDF2 (the others BDF1)" if r["bdf"] == 2 else None)
        # One cell per sub-column; a note goes on the cell it qualifies: the budget's on n, Newton's on its outer
        # iterations o (BasketOfFruit's 5 x 1, the config mark), the tolerances' on rstol
        budget = [st["none"]] * 3 if r["budget"] is None else [str(v) for v in r["budget"]]
        budget[2] = cell(budget[2], s, "budget")
        sweeps = r["equal_time"]
        equal = [str(sweeps[v]) if v in sweeps else st["none"] for v in EQUAL_TIME_SOLVERS]
        equal[0] = cell(equal[0], s, "equal_time")
        n = r["newton"]
        newton = [st["none"]] * 3 if n is None else [str(n[k]) for k in ("substeps", "outer", "inner")]
        if n is not None:
            newton[1] = cell(newton[1], s, "newton", "from its scene config: the run hadn't finished"
                             if n["from_config"] else None)
        cg = [st["none"], st["none"]]
        if n is not None and n.get("cg") is not None:
            cg[0] = str(n["cg"])
        if n is not None and n.get("cg_rtol") is not None:
            cg[1] = _power(n["cg_rtol"], style) + ("" if one_atol else ", " + _power(n["cg_atol"], style))
        t = r["tolerances"]
        tol = [_num(t["rstol"]), _num(t["astol"]), _num(t["ftol"])]
        tol[0] = cell(tol[0], s, "tolerances")
        body.append([label, h, *budget, *equal, *newton, *cg, *tol] + ([str(t["smoothing"])] if with_k else []))
    common = []
    if not with_k and smoothing:
        common.append(f"{smoothing.pop()} smoothing passes")
    if len(intervals) == 1:
        common.append(f"a convergence check every {intervals.pop()} sweeps")
    return {"body": body, "notes": [(m, text) for text, m in marks.items()], "common": common, "with_k": with_k,
            "atol": next(iter(atols)) if one_atol and atols else None}


SCENE_LABEL_WIDTH_SETUP = "2.4cm"  # scene_setup, simulation: the scene column, its label wrapped


def latex_scene_setup(rows: list[dict], labels: dict[str, str],
                      notes: dict[str, dict[str, str]] | None = None) -> str:
    """The ``scene_setup`` table (booktabs) of :func:`setup_rows`: per scene the mesh's size and its bodies' Young's
    modulus, Poisson's ratio and density."""
    t = _scene_setup_table(rows, labels, notes, "latex")
    caption = (r"\caption{\textbf{Scene setup.} Per scene: its mesh's nodes and tetrahedra, and the Young's modulus "
               r"$E$, Poisson's ratio $\nu$ and density $\rho$ of its bodies, a range where bodies or regions differ."
               + "".join(rf" $^{{\mathrm{{{m}}}}}${_latex_text(text)}." for m, text in t["notes"]) + "}")
    out = [r"\begin{table*}", r"\centering", r"\small", caption, r"\label{tab:scene_setup}",
           r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrccc}", r"\toprule",
           r"Scene & Nodes & Tets & $E$ & $\nu$ & $\rho$ (kg/m$^3$) \\", r"\midrule",
           *[" & ".join(cells) + r" \\" for cells in t["body"]], r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def _simulation_caption(t: dict, style: str) -> str:
    """The ``simulation`` table's caption without its footnotes, its quantities written as :data:`SIMULATION_SYMBOLS`
    (LaTeX: their macros; Markdown: plain; Matplotlib: plain, mathtext for powers of ten)."""
    if style == "latex":
        sym = lambda *keys: "$" + r" \times ".join("\\" + SIMULATION_SYMBOLS[k][0] for k in keys) + "$"  # noqa: E731
        step = r"$\SimStep/\SimSubsteps$"
        norm = r"$\max(\SimCGAtol, \SimCGRtol \lVert g \rVert)$"
    else:
        sym = lambda *keys: " × ".join(SIMULATION_SYMBOLS[k][2] for k in keys)  # noqa: E731
        step = "h/s"
        norm = "max(a_CG, r_CG‖g‖)" if style == "markdown" else "max(a_CG, r_CG|g|)"
    g = "$g$" if style == "latex" else "g"
    if t["atol"] is not None:
        value = _power(t["atol"], style)
        atol = (rf", and $\SimCGAtol = {value[1:-1]}$ in every scene" if style == "latex"
                else f", and a_CG = {value} in every scene")
    else:
        atol = f", with the absolute tolerance {sym('cg_a')} after {sym('cg_r')}"
    return (f"Per scene: the frame time step {sym('h')}, split into {sym('s')} substeps, so that each solve's time "
            f"step is {step}. The budget per frame, {sym('s', 'm', 'n')}: {sym('s')} substeps, {sym('m')} outer "
            f"iterations (subproblems) per substep and {sym('n')} VBD sweeps per subproblem for the VBD-family solvers "
            f"at equal sweeps (VBD, AAAVBD, Chebyshev, Anderson); at equal time, the sweeps per subproblem {sym('n')} "
            f"of VBD, Chebyshev and Anderson (AAAVBD's time per step, {sym('s')} and {sym('m')} unchanged); for "
            f"Newton, {sym('s', 'm', 'k')} with {sym('k')} Newton steps per subproblem, each solving its linear system "
            f"by conjugate gradients (CG) with at most {sym('cg_cap')} iterations and the relative tolerance "
            f"{sym('cg_r')} (CG stops at a residual norm of {norm}, {g} the gradient, the Newton step's right-hand "
            f"side{atol}). Convergence tolerances: the stationarity test's relative and absolute tolerances "
            f"{sym('rstol')}, {sym('astol')} and the feasibility test's {sym('ftol')}, the same for every solver of a "
            "scene" + (" (" + ", ".join(t["common"]) + " in every scene)" if t["common"] else "") + ".")


def latex_simulation(rows: list[dict], labels: dict[str, str], notes: dict[str, dict[str, str]] | None = None) -> str:
    """The ``simulation`` table (booktabs) of :func:`setup_rows`: per scene its frame step, budgets and convergence
    tolerances, compact enough for the text width (two-line headers, the scene labels wrapped, 3 pt column
    separation); ``notes`` (:func:`load_setup_notes`) become footnote marks explained in the caption, plus automatic
    ones for BDF2 scenes and for Newton budgets read from a config."""
    t = _simulation_table(rows, labels, notes, "latex")
    with_k = t["with_k"]
    caption = (r"\caption{\textbf{Simulation parameters.} " + _simulation_caption(t, "latex")
               + "".join(rf" $^{{\mathrm{{{m}}}}}${_latex_text(text)}." for m, text in t["notes"]) + "}")
    body = [rf"\parbox[t]{{{SCENE_LABEL_WIDTH_SETUP}}}{{\raggedright {cells[0]}}} & " + " & ".join(cells[1:]) + r" \\"
            for cells in t["body"]]
    spans, rules, names, column = [], [], [], 3  # LaTeX columns are 1-based: Scene, h, then the groups
    for title, symbol, keys in SIMULATION_GROUPS:
        spans.append(rf"\multicolumn{{{len(keys)}}}{{c}}{{{_group_title(title, symbol, 'latex')}}}")
        rules.append(rf"\cmidrule(lr){{{column}-{column + len(keys) - 1}}}")
        names += [_symbol(key, "latex") for key in keys]
        column += len(keys)
    k = " & passes" if with_k else ""
    columns = "l c " + " ".join("c" * len(g[2]) for g in SIMULATION_GROUPS) + (" c" if with_k else "")
    out = [*_simulation_macros(), r"\begin{table*}", r"\centering", r"\small", caption, r"\label{tab:simulation}",
           r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
           " & & " + " & ".join(spans) + (" &" if with_k else "") + r" \\", " ".join(rules),
           r"Scene & $\SimStep$ (s) & " + " & ".join(names) + k + r" \\",
           r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


SCENE_SETUP_HEADERS = ["Scene", "Nodes", "Tets", "E", "ν", "ρ (kg/m³)"]  # Markdown; the PDF's in mathtext
# simulation: the table's quantities as variables (maintainer, 2026-11-16: "easy for me to rename the variables
# later"): key -> (LaTeX macro, its definition, plain text, Matplotlib mathtext). The .tex declares each macro once
# at its top with \providecommand, so a rename is a one-line edit there, or in the paper's preamble.
SIMULATION_SYMBOLS = {
    "h": ("SimStep", "h", "h", "$h$"),
    "s": ("SimSubsteps", "s", "s", "$s$"),
    "m": ("SimSubproblems", "m", "m", "$m$"),
    "n": ("SimSweeps", "n", "n", "$n$"),
    "k": ("SimNewtonSteps", "k", "k", "$k$"),
    "cg_cap": ("SimCGCap", r"n_\mathrm{CG}", "n_CG", r"$n_\mathrm{CG}$"),
    "cg_r": ("SimCGRtol", r"r_\mathrm{CG}", "r_CG", r"$r_\mathrm{CG}$"),
    "cg_a": ("SimCGAtol", r"a_\mathrm{CG}", "a_CG", r"$a_\mathrm{CG}$"),
    "rstol": ("SimRstol", r"\epsilon_r", "ε_r", r"$\epsilon_r$"),
    "astol": ("SimAstol", r"\epsilon_a", "ε_a", r"$\epsilon_a$"),
    "ftol": ("SimFtol", r"\epsilon_f", "ε_f", r"$\epsilon_f$"),
}
# simulation: the column groups after Scene and h: (title, title symbol or None, sub-columns: symbols or labels)
SIMULATION_GROUPS = (
    ("VBD family", None, ("s", "m", "n")),
    ("Equal time", "n", ("VBD", "Cheb.", "And.")),
    ("Newton", None, ("s", "m", "k")),
    ("CG", None, ("cg_cap", "cg_r")),
    ("Convergence tolerances", None, ("rstol", "astol", "ftol")),
)


def _symbol(key: str, style: str) -> str:
    """A table quantity in a rendering's notation: LaTeX its macro in math mode, Markdown plain, Matplotlib mathtext;
    a label that isn't a symbol (VBD, Cheb.) as it is."""
    if key not in SIMULATION_SYMBOLS:
        return key
    macro, _, plain, mathtext = SIMULATION_SYMBOLS[key]
    return {"latex": "$\\" + macro + "$", "markdown": plain, "mpl": mathtext}[style]


def _group_title(title: str, symbol: str | None, style: str) -> str:
    return title if symbol is None else f"{title}, {_symbol(symbol, style)}"


def _simulation_macros() -> list[str]:
    """The LaTeX lines declaring :data:`SIMULATION_SYMBOLS`, for the top of the table's file."""
    return (["% The table's symbols, declared once: rename one here (or define it in the paper's preamble, which wins:",
             "% \\providecommand keeps an existing definition)"]
            + [rf"\providecommand{{\{macro}}}{{{body}}}" for macro, body, _, _ in SIMULATION_SYMBOLS.values()] + [""])


def _simulation_headers(with_k: bool) -> list[str]:
    """The ``simulation`` table's plain-text column names (Markdown)."""
    prefixes = {"Convergence tolerances": "Convergence"}
    return (["Scene", "h (s)"]
            + [f"{prefixes.get(title, _group_title(title, symbol, 'markdown'))}: {_symbol(key, 'markdown')}"
               for title, symbol, keys in SIMULATION_GROUPS for key in keys] + (["passes"] if with_k else []))


def _markdown_table(title: str, headers: list[str], body: list[list[str]], caption: str,
                    notes: list[tuple[str, str]]) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    lines += ["", caption, ""] + [f"- {SETUP_STYLES['markdown']['mark'](m)} {text}" for m, text in notes]
    return "\n".join(lines) + "\n"


SCENE_SETUP_CAPTION = ("Per scene: its mesh's nodes and tetrahedra (tets), and the Young's modulus E, Poisson's "
                       "ratio ν and density ρ of its bodies, a range where bodies or regions differ.")


def markdown_scene_setup(rows: list[dict], labels: dict[str, str],
                         notes: dict[str, dict[str, str]] | None = None) -> str:
    """The ``scene_setup`` table as Markdown, for reading without LaTeX."""
    t = _scene_setup_table(rows, labels, notes, "markdown")
    return _markdown_table("Scene setup", SCENE_SETUP_HEADERS, t["body"], SCENE_SETUP_CAPTION, t["notes"])


def markdown_simulation(rows: list[dict], labels: dict[str, str],
                        notes: dict[str, dict[str, str]] | None = None) -> str:
    """The ``simulation`` table as Markdown, for reading without LaTeX."""
    t = _simulation_table(rows, labels, notes, "markdown")
    return _markdown_table("Simulation parameters", _simulation_headers(t["with_k"]), t["body"],
                           _simulation_caption(t, "markdown"), t["notes"])


def _plot_table(title: str, headers: list[str], body: list[list[str]], widths: list[int], caption: str,
                notes: list[tuple[str, str]], groups: list[tuple[int, int, str]] = (),
                size: tuple[float, float] = (11.69, 8.27)):
    """A table drawn by Matplotlib on a landscape page (A4 by default): ``headers`` and ``body`` cells wrapped to
    ``widths`` characters per column, one height per row, ``groups`` (first column, last column, title) as spanning
    headers above their columns, then the caption and the footnotes under the table."""
    import textwrap

    def fill(text: str, width: int) -> str:
        """Wrapped to ``width`` characters, a trailing footnote mark ($...$) kept whole after the last line."""
        plain, sep, mark = text.partition("$")
        wrapped = "\n".join(textwrap.fill(line, width) for line in plain.split("\n"))
        return wrapped + (" " if plain.endswith(" ") else "") + sep + mark

    top = [""] * len(headers)
    for first, last, name in groups:
        top[(first + last) // 2] = name
    header_rows = [top] if groups else []
    wrap = [[fill(text, width) for text, width in zip(cells, widths)] for cells in [headers] + body]
    if groups:  # a group's title spans its columns: wrapped to their width
        spans = {(f + l) // 2: sum(widths[f:l + 1]) for f, l, _ in groups}
        wrap.insert(0, [fill(text, spans.get(col, width)) for col, (text, width) in enumerate(zip(top, widths))])
    lines = [max(c.count("\n") + 1 for c in cells) for cells in wrap]
    fig = plt.figure(figsize=size)
    fig.text(0.04, 0.96, title, fontsize=14, fontweight="bold", va="top", color=INK)
    ax = fig.add_axes((0.04, 0.10, 0.92, 0.82))
    ax.set_axis_off()
    table = ax.table(cellText=wrap, colWidths=[w / sum(widths) for w in widths], loc="upper center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    n_head = len(header_rows) + 1
    axes_height = 0.82 * size[1]  # inches
    for (row, col), c in table.get_celld().items():
        # One height per table row, so the cells line up: 0.12 in per text line plus 0.07 in of padding
        c.set_height((0.12 * lines[row] + 0.07) / axes_height)
        c.set_edgecolor("#bbbbbb")
        c.set_linewidth(0.5)
        if row < n_head:
            c.set_facecolor("#eeeeee")
            c.get_text().set_fontweight("bold")
        if row == 0 and groups:  # the spanning headers: open inside a group, no cell outside one
            group = next(((f, l) for f, l, _ in groups if f <= col <= l), None)
            c.set_facecolor("none")  # an open-edged cell fills as a stray polygon
            if group is None:
                c.visible_edges = ""
            else:
                c.visible_edges = "T" + "B" + ("L" if col == group[0] else "") + ("R" if col == group[1] else "")
        if col == 0 and row >= n_head:
            c.get_text().set_ha("left")
            c.PAD = 0.03
    fig.canvas.draw()
    bottom = table.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
    notes_text = "\n".join(textwrap.fill(SETUP_STYLES["mpl"]["mark"](m) + " " + text, 180) for m, text in notes)
    fig.text(0.04, bottom - 0.03, textwrap.fill(caption, 180) + ("\n\n" + notes_text if notes else ""),
             fontsize=9, va="top", color=INK, linespacing=1.4)
    return fig


def plot_scene_setup(rows: list[dict], labels: dict[str, str], notes: dict[str, dict[str, str]] | None = None):
    """The ``scene_setup`` table drawn by Matplotlib (a small page: five columns)."""
    t = _scene_setup_table(rows, labels, notes, "mpl")
    headers = ["Scene", "Nodes", "Tets", "$E$", r"$\nu$", r"$\rho$ (kg/m³)"]
    return _plot_table("Scene setup", headers, t["body"], [24, 8, 8, 14, 7, 12], SCENE_SETUP_CAPTION,
                       t["notes"], size=(8.27, 6.0))


def plot_simulation(rows: list[dict], labels: dict[str, str], notes: dict[str, dict[str, str]] | None = None):
    """The ``simulation`` table drawn by Matplotlib on a landscape page, each group's sub-columns under one
    spanning header (:data:`SIMULATION_GROUPS`)."""
    t = _simulation_table(rows, labels, notes, "mpl")
    headers, groups, column = ["Scene", "$h$ (s)"], [], 2
    for title, symbol, keys in SIMULATION_GROUPS:
        headers += [_symbol(key, "mpl") for key in keys]
        groups.append((column, column + len(keys) - 1, _group_title(title, symbol, "mpl")))
        column += len(keys)
    headers += ["passes"] if t["with_k"] else []
    widths = [20, 7] + [5, 5, 6] + [6, 6, 6] + [5, 5, 5] + [6, 7] + [6, 6, 6] + ([4] if t["with_k"] else [])
    return _plot_table("Simulation parameters", headers, t["body"], widths, _simulation_caption(t, "mpl"),
                       t["notes"], groups=groups)


LEAD_REFERENCE = "AAAVBD"  # lead: the solver every other one is divided by
LEAD_SOLVERS = ("VBD", "Chebyshev", "AndersonSolver")  # lead: one line each
# The stationarity ratios of a status row (status_ratio): the decisive one, at most 1 iff the test passed, and the
# smoothed test's own, which ranks solvers where float32 noise makes the unsmoothed one rough (NaN on rows without
# smoothing)
STATIONARITY_RATIOS = {"decisive": status_ratio.stationarity_test_ratio, "smoothed": status_ratio.smoothed_test_ratio}
# lead: the ratios compared, the smoothed one only (maintainer, 2026-11-12: the only one trusted, as in the scenes
# table and the timeline; until then the decisive ratio was drawn too)
LEAD_RATIOS = ("smoothed",)
LEAD_SOURCES = {"full": "o", "window": "s"}  # lead: the marker of points from full runs and from time windows


def solve_end_times(result: dict) -> np.ndarray:
    """Simulated time (s) at the end of each solve of :func:`solve_rows`: its step's start plus its substep's share."""
    meta = result["metadata"]
    dt, substeps = float(meta.get("dt", 1.0)), max(int(meta.get("substeps", 1)), 1)
    return np.array([(r["t"] + (r.get("substep", 0) + 1) / substeps) * dt for r in solve_rows(result)])


def window_median(result: dict, ratio: str, t0: float, t1: float) -> float | None:
    """The median of a :data:`STATIONARITY_RATIOS` ratio over the result's solves that end within (t0, t1] (s),
    leaving out non-finite values; None if none is left."""
    rows = solve_rows(result)
    values = np.array([np.nan if (v := STATIONARITY_RATIOS[ratio](r)) is None else v for r in rows], dtype=float)
    end = solve_end_times(result)
    # Window ends on the step grid, and dt is float32-rounded (90 x 1/30 s ends at 3.0000001 s): compare within a
    # small fraction of a substep
    eps = 1e-3 * float(result["metadata"].get("dt", 1.0)) / max(int(result["metadata"].get("substeps", 1)), 1)
    inside = (end > t0 + eps) & (end <= t1 + eps)
    values = values[inside & np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def lead_points(full: list[dict], windowed: list[dict] | None = None,
                windows: dict[str, tuple[float, float]] | None = None) -> list[dict]:
    """The ``lead`` figure's points: per scene, solver step h (dt / substeps) and solver of :data:`LEAD_SOLVERS`, the
    median of each :data:`LEAD_RATIOS` ratio over its solves divided by :data:`LEAD_REFERENCE`'s at the same scene and
    step, over the same window (the medians floored at :data:`RATIO_FLOOR`). ``full`` runs are compared over their
    scene's common window: from the latest first recorded step of any of its runs (production runs skip their warm-up
    steps) to the end of the shortest campaign (the longest run at each step); ``windowed`` runs over their scene's
    ``windows`` entry (s). Either is narrowed to the span both runs recorded and to before either blew up
    (:func:`blowup_time`), so a run stopped early or blown up is compared with AAAVBD up to then: after a blow-up its
    ratios measure nothing (BirthdayBanner at 10 ms, VBD from 2.66 s: smoothed ratios at 0).

    Returns:
        list[dict]: ``scene``, ``source`` (``full`` or ``window``), ``h``, ``dt``, ``substeps`` (s), ``solver``,
        ``span`` (the scene's window or common span, s), ``window`` (the point's, narrowed), ``blowup`` (whether a
        blow-up cut the window), and per ratio its lead (None where either
        median is unknown) and AAAVBD's median (``<ratio>_reference``).
    """
    points = []
    for source, runs in (("full", full), ("window", windowed or [])):
        for scene, scene_runs in by_scene(runs).items():
            steps: dict[tuple, dict[str, dict]] = {}
            for r in scene_runs:
                meta = r["metadata"]
                key = (round(float(meta["dt"]), 9), max(int(meta.get("substeps", 1)), 1))
                steps.setdefault(key, {})[meta.get("solver")] = r
            if source == "full":
                t0 = max(float(r["steps"][0]["t"]) * float(r["metadata"]["dt"]) for r in scene_runs)
                t1 = min(max(run_end(r) for r in by_solver.values()) for by_solver in steps.values())
            elif scene in (windows or {}):
                t0, t1 = windows[scene]
            else:
                raise SystemExit(f"lead: the windowed runs of {scene} need --window {scene}=T0:T1")
            for (dt, substeps), by_solver in sorted(steps.items()):
                reference = by_solver.get(LEAD_REFERENCE)
                if reference is None:
                    warnings.warn(f"lead: {scene} at dt {dt:g} s x {substeps} has no {LEAD_REFERENCE} run")
                    continue
                for solver in LEAD_SOLVERS:
                    run = by_solver.get(solver)
                    if run is None:
                        continue
                    lo = max(t0, *(float(r["steps"][0]["t"]) * dt for r in (run, reference)))
                    hi = min(t1, run_end(run), run_end(reference))
                    # Up to the step before the first blown-up one
                    blowups = [b - dt for b in map(blowup_time, (run, reference)) if b is not None and b - dt < hi]
                    hi = min([hi, *blowups])
                    point = {"scene": scene, "source": source, "h": dt / substeps, "dt": dt, "substeps": substeps,
                             "solver": solver, "span": (t0, t1), "window": (lo, hi), "blowup": bool(blowups)}
                    for ratio in LEAD_RATIOS:
                        ref, other = window_median(reference, ratio, lo, hi), window_median(run, ratio, lo, hi)
                        point[f"{ratio}_reference"] = ref
                        point[ratio] = (None if ref is None or other is None
                                        else max(other, RATIO_FLOOR) / max(ref, RATIO_FLOOR))
                    points.append(point)
    return points


def lead_text(points: list[dict], labels: dict[str, str]) -> str:
    """Markdown: the ``lead`` figure's points, for checking them against the text."""
    out = ["| Scene | Points | Window (s) | h (ms) | dt (ms) x substeps | AAAVBD stationarity ratio | Solver | "
           "Lead |", "|---|---|---|---|---|---|---|---|"]

    def times(x):
        return "--" if x is None else f"{_num(x, 2)}x"

    for p in points:
        ref = _num(p["smoothed_reference"])
        lead = times(p["smoothed"])
        window = f"{p['window'][0]:.3g}-{p['window'][1]:.3g}" + (" (blow-up)" if p["blowup"] else "")
        out.append(f"| {labels.get(p['scene'], p['scene'])} | {p['source']} | {window} | {_num(1e3 * p['h'])} | "
                   f"{_num(1e3 * p['dt'])} x {p['substeps']} | {ref} | {SOLVER_NAMES.get(p['solver'], p['solver'])} | "
                   f"{lead} |")
    return "\n".join(out) + "\n"


def plot_lead(points: list[dict], labels: dict[str, str], width: float = 7.0):
    """The ``lead`` figure (see the module docstring): one panel per scene, each other solver's lead over AAAVBD
    (:func:`lead_points`) against the solver step h, both on log axes."""
    scenes = list(dict.fromkeys(p["scene"] for p in points))
    ncols = min(len(scenes), max(1, int(width // 2.2)))
    nrows = -(-len(scenes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 1.5 * nrows + 0.2), squeeze=False,
                             gridspec_kw={"wspace": 0.38, "hspace": 0.75})
    for ax, scene in zip(axes.flat, scenes):
        mine = [p for p in points if p["scene"] == scene]
        values = [1.0]
        for solver in LEAD_SOLVERS:
            color = SOLVER_COLORS.get(solver, INK)
            for source, marker in LEAD_SOURCES.items():
                for ratio in LEAD_RATIOS:
                    series = sorted((p["h"], p[ratio]) for p in mine
                                    if p["solver"] == solver and p["source"] == source and p[ratio] is not None)
                    if not series:
                        continue
                    h, y = (np.array(v) for v in zip(*series))
                    values += list(y)
                    ax.plot(1e3 * h, y, color=color, linewidth=0.9, marker=marker, markersize=3.2)
        for p in mine:  # compared only up to a blow-up
            if p["blowup"]:
                y = [p[ratio] for ratio in LEAD_RATIOS if p[ratio] is not None]
                ax.plot([1e3 * p["h"]] * len(y), y, color=INK, marker="x", markersize=4.5, markeredgewidth=0.7,
                        linestyle="none", zorder=5)
        ax.axhline(1.0, color=INK, linewidth=0.5, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        steps = sorted({1e3 * p["h"] for p in mine})
        ax.set_xlim(steps[0] / 1.35, steps[-1] * 1.35)
        ticks = ax.set_xticks(steps, [_num(h) for h in steps])
        for a, b, label_a, label_b in zip(steps, steps[1:], ticks, ticks[1:]):
            if b / a < 1.4:  # close steps (8.33 and 10 ms): their labels apart
                label_a.label1.set_horizontalalignment("right")
                label_b.label1.set_horizontalalignment("left")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(min(values) / 1.4, max(values) * 1.4)
        _log_axis(ax, "y")
        _style(ax)
        ax.text(0, 1.06, labels.get(scene, scene), transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                fontweight="bold", color=INK)
        spans = sorted({p["span"] for p in mine if p["source"] == "window"})
        if spans:  # the scene's window, which the caption can't hold per scene
            ax.text(1, 1.06, ", ".join(f"{a:g}–{b:g} s" for a, b in spans), transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=FONT_SIZE - 1, color="#7f7f7f")
    for ax in axes.flat[len(scenes):]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Ratio to {LEAD_REFERENCE}", fontsize=FONT_SIZE)
    for c in range(ncols):  # the lowest panel of each column
        rows = [r for r in range(nrows) if r * ncols + c < len(scenes)]
        if rows:
            axes[rows[-1], c].set_xlabel("Time step $h$ (ms)", fontsize=FONT_SIZE)
    handles = [Line2D([], [], color=SOLVER_COLORS.get(s, INK), linewidth=0.9, label=SOLVER_NAMES.get(s, s))
               for s in LEAD_SOLVERS if any(p["solver"] == s for p in points)]
    sources = {p["source"] for p in points}
    handles += [Line2D([], [], color=INK, marker=LEAD_SOURCES[s], markersize=3.2, linestyle="none",
                       label={"full": "Full run", "window": "Window"}[s]) for s in LEAD_SOURCES if s in sources]
    if any(p["blowup"] for p in points):
        handles.append(Line2D([], [], color=INK, marker="x", markersize=4.5, markeredgewidth=0.7, linestyle="none",
                              label="Until a blow-up"))
    handles.append(Line2D([], [], color=INK, linewidth=0.5, linestyle=(0, (3, 2)), label=LEAD_REFERENCE))
    _legend(fig, handles)
    return fig


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
    parser.add_argument("kind", choices=["timeline", "solvers", "slices", "sweeps", "budget", "scenes", "lead", "scene_setup",
                                             "simulation"],
                        help="What to write (see the docstring).")
    parser.add_argument("kernels", nargs="+",
                        help="Results (.json), one per row: kernels-backend runs (timeline, solvers, slices, sweeps) "
                        "or production runs (budget, scenes; scene_setup, simulation: at equal sweeps, Newton's "
                        "included; lead: the runs compared over their full common span).")
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
    parser.add_argument("--rms-speed", choices=RMS_SPEED_MODES, default="paired", dest="rms_speed",
                        help="timeline: draw the RMS speed of the paired production run only (default), or of every "
                        "production run (the others' as dotted lines).")
    parser.add_argument("--vbd-ratio", action="store_true", dest="vbd_ratio",
                        help="timeline: add a strip of each production run's step time over VBD's, which replaces the "
                        "other runs' step-time lines on the cost plot (off by default).")
    parser.add_argument("--snapshots", nargs="*", default=[], metavar="TIME=IMAGE",
                        help="timeline: snapshot images and their simulated times (s), e.g. 0.9=snap_1.png.")
    parser.add_argument("--slices", default="10,90",
                        help="slices: percentile edges of the step-time slices (default: 10,90).")
    parser.add_argument("--outcomes", default=None, metavar="FILE.toml",
                        help="scenes: per scene and solver, a short note on the run's outcome (see load_outcomes).")
    parser.add_argument("--windowed", nargs="*", default=[],
                        help="lead: runs compared over their scene's --window instead of their full common span.")
    parser.add_argument("--window", nargs="*", default=[], metavar="SCENE=T0:T1",
                        help="lead: the time window (s) of a scene's --windowed runs, e.g. armadillo-pull=0:3.")
    parser.add_argument("--equal-time", nargs="*", default=[], dest="equal_time",
                        help="scene_setup, simulation: the equal-time production runs (their sweeps per subproblem get a column).")
    parser.add_argument("--notes", default=None, metavar="FILE.toml",
                        help="scene_setup, simulation: per scene, footnotes on its cells (see load_setup_notes).")
    parser.add_argument("--newton-config", nargs="*", default=[], dest="newton_config", metavar="SCENE=FILE.toml",
                        help="scene_setup, simulation: a scene config whose Newton budget stands in for a Newton run that hasn't "
                        "finished (marked in the table).")
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
    if args.kind in ("scene_setup", "simulation"):
        materials, cg_params, checked = {}, {}, {}
        for path, run in zip(args.kernels, kernels):
            key = scene_key(run)
            scene = str(run["metadata"].get("scene") or "")
            if key not in materials and scene.endswith(".h5"):  # a scene file, not a Dump's "path:group"
                materials[key] = scene_material(scene)
            simulated = run_material(os.path.splitext(path)[0] + ".h5")  # what the run simulated: its Lame params
            if materials.get(key) is not None and simulated is not None:
                check_material(key, materials[key], simulated)
                checked[key] = checked.get(key, 0) + 1
            if run["metadata"].get("solver") == "Newton":
                cg_params[key] = newton_cg_params(os.path.splitext(path)[0] + ".h5")
        print("material checked against the runs' Lame parameters: " + ", ".join(
            f"{k} {checked.get(k, 0)} runs" for k, v in materials.items() if v is not None))
        configs = {k: config_newton_budget(v) for k, v in _parse_labels(args.newton_config).items()}
        rows = setup_rows(kernels, [results.load(p) for p in args.equal_time], materials, cg_params, configs)
        notes = load_setup_notes(args.notes) if args.notes else None
        latex, markdown, plot = ((latex_scene_setup, markdown_scene_setup, plot_scene_setup)
                                 if args.kind == "scene_setup"
                                 else (latex_simulation, markdown_simulation, plot_simulation))
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("% Needs \\usepackage{booktabs}\n\n" + latex(rows, labels, notes))
        print(f"Wrote {args.output}")
        # Readable copies next to it, without LaTeX: Markdown and a PDF drawn by Matplotlib
        stem = os.path.splitext(args.output)[0]
        with open(stem + ".md", "w", encoding="utf-8") as f:
            f.write(markdown(rows, labels, notes))
        use_paper_fonts(args.font_dir)
        fig = plot(rows, labels, notes)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        print(f"Wrote {stem}.md, {stem}.pdf")
        return
    if args.kind in ("scenes", "slices"):
        if args.kind == "scenes":
            outcomes = load_outcomes(args.outcomes) if args.outcomes else None
            text = "% Needs \\usepackage{booktabs,multirow}\n\n" + latex_scenes(kernels, labels, outcomes)
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
                            args.smooth, args.rms_speed, args.vbd_ratio)
    elif args.kind == "solvers":
        fig = plot_solvers(kernels, production_runs, args.width, args.step_times)
    elif args.kind == "sweeps":
        fig = plot_sweeps(kernels, labels, args.width)
    elif args.kind == "lead":
        windows = {}
        for item in args.window:
            scene, sep, span = item.rpartition("=")
            t0, sep2, t1 = span.partition(":")
            if not sep or not sep2:
                raise SystemExit(f"--window expects SCENE=T0:T1, got {item!r}")
            windows[scene] = (float(t0), float(t1))
        points = lead_points(kernels, [results.load(p) for p in args.windowed], windows)
        print(lead_text(points, labels))
        fig = plot_lead(points, labels, args.width)
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
                                    "smoothed_ratio": 0.2 if t % 4 == 0 else 1.5,
                                    "max_outer": 5, "max_sweeps": 100}]},
                "rms_speed": 0.1 * t,
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
        basket = self._production(k)
        # Smoothed ratios 0.1, 0.2, ..., 3.9, then a blow-up's NaN; the decisive ratios (0.5, 3.0) don't count
        for i, step in enumerate(basket["steps"]):
            step["status"] = {"vbd": [{**step["status"]["vbd"][0],
                                       "smoothed_ratio": round(0.1 * (i + 1), 1) if i < 39 else float("nan")}]}
        self.assertEqual(residual_summary(basket), {"median": 2.0, "min": 0.1, "max": 3.9})
        self.assertIsNone(residual_summary(newton))  # recorded without smoothing
        text = latex_scenes([basket, newton, capsules], {"basket_of_fruit": "Basket of fruit"})
        self.assertIn(r"\multirow[t]{4}{2.6cm}{\raggedright Basket of fruit} & 54.5k & 231k & 10 & -- & 13.4 (15.6) "
                      r"& -- & -- & 13.4 (15.6) \\", text)
        self.assertIn(r" &  &  &  & -- & 100, 25\% & -- & -- & 6, 50\% \\", text)
        self.assertIn(r" &  &  &  & -- & 2 [0.1, 3.9] & -- & -- & -- \\", text)  # smoothed ratio, median [min, max]
        self.assertIn(r" &  &  &  & -- & 0.1 (0) & -- & -- & 0 (0) \\", text)  # max penetration ratio (solves > 1)
        self.assertIn(r"\multirow[t]{4}{2.6cm}{\raggedright capsule\_bowl} & 54.5k & 231k & 10 & 13.4 (15.6) & -- & "
                      r"-- & -- & -- \\", text)
        # Lines in order: time, iterations, residual, contact
        self.assertLess(text.index("2 [0.1, 3.9]"), text.index(" & 0.1 (0) & "))
        self.assertLess(text.index("100, 25"), text.index("2 [0.1, 3.9]"))
        self.assertNotIn(r"\parbox", text)  # no notes and no blow-ups: no outcome line
        self.assertNotIn(r"\;/\; outcome", text)
        self.assertEqual((_compact(3269), _compact(1809486)), ("3269", "1.81M"))

    def test_blowup_marks(self):
        k = self._kernels()
        ok = self._production(k)
        for step in ok["steps"]:
            step.update(peak_speed=5.0, rms_speed=1.0)
        self.assertEqual(run_end(ok), 0.5)  # steps 10-49 at 10 ms
        self.assertIsNone(blowup_time(ok))
        self.assertIsNone(automatic_outcome(ok, 0.5))
        far = self._production(k)
        for step in far["steps"]:
            step.update(peak_speed=5.0 if step["t"] < 30 else 2 * BLOWUP_SPEED, rms_speed=1.0)
        self.assertAlmostEqual(blowup_time(far), 0.31)
        self.assertEqual(automatic_outcome(far, 0.5), "blew up at t = 0.31 s")
        nan = self._production(k)
        nan["steps"] = [{**s, "status": {"vbd": [{**s["status"]["vbd"][0], "gradient_ratio": None}]}} if s["t"] >= 40
                        else s for s in nan["steps"]]
        self.assertAlmostEqual(blowup_time(nan), 0.41)  # a non-finite residual: non-finite positions
        stopped = {**far, "steps": far["steps"][:25]}  # a watchdog's stop at step 34, after the blow-up at 30
        self.assertEqual(automatic_outcome(stopped, 0.5), "blew up at t = 0.31 s")  # where its statistics stop
        stopped = {**ok, "steps": ok["steps"][:25]}  # stopped without blowing up
        self.assertEqual(automatic_outcome(stopped, 0.5), "stopped at t = 0.35 s")
        self.assertIsNone(automatic_outcome(ok, 0.505))  # within half a step of the scene's end

    def test_statistics_stop_at_blowup(self):
        run = self._production(self._kernels())
        steps = []
        for i, step in enumerate(run["steps"]):  # steps 10-49 at 10 ms
            blown = step["t"] >= 30  # a node at 2 km/s from step 30: blown up, its ratios huge but finite
            row = {**step["status"]["vbd"][0], "smoothed_ratio": 1e16 if blown else 0.5 + 0.1 * (i % 3),
                   "feasibility_ratio": 900.0 if blown else 0.6 * (i % 3)}
            steps.append({**step, "status": {"vbd": [row]}, "peak_speed": 2 * BLOWUP_SPEED if blown else 5.0})
        run["steps"] = steps
        self.assertAlmostEqual(blowup_time(run), 0.31)
        self.assertEqual([s["t"] for s in before_blowup(run)["steps"]], list(range(10, 30)))  # step 30 left out too
        self.assertEqual(residual_summary(run)["max"], 1e16)  # uncut
        cell = scene_cell(run)
        self.assertEqual(cell["residual"], {"median": 0.6, "min": 0.5, "max": 0.7})
        self.assertEqual(cell["contact"], {"max": 1.2, "unmet": 6})  # 0.6 (i % 3) over steps 10-29: 1.2 six times
        self.assertEqual(scene_cell(run)["iterations"], 100.0)  # the other lines keep every step
        # A run stopped by a watchdog without blowing up keeps all its steps
        stopped = self._production(self._kernels())
        stopped["steps"] = stopped["steps"][:25]
        self.assertIs(before_blowup(stopped), stopped)
        self.assertEqual(scene_cell(stopped)["contact"], {"max": 0.1, "unmet": 0})

    def test_outcomes(self):
        import tempfile

        k = self._kernels()
        aaavbd, vbd, anderson = (self._production(self._kernels(s)) for s in ("AAAVBD", "VBD", "AndersonSolver"))
        anderson["steps"] = anderson["steps"][:20]  # stopped at 0.3 s
        capsules = self._production(self._kernels("VBD", scene="path/to/capsule_bowl.h5"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "outcomes.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write('[basket_of_fruit]\nAAAVBD = "all 9 stay"\nVBD = "loses 9 of 9 & more"\n'
                        'Anderson = ""\n\n'  # a placeholder: the automatic mark shows
                        '[capsule_bowl.30Hz]\nVBD = "21 stuck"\n[capsule_bowl.100Hz]\nVBD = "funnel to −2.6 m"\n')
            outcomes = load_outcomes(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write('[celebration]\nVDB = "typo"\n')
            with self.assertWarns(UserWarning):
                load_outcomes(path)
        self.assertEqual(set(outcomes), {"basket_of_fruit", "capsule_bowl.30Hz", "capsule_bowl.100Hz"})
        self.assertEqual(frame_rate(capsules), "100Hz")
        self.assertEqual(outcome_note(outcomes, aaavbd), "all 9 stay")
        self.assertIsNone(outcome_note(outcomes, anderson))  # by display name, empty
        self.assertEqual(outcome_note(outcomes, capsules), "funnel to −2.6 m")  # the 100 Hz table, not the 30 Hz
        text = latex_scenes([aaavbd, vbd, anderson, capsules], {}, outcomes)
        self.assertIn(r"\multirow[t]{5}{2.6cm}{\raggedright basket\_of\_fruit}", text)
        self.assertIn(r" &  &  &  & \parbox[t]{1.9cm}{\centering\scriptsize loses 9 of 9 \& more} & "
                      r"\parbox[t]{1.9cm}{\centering\scriptsize all 9 stay} & -- & "
                      r"\parbox[t]{1.9cm}{\centering\scriptsize stopped at t = 0.3 s} & -- \\", text)
        self.assertIn(r"\parbox[t]{1.9cm}{\centering\scriptsize funnel to $-$2.6 m}", text)
        self.assertIn(r"\;/\; outcome", text)
        # Without notes, the automatic marks alone
        text = latex_scenes([aaavbd, vbd, anderson], {})
        self.assertIn(r" &  &  &  &  &  & -- & \parbox[t]{1.9cm}{\centering\scriptsize stopped at t = 0.3 s} & -- \\",
                      text)

    def test_contact_summary(self):
        run = self._production(self._kernels())
        rows = run["steps"][3]["status"]["vbd"]
        rows.append({**rows[0], "substep": 1, "feasibility_ratio": 2.5})  # a second substep, unmet
        run["steps"][7]["status"]["vbd"][0]["feasibility_ratio"] = 1.0  # met: at most 1
        run["steps"][8]["status"]["vbd"][0]["feasibility_ratio"] = 1.5
        run["steps"][9]["status"]["vbd"][0]["feasibility_ratio"] = None  # unknown (after a blow-up)
        self.assertEqual(len(solve_rows(run)), 41)
        self.assertEqual(solve_rows(run)[4]["t"], 13)
        self.assertEqual(contact_summary(run), {"max": 2.5, "unmet": 2})
        stopped = {**run, "steps": run["steps"][:8]}  # stopped by a watchdog: its solves up to the stop
        self.assertEqual(contact_summary(stopped), {"max": 2.5, "unmet": 1})
        for step in run["steps"]:
            del step["status"]
        self.assertIsNone(contact_summary(run))

    @staticmethod
    def _lead_run(solver, dt, ratio, smoothed=None, steps=range(10, 50), substeps=1, scene="path/to/capsule_bowl.h5"):
        """A production run whose every solve records the stationarity ratio ``ratio(t)`` (and ``smoothed(t)``)."""
        def rows(t):
            return [{"substep": s, "outer": 1, "sweeps": 100, "reason": "GRADIENT_UNMET", "gradient_ratio": ratio(t),
                     "no_floor_ratio": ratio(t), "floor_clause_ratio": ratio(t), "feasibility_ratio": 0.0,
                     **({} if smoothed is None else {"smoothed_ratio": smoothed(t)})} for s in range(substeps)]

        return {"source": "production",
                "metadata": {"scene": scene, "solver": solver, "dt": dt, "substeps": substeps},
                "steps": [{"t": t, "step_timeline_ms": 1.0, "peak_speed": 1.0, "status": {"vbd": rows(t)}}
                          for t in steps]}

    def test_lead(self):
        run = self._lead_run("VBD", 0.02, lambda t: float(t), substeps=2, steps=range(3, 6))
        np.testing.assert_allclose(solve_end_times(run), [0.07, 0.08, 0.09, 0.1, 0.11, 0.12])
        self.assertEqual(window_median(run, "decisive", 0.08, 0.11), 4.0)  # solves ending in (0.08, 0.11]: 4, 4, 5
        self.assertIsNone(window_median(run, "smoothed", 0.0, 1.0))  # recorded without smoothing
        self.assertIsNone(window_median(run, "decisive", 0.2, 0.3))
        # Two campaigns of one scene: 10 ms (steps 2-99, 0.02-1 s) and 20 ms (steps 10-39, 0.2-0.8 s); the decisive
        # ratios (first function) don't count, only the smoothed ones (second)
        full = [self._lead_run("AAAVBD", 0.01, lambda t: 2.0, lambda t: 0.5, steps=range(2, 100)),
                self._lead_run("VBD", 0.01, lambda t: 1.0, lambda t: 3.0 if t > 20 else 1e3, steps=range(2, 100)),
                self._lead_run("AAAVBD", 0.02, lambda t: 2.0, lambda t: 0.5, steps=range(10, 40)),
                self._lead_run("VBD", 0.02, lambda t: 1.0, lambda t: 10.0, steps=range(10, 40)),
                self._lead_run("Chebyshev", 0.02, lambda t: 1.0, lambda t: 0.0005, steps=range(10, 30))]
        points = lead_points(full)
        self.assertEqual([(p["solver"], p["h"]) for p in points], [("VBD", 0.01), ("VBD", 0.02), ("Chebyshev", 0.02)])
        p = points[0]
        # The common span: from the latest first step (0.2 s) to the shorter campaign's end (0.8 s), so the 10 ms
        # VBD run's early 1e3 ratios (steps up to 20, ending by 0.21 s) are left out but one
        self.assertEqual((p["source"], p["span"], p["window"], p["blowup"]), ("full", (0.2, 0.8), (0.2, 0.8), False))
        self.assertEqual((p["smoothed"], p["smoothed_reference"]), (6.0, 0.5))
        self.assertNotIn("decisive", p)
        self.assertEqual(points[1]["smoothed"], 20.0)
        # Chebyshev stops at 0.6 s: compared up to its stop; its median 5e-4 is floored, never divided by 0
        np.testing.assert_allclose(points[2]["window"], (0.2, 0.6))
        self.assertEqual(points[2]["smoothed"], RATIO_FLOOR / 0.5)
        # A blow-up (a node at 2 km/s from step 30) cuts the window before it; the ratios after it don't count
        blown = self._lead_run("VBD", 0.02, lambda t: 1.0, lambda t: 5.0 if t < 30 else 1e9, steps=range(10, 40))
        for step in blown["steps"][20:]:
            step["peak_speed"] = 2 * BLOWUP_SPEED
        p = lead_points([full[2], blown])[0]
        np.testing.assert_allclose(p["window"], (0.2, 0.6))
        self.assertEqual((p["blowup"], p["smoothed"]), (True, 10.0))
        # Without smoothed ratios: no point
        unsmoothed = self._lead_run("VBD", 0.02, lambda t: 9.0, steps=range(10, 40))
        self.assertIsNone(lead_points([full[2], unsmoothed])[0]["smoothed"])
        # Windowed runs: over their scene's window, narrowed to what both recorded
        windowed = [self._lead_run("AAAVBD", 0.01, lambda t: 1.0, lambda t: 2.0 if t < 50 else 4.0,
                                   steps=range(0, 100)),
                    self._lead_run("AndersonSolver", 0.01, lambda t: 1.0, lambda t: 1.0, steps=range(42, 60))]
        with self.assertRaises(SystemExit):  # no window for the scene
            lead_points([], windowed)
        p = lead_points([], windowed, {"capsule_bowl": (0.3, 0.55)})[0]
        self.assertEqual((p["source"], p["span"]), ("window", (0.3, 0.55)))
        self.assertAlmostEqual(p["window"][0], 0.42)
        self.assertEqual(p["smoothed"], 0.5)  # AAAVBD's median over (0.42, 0.55]: 2.0 (8 solves) vs 4.0 (5)
        text = lead_text(points, {"capsule_bowl": "Capsules"})
        self.assertIn("| Capsules | full | 0.2-0.8 | 10 | 10 x 1 | 0.5 | VBD | 6x |", text)
        fig = plot_lead(points + lead_points([full[2], blown]) + [p], {})
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                         ["VBD", "Chebyshev", "Anderson", "Full run", "Window", "Until a blow-up", "AAAVBD"])
        # One solid line per solver and source (no dotted decisive/smoothed pairs), and the dashed line at 1
        self.assertEqual({line.get_linestyle() for ax in fig.axes for line in ax.lines}, {"-", "None", "--"})
        plt.close(fig)

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

    def test_snapshot_row_aspects(self):
        import tempfile

        k = self._kernels()
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, shape in (("wide", (60, 240)), ("tall", (240, 120)), ("square", (90, 90))):
                paths.append(os.path.join(d, f"{name}.png"))
                plt.imsave(paths[-1], np.random.default_rng(0).random(shape))
            fig = plot_timeline([k], [self._production(k)], ["A"],
                                snapshots=[(0.15, paths[0]), (0.25, paths[1]), (0.35, paths[2])])
            fig.canvas.draw()
            extents = [ax.images[0].get_window_extent() for ax in fig.axes if ax.images]
            self.assertEqual(len(extents), 3)
            heights = [e.height for e in extents]
            self.assertLess(max(heights) - min(heights), 1.0)  # one height for the row, in pixels
            widths = np.array([e.width for e in extents])
            np.testing.assert_allclose(widths / widths[2], [4.0, 0.5, 1.0], rtol=0.02)  # each image's own aspect
            plt.close(fig)

    def test_setup_table(self):
        import tempfile

        import h5py

        def run(solver, scene, sweeps=100, outer=1, substeps=2, rstol=0.02, newton=None):
            r = self._production(self._kernels(solver, scene=scene))
            params = ({"n_max_iters": newton[0], "newton": {"n_max_iters": newton[1]}} if newton else
                      {"vbd_params": {"n_max_iters": outer, "n_subproblem_max_iters": sweeps}})
            r["metadata"].update(dt=1 / 60, substeps=substeps, bdf_order=1, solver_params=params,
                                 convergence_params={"rstol": rstol, "astol": 5 * rstol, "ftol": 0.5,
                                                     "smoothing_passes": 4, "check_interval": 20})
            return r

        with tempfile.TemporaryDirectory() as d:
            scene = os.path.join(d, "banner.h5")  # a CPU editor scene file: two bodies, 1 and 100 MPa
            with h5py.File(scene, "w") as f:
                for i, young in enumerate((1e6, 1e8)):
                    body = f.create_group(f"fem_tet_elastic_bodies/{i}/tools.vbd.ui.TetrahedralElastodynamicsBody")
                    body["Ye"], body["nue"], body["rhoe"] = np.full(3, young), np.full(3, 0.45), np.full(3, 1000.0)
            material = scene_material(scene)
            self.assertEqual(material, {"E": (1e6, 1e8), "nu": (0.45, 0.45), "rho": (1000.0, 1000.0)})
            # A run's output: per-element Lame parameters (float32) of 1 and 100 MPa, nu = 0.45, agree; 0.3 doesn't
            run_h5 = os.path.join(d, "run.h5")
            young, nu = np.array([1e6, 1e8]), 0.45
            with h5py.File(run_h5, "w") as f:
                f["FemElastoDynamics/lame_mu"] = (young / (2 * (1 + nu))).astype(np.float32)
                f["FemElastoDynamics/lame_lambda"] = (young * nu / ((1 + nu) * (1 - 2 * nu))).astype(np.float32)
            simulated = run_material(run_h5)
            check_material("banner", material, simulated)
            with self.assertRaises(ValueError):
                check_material("banner", {**material, "nu": (0.3, 0.3)}, simulated)
            self.assertIsNone(run_material(os.path.join(d, "missing.h5")))
            self.assertIsNone(scene_material(os.path.join(d, "missing.h5")))
            newton_h5 = os.path.join(d, "newton.h5")
            with h5py.File(newton_h5, "w") as f:
                f.create_group("params/Solver/Newton/LinearSolver").attrs.update(n_max_iters=3000, rtol=1e-6,
                                                                                  atol=1e-10)
            self.assertEqual(newton_cg_params(newton_h5), {"cg": 3000, "cg_rtol": 1e-6, "cg_atol": 1e-10})
            self.assertIsNone(newton_cg_params(os.path.join(d, "missing.h5")))
            config = os.path.join(d, "Mat.60Hz.toml")  # a scene config whose Newton run hasn't finished
            with open(config, "w") as f:
                f.write("[Integration]\nsubsteps = 1\n[Solver.Newton]\nn_max_iters = 5\n"
                        "[Solver.Newton.newton]\nn_max_iters = 1\n[Solver.Newton.LinearSolver]\nn_max_iters = 4000\n"
                        "rtol = 1e-9\natol = 1e-10\n")
            configs = {"mat": config_newton_budget(config)}
            self.assertEqual(configs["mat"], {"substeps": 1, "outer": 5, "inner": 1, "cg": 4000, "cg_rtol": 1e-9,
                                              "cg_atol": 1e-10})
            banner, mat = "path/to/banner.h5", "path/to/mat.h5"
            runs = [run(v, banner) for v in VBD_FAMILY] + [run("Newton", banner, newton=(10, 2))]
            runs += [run(v, mat, sweeps=80, outer=5, substeps=1, rstol=0.08) for v in VBD_FAMILY]
            equal_time = [run(v, banner, sweeps=n) for v, n in zip(EQUAL_TIME_SOLVERS, (120, 110, 105))]
            rows = setup_rows(runs, equal_time, {"banner": scene_material(scene)},
                              {"banner": newton_cg_params(newton_h5)}, configs)
        self.assertEqual([r["scene"] for r in rows], ["banner", "mat"])
        self.assertEqual(rows[0]["budget"], (2, 1, 100))
        self.assertEqual(rows[0]["equal_time"], {"VBD": 120, "Chebyshev": 110, "AndersonSolver": 105})
        self.assertEqual(rows[0]["newton"], {"substeps": 2, "outer": 10, "inner": 2, "cg": 3000, "cg_rtol": 1e-6,
                                             "cg_atol": 1e-10, "from_config": False})
        self.assertTrue(rows[1]["newton"]["from_config"])
        labels = {"banner": "Birthday banner"}
        notes = {"mat": {"tolerances": "4x the shared ones"}, "banner": {"material": "two materials"}}
        # Scene setup: the mesh's size (the scenes table's format), E, nu and rho per scene, its own footnotes
        self.assertEqual((rows[0]["nodes"], rows[0]["tets"]), (54502, 231338))
        text = latex_scene_setup(rows, labels, notes)
        self.assertIn(r"Scene & Nodes & Tets & $E$ & $\nu$ & $\rho$ (kg/m$^3$) \\", text)
        self.assertIn(r"Birthday banner & 54.5k & 231k & 1--100 MPa$^{\mathrm{a}}$ & 0.45 & 1000 \\", text)
        self.assertIn(r"mat & 54.5k & 231k & -- & -- & -- \\", text)
        self.assertIn(r"Poisson's ratio $\nu$", text)
        self.assertIn(r"$^{\mathrm{a}}$two materials.", text)
        self.assertNotIn("4x the shared ones", text)
        # Simulation: h alone, then every group's sub-columns under a spanning header (the budgets, the CG cap and its
        # relative tolerance, the absolute one, the same in every scene, in the caption; the convergence tolerances)
        text = latex_simulation(rows, labels, notes)
        self.assertIn(r"\begin{tabular}{l c ccc ccc ccc cc ccc}", text)
        # Every quantity a macro declared once on top, so the paper renames it in one line
        self.assertIn(r"\providecommand{\SimSubproblems}{m}", text)
        self.assertIn(r"\providecommand{\SimCGCap}{n_\mathrm{CG}}", text)
        self.assertIn(r"\providecommand{\SimRstol}{\epsilon_r}", text)
        self.assertLess(text.index(r"\providecommand{\SimStep}{h}"), text.index(r"\begin{table*}"))
        self.assertIn(r" & & \multicolumn{3}{c}{VBD family} & \multicolumn{3}{c}{Equal time, $\SimSweeps$} & "
                      r"\multicolumn{3}{c}{Newton} & \multicolumn{2}{c}{CG} & \multicolumn{3}{c}{Convergence tolerances}",
                      text)
        self.assertIn(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-13} "
                      r"\cmidrule(lr){14-16}", text)
        self.assertIn(r"Scene & $\SimStep$ (s) & $\SimSubsteps$ & $\SimSubproblems$ & $\SimSweeps$ & VBD & Cheb. & "
                      r"And. & $\SimSubsteps$ & $\SimSubproblems$ & $\SimNewtonSteps$ & $\SimCGCap$ & $\SimCGRtol$ & "
                      r"$\SimRstol$ & $\SimAstol$ & $\SimFtol$ \\", text)
        self.assertIn(r"\parbox[t]{2.4cm}{\raggedright Birthday banner} & 1/60 & 2 & 1 & 100 & 120 & 110 & 105 & 2 & 10 "
                      r"& 2 & 3000 & $10^{-6}$ & 0.02 & 0.1 & 0.5 \\", text)
        # A note on the cell it qualifies: Newton's config mark on its outer iterations m
        self.assertIn(r"\parbox[t]{2.4cm}{\raggedright mat} & 1/60 & 1 & 5 & 80 & -- & -- & -- & 1 & 5$^{\mathrm{a}}$ "
                      r"& 1 & 4000 & $10^{-9}$ & 0.08$^{\mathrm{b}}$ & 0.4 & 0.5 \\", text)
        self.assertIn(r"each solve's time step is $\SimStep/\SimSubsteps$", text)
        self.assertIn(r"The budget per frame, $\SimSubsteps \times \SimSubproblems \times \SimSweeps$", text)
        self.assertIn(r"$\max(\SimCGAtol, \SimCGRtol \lVert g \rVert)$", text)
        self.assertIn(r"and $\SimCGAtol = 10^{-10}$ in every scene", text)
        self.assertIn(r"$^{\mathrm{a}}$from its scene config: the run hadn't finished.", text)
        self.assertIn(r"$^{\mathrm{b}}$4x the shared ones.", text)
        self.assertNotIn("two materials", text)
        self.assertIn("4 smoothing passes, a convergence check every 20 sweeps in every scene", text)
        # The readable copies: the same cells in Markdown (group-prefixed names, Unicode marks and powers) and drawn by
        # Matplotlib (spanning headers)
        md = markdown_simulation(rows, labels, notes)
        self.assertIn("| Scene | h (s) | VBD family: s | VBD family: m | VBD family: n | Equal time, n: VBD | Equal "
                      "time, n: Cheb. | Equal time, n: And. | Newton: s | Newton: m | Newton: k | CG: n_CG | CG: r_CG | "
                      "Convergence: ε_r | Convergence: ε_a | Convergence: ε_f |", md)
        self.assertIn("budget per frame, s × m × n:", md)
        self.assertIn("| Birthday banner | 1/60 | 2 | 1 | 100 | 120 | 110 | 105 | 2 | 10 | 2 | 3000 | 10⁻⁶ | 0.02 | 0.1 "
                      "| 0.5 |", md)
        self.assertIn("| mat | 1/60 | 1 | 5 | 80 | – | – | – | 1 | 5ᵃ | 1 | 4000 | 10⁻⁹ | 0.08ᵇ | 0.4 | 0.5 |", md)
        self.assertIn("a_CG = 10⁻¹⁰ in every scene", md)
        self.assertIn("- ᵃ from its scene config: the run hadn't finished", md)
        self.assertIn("- ᵇ 4x the shared ones", md)
        self.assertIn("| Birthday banner | 54.5k | 231k | 1–100 MPaᵃ | 0.45 | 1000 |",
                      markdown_scene_setup(rows, labels, notes))
        setup = plot_scene_setup(rows, labels, notes).axes[0].tables[0]
        self.assertEqual([setup[0, c].get_text().get_text() for c in (3, 4)], ["$E$", r"$\nu$"])
        plt.close("all")
        fig = plot_simulation(rows, labels, notes)
        table = fig.axes[0].tables[0]
        self.assertEqual(table[0, 14].get_text().get_text(), "Convergence\ntolerances")  # spanning, wrapped to the group
        self.assertEqual((table[0, 13].visible_edges, table[0, 14].visible_edges, table[0, 15].visible_edges),
                         ("TBL", "TB", "TBR"))
        self.assertEqual([table[0, c].get_text().get_text() for c in (3, 6, 9, 11)],
                         ["VBD family", "Equal time, $n$", "Newton", "CG"])
        self.assertEqual(table[0, 1].visible_edges, "")
        self.assertEqual([table[1, c].get_text().get_text() for c in range(2, 5)], ["$s$", "$m$", "$n$"])
        self.assertEqual(table[2, 0].get_text().get_text(), "Birthday banner")
        self.assertEqual(table[3, 9].get_text().get_text(), r"5$^{\mathrm{a}}$")
        self.assertEqual(table[3, 12].get_text().get_text(), "$10^{-9}$")
        self.assertEqual(table[3, 13].get_text().get_text(), r"0.08$^{\mathrm{b}}$")
        heights = {(r, round(c.get_height(), 6)) for (r, _), c in table.get_celld().items()}
        self.assertEqual(len(heights), len({r for r, _ in heights}))  # one height per row: the cells line up
        with tempfile.TemporaryDirectory() as d:
            for name, f in (("simulation", fig), ("scene_setup", plot_scene_setup(rows, labels, notes))):
                f.savefig(os.path.join(d, f"{name}.pdf"))
                self.assertGreater(os.path.getsize(os.path.join(d, f"{name}.pdf")), 1000)
                plt.close(f)
        # Different absolute tolerances go into the cells, after the relative one
        rows[1]["newton"]["cg_atol"] = 1e-8
        text = latex_simulation(rows, {})
        self.assertIn(r"& 4000 & $10^{-9}$, $10^{-8}$ &", text)
        self.assertIn(r"with the absolute tolerance $\SimCGAtol$ after $\SimCGRtol$", text)
        # Runs of the VBD family with different budgets aren't one equal-sweep budget
        with self.assertRaisesRegex(ValueError, "equal-sweep budgets"):
            setup_rows(runs[:3] + [run("AndersonSolver", banner, sweeps=90)])
        with self.assertRaisesRegex(ValueError, "convergence tolerances"):
            setup_rows(runs[:3] + [run("AndersonSolver", banner, rstol=0.04)])

    def test_spike_markers(self):
        self.assertEqual([_spike_label(v) for v in (14.6, 999.0, 25601.0, 1.2e6)], ["14.6", "999", "2.6e4", "1.2e6"])
        k = self._kernels()  # steps 10-49 at 10 ms, smoothed ratios 0.2 and 1.5
        aaavbd = self._production(k)
        # One step far above the y range (ArmadilloPull's release), one inside it
        for s in aaavbd["steps"]:
            ratio = {20: 25601.0, 40: 18.0}.get(s["t"])
            if ratio is not None:
                s["status"] = {"vbd": [{**s["status"]["vbd"][0], "smoothed_ratio": ratio}]}
        vbd = self._production(self._kernels("VBD"))  # no spike; 100x higher, so its curve sets the top of the range
        for s in vbd["steps"]:
            row = s["status"]["vbd"][0]
            s["status"] = {"vbd": [{**row, "smoothed_ratio": 100.0 * row["smoothed_ratio"]}]}
        # Near convergence: 100x its median of ~0.015, but below SPIKE_MIN_RATIO, so not marked
        cheb = self._production(self._kernels("Chebyshev"))
        for s in cheb["steps"]:
            row = s["status"]["vbd"][0]
            s["status"] = {"vbd": [{**row, "smoothed_ratio": 5.0 if s["t"] == 30 else 0.01 * row["smoothed_ratio"]}]}
        fig = plot_timeline([k], [aaavbd], ["A"], others=[[vbd, cheb]])
        strip = next(ax for ax in fig.axes if ax.get_ylabel() == "Stationarity\nratio")
        lo, hi = strip.get_ylim()
        self.assertAlmostEqual(hi, 150.0 * 1.25)  # the range is the median curves', not the spikes'
        markers = [line for line in strip.lines if line.get_marker() == "o" and len(line.get_xdata())]
        self.assertEqual(sorted((round(float(x), 2), float(y)) for m in markers
                                for x, y in zip(m.get_xdata(), m.get_ydata())), [(0.2, hi), (0.4, 18.0)])
        self.assertTrue(all(m.get_color() == solver_color(aaavbd) for m in markers))
        self.assertEqual([t.get_text() for t in strip.texts], ["2.6e4"])  # only the clipped one is labeled
        plt.close(fig)

    def test_divergence_cut(self):
        def run(ratio_of, speed_of, solver="AndersonSolver"):
            r = self._production(self._kernels(solver))  # steps 10-49 at 10 ms
            r["steps"] = [{**s, "peak_speed": speed_of(s["t"]), "status": {"vbd": [
                {**s["status"]["vbd"][0], "smoothed_ratio": ratio_of(s["t"])}]}} for s in r["steps"]]
            return r

        # Diverging from step 31 (1e4, ~1e4 times its earlier ~1-2), blown up at step 40 (2 km/s)
        diverging = run(lambda t: 1e4 * (t - 30) if t > 30 else 1.0 + t % 2, lambda t: 2e3 if t >= 40 else 5.0)
        self.assertAlmostEqual(blowup_time(diverging), 0.41)
        self.assertAlmostEqual(divergence_time(diverging), 0.31)
        table = diverged_table(diverging)
        self.assertAlmostEqual(table["diverged"], 0.31)
        self.assertTrue(np.isfinite(table[STRIP_RATIO][table["t"] < 31]).all())
        self.assertTrue(np.isnan(table[STRIP_RATIO][table["t"] >= 31]).all())
        self.assertTrue(np.isnan(table["feasibility_ratio"][table["t"] >= 31]).all())
        self.assertEqual(table["cut_at"][STRIP_RATIO], 1.0)  # step 30's
        self.assertEqual(table["time_ms"].size, 40)  # step times aren't cut
        # A one-solve spike isn't a blow-up: nothing is cut (ArmadilloPull.60Hz AAAVBD's release)
        spike = run(lambda t: 2e3 if t == 20 else 0.2, lambda t: 50.0, "AAAVBD")
        self.assertIsNone(divergence_time(spike))
        table = diverged_table(spike)
        self.assertIsNone(table["diverged"])
        self.assertEqual(np.nanmax(table[STRIP_RATIO]), 2e3)
        # A blow-up without a diverging ratio is cut at the blow-up
        sudden = run(lambda t: 1.0, lambda t: 2e3 if t >= 40 else 5.0, "VBD")
        self.assertAlmostEqual(divergence_time(sudden), blowup_time(sudden))
        # The figures mark the cut and name it in the legend
        k = self._kernels("AndersonSolver")
        fig = plot_solvers([k, self._kernels("AAAVBD")], [diverging, spike], step_times=True)
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertIn("Anderson: diverged at 0.31 s", texts)
        self.assertNotIn("AAAVBD: diverged", " ".join(texts))
        crosses = [line for line in fig.axes[1].lines if line.get_marker() == "x"]
        self.assertEqual(len(crosses), 1)
        self.assertEqual(crosses[0].get_xdata()[0], 1.0)
        self.assertLess(fig.axes[1].get_xlim()[1], 1e4)  # the axis comes from the other runs
        plt.close(fig)
        fig = plot_budget([diverging, spike], {})
        self.assertIn("Anderson: diverged at 0.31 s", [t.get_text() for t in fig.legends[0].get_texts()])
        plt.close(fig)
        # The timeline's strip stops the diverging run's curve there, marks it and keeps the others' range
        fig = plot_timeline([k], [diverging], ["A"], others=[[spike]])
        strip = next(ax for ax in fig.axes if ax.get_ylabel() == "Stationarity\nratio")
        self.assertLess(strip.get_ylim()[1], 10.0)  # its 1e4-9e4 left out: the others' curves set the range
        curve = next(line for line in strip.lines if line.get_color() == solver_color(diverging)
                     and line.get_linestyle() == "-")
        x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())
        self.assertTrue(np.isnan(y[x >= 0.31 - 1e-9]).all())
        self.assertTrue(np.isfinite(y[(x > 0.15) & (x < 0.3)]).all())
        crosses = [line for line in strip.lines if line.get_marker() == "x"]
        self.assertEqual([(round(float(c.get_xdata()[0]), 2), c.get_color()) for c in crosses],
                         [(0.31, solver_color(diverging))])
        self.assertIn("Anderson diverged (0.31 s)", [t.get_text() for t in strip.texts])
        plt.close(fig)

    def test_decade_labels_fit(self):
        self.assertEqual([_decade_label(v) for v in (1e-3, 1.0, 1e3, 1e4, 1e6, 1e-4)],
                         ["0.001", "1", "1000", "$10^{4}$", "$10^{6}$", "$10^{-4}$"])
        # A solvers panel's width (about 0.9 in at 7 in): 1 to 10^6 can't label every decade
        fig, ax = plt.subplots(figsize=(1.2, 0.6))
        fig.subplots_adjust(left=0.1, right=0.9)
        _box(ax, 0.0, np.array([0.3, 0.5, 1.0, 5.0, 30.0, 2e5, 1e6]))
        _log_axis(ax, dense=NARROW_LOG_RANGE, fit_labels=True)
        fig.canvas.draw()
        labels = [t for t in ax.xaxis.get_majorticklabels() if t.get_visible() and t.get_text()]
        boxes = sorted((t.get_window_extent() for t in labels), key=lambda b: b.x0)
        gap = TICK_LABEL_GAP * fig.dpi / 72.0
        self.assertGreaterEqual(len(boxes), 2)
        self.assertTrue(all(b.x0 >= a.x1 + gap for a, b in zip(boxes, boxes[1:])))
        exponents = [int(round(np.log10(v))) for v in ax.xaxis.get_majorticklocs()]
        k = exponents[1] - exponents[0]
        self.assertGreater(k, 1)  # thinned
        self.assertIn(0, exponents)  # 1 keeps its label
        self.assertTrue(all(e % k == 0 for e in exponents))
        # The unlabeled decades keep their ticks
        self.assertEqual(sorted(exponents + [int(round(np.log10(v))) for v in ax.xaxis.get_minorticklocs()]),
                         list(range(0, 7)))
        plt.close(fig)
        # A narrow span keeps every decade labeled; without fit_labels nothing changes (timelines, lead)
        for fit in (True, False):
            fig, ax = plt.subplots(figsize=(3.0, 0.6))
            _box(ax, 0.0, np.array([0.5, 1.0, 50.0, 200.0]))
            _log_axis(ax, dense=NARROW_LOG_RANGE, fit_labels=fit)
            fig.canvas.draw()
            self.assertEqual([t.get_text() for t in ax.xaxis.get_majorticklabels() if t.get_text()
                              and ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1]],
                             ["1", "10", "100"])
            plt.close(fig)

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
            # No phase bands, only the production runs' step times; the speed fill falls back to the kernels run's
            self.assertEqual([t.get_text() for t in timeline.legends[0].get_texts()],
                             ["AAAVBD", "VBD", "AAAVBD RMS speed"])
            self.assertEqual(len(timeline.axes[0].collections), 0)  # no stacked areas
            solvers = plot_solvers([self._kernels("VBD"), k[0]], [p[0]], step_times=True)
            # Contact update, residuals and Other left out; no early-exit panel
            self.assertEqual([t.get_text() for t in solvers.legends[0].get_texts()][:3],
                             ["Vertex sweeps", "Contact detection", "Quartiles"])
            self.assertEqual([ax.get_xlabel() for ax in solvers.axes[1:]],
                             [STRIP_RATIO_LABEL, "Penetration ratio", "Sweeps", "Step time (ms)"])
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

    def test_strip_ratio(self):
        run = self._production(self._kernels())
        # The worst smoothed ratio over a step's substeps, not the decisive ratio (0.5 or 3.0)
        run["steps"][0] = {**run["steps"][0], "status": {"vbd": [
            {**run["steps"][0]["status"]["vbd"][0], "smoothed_ratio": 0.4},
            {**run["steps"][0]["status"]["vbd"][0], "substep": 1, "smoothed_ratio": 0.9}]}}
        table = ratio_table(run)
        self.assertTrue(table["smoothed"])
        self.assertFalse(table["decisive_fallback"])
        self.assertEqual(table[STRIP_RATIO][:3].tolist(), [0.9, 1.5, 0.2])
        self.assertEqual(table[TEST_RATIO][1], 3.0)
        # From a detected blow-up on (a node at 2 km/s at step 40), the steps are left out
        for step in run["steps"]:
            step["peak_speed"] = 2 * BLOWUP_SPEED if step["t"] >= 40 else 5.0
        ratio = ratio_table(run)[STRIP_RATIO]
        self.assertTrue(np.isfinite(ratio[:30]).all())
        self.assertTrue(np.isnan(ratio[30:]).all())
        # Without smoothed ratios: the decisive ratio, marked in the legend and row labels
        vbd = self._production(self._kernels("VBD"))
        vbd["steps"] = [{**s, "status": {"vbd": [{k: v for k, v in s["status"]["vbd"][0].items()
                                                   if k != "smoothed_ratio"}]}} for s in vbd["steps"]]
        table = ratio_table(vbd)
        self.assertEqual((table["smoothed"], table["decisive_fallback"]), (False, True))
        np.testing.assert_array_equal(table[STRIP_RATIO], table[TEST_RATIO])
        k = self._kernels()
        fig = plot_timeline([k], [self._production(k)], ["A"], others=[[vbd]])
        self.assertIn(f"VBD{DECISIVE_MARK}", [t.get_text() for t in fig.legends[0].get_texts()])
        self.assertIn("AAAVBD", [t.get_text() for t in fig.legends[0].get_texts()])
        self.assertIn("Stationarity\nratio", [ax.get_ylabel() for ax in fig.axes])
        plt.close(fig)
        fig = plot_budget([self._production(k), vbd], {})
        self.assertEqual([t.get_text() for t in fig.axes[0].get_yticklabels()], [f"VBD{DECISIVE_MARK}", "AAAVBD"])
        self.assertEqual(fig.axes[0].get_xlabel(), STRIP_RATIO_LABEL)
        plt.close(fig)

    def test_vbd_time_ratio(self):
        k = self._kernels()  # steps 10-49 at 10 ms
        aaavbd = self._production(k)
        vbd = self._production(self._kernels("VBD"))
        for step in vbd["steps"]:
            step["step_timeline_ms"] *= 2.0  # VBD twice as slow: AAAVBD's ratio 0.5
        vbd["steps"] = [s for s in vbd["steps"] if s["t"] != 20]  # a step VBD lacks: a gap
        aaavbd["steps"] = [s for s in aaavbd["steps"] if s["t"] != 25]  # one AAAVBD lacks
        x, ratio = vbd_time_ratio(aaavbd, vbd)
        np.testing.assert_allclose(x[:2], [0.1, 0.11])
        self.assertTrue(np.isnan(ratio[[10, 15]]).all())  # t = 20 and 25
        np.testing.assert_allclose(np.delete(ratio, [10, 15]), 0.5)
        # From a detected blow-up of either run on (VBD's node at 2 km/s from step 40), steps are left out
        for step in vbd["steps"]:
            step["peak_speed"] = 2 * BLOWUP_SPEED if step["t"] >= 40 else 5.0
        _, ratio = vbd_time_ratio(aaavbd, vbd)
        self.assertTrue(np.isnan(ratio[30:]).all())  # t = 40 on: the blown-up step and after
        self.assertTrue(np.isfinite(ratio[[0, 9, 29]]).all())
        # With the strip (vbd_ratio) and VBD: the cost plot draws the paired run's step time only, the strip every
        # other run's ratio to VBD
        cheb = self._production(self._kernels("Chebyshev"))
        runs = [cheb, self._production(self._kernels("VBD"))]
        fig = plot_timeline([k], [self._production(k)], ["A"], others=[runs], vbd_ratio=True)
        cost, relative = fig.axes[0], fig.axes[1]
        self.assertEqual(relative.get_ylabel(), "Step time /\nVBD's")
        self.assertEqual([line.get_color() for line in cost.lines], [SOLVER_COLORS["AAAVBD"]])
        self.assertEqual({line.get_color() for line in relative.lines if line.get_linestyle() == "-"},
                         {SOLVER_COLORS["AAAVBD"], SOLVER_COLORS["Chebyshev"]})  # no VBD curve
        self.assertEqual(relative.get_yscale(), "linear")  # every ratio 1: within VBD_RATIO_LOG_SPAN
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()][-4:-1], ["AAAVBD", "Chebyshev", "VBD"])
        plt.close(fig)
        with_strip = fig.get_size_inches()[1]
        plt.close(fig)
        # By default (the strip off, the maintainer's choice): no strip, every run's step time on the cost plot in its
        # solver's color; the cost plot keeps its height, so the figure is shorter by the strip
        fig = plot_timeline([k], [self._production(k)], ["A"], others=[runs])
        self.assertNotIn("Step time /\nVBD's", [ax.get_ylabel() for ax in fig.axes])
        self.assertEqual([line.get_color() for line in fig.axes[0].lines],
                         [SOLVER_COLORS[s] for s in ("AAAVBD", "Chebyshev", "VBD")])
        self.assertLess(fig.get_size_inches()[1], with_strip)
        plt.close(fig)
        # With the strip asked for but no VBD run: no strip, every run's step time on the cost plot
        fig = plot_timeline([k], [self._production(k)], ["A"], others=[[cheb]], vbd_ratio=True)
        self.assertNotIn("Step time /\nVBD's", [ax.get_ylabel() for ax in fig.axes])
        self.assertEqual(len(fig.axes[0].lines), 2)
        plt.close(fig)

    def test_feasibility_strip(self):
        k = self._kernels()  # steps 10-49 at 10 ms, feasibility ratio 0.1
        run = self._production(k)
        run["steps"][0] = {**run["steps"][0], "status": {"vbd": [
            {**run["steps"][0]["status"]["vbd"][0], "feasibility_ratio": 0.3},
            {**run["steps"][0]["status"]["vbd"][0], "substep": 1, "feasibility_ratio": 4.0}]}}
        for step in run["steps"]:
            step["peak_speed"] = 2 * BLOWUP_SPEED if step["t"] >= 40 else 5.0  # blown up from step 40
        feasibility = ratio_table(run)[FEASIBILITY_STRIP]
        self.assertEqual(feasibility[0], 4.0)  # the worst over the step's substeps
        np.testing.assert_allclose(feasibility[1:30], 0.1)
        self.assertTrue(np.isnan(feasibility[30:]).all())  # cut at the blow-up
        self.assertTrue(has_contact(run))
        no_contact = self._production(self._kernels("VBD"))
        no_contact["steps"] = [{**s, "status": {"vbd": [{**s["status"]["vbd"][0], "feasibility_ratio": 0.0}]}}
                               for s in no_contact["steps"]]
        self.assertFalse(has_contact(no_contact))
        # Its own thin strip, last, on a log axis
        vbd = self._production(self._kernels("VBD"))
        fig = plot_timeline([k], [self._production(k)], ["A"], others=[[vbd]])
        labels = [ax.get_ylabel() for ax in fig.axes]
        # the cost plot, the residual and penetration strips, then the cost plot's RMS speed axis
        self.assertEqual(labels, ["GPU time (ms)", "Stationarity\nratio", "Penetration\nratio", f"{SPEED_NAME}\n(m/s)"])
        penetration = fig.axes[2]
        self.assertEqual(penetration.get_yscale(), "log")
        self.assertEqual([line.get_color() for line in penetration.lines if line.get_linestyle() == "-"],
                         [SOLVER_COLORS["AAAVBD"], SOLVER_COLORS["VBD"]])
        self.assertEqual(len(fig.axes[1].collections), 0)  # no speed fill behind the residual strip
        height = fig.get_size_inches()[1]
        plt.close(fig)
        without = self._production(k)
        without["steps"] = [{**s, "status": {"vbd": [{**s["status"]["vbd"][0], "feasibility_ratio": 0.0}]}}
                            for s in without["steps"]]
        fig = plot_timeline([k], [without], ["A"])  # no VBD, no contact: the cost plot and the residual strip
        self.assertNotIn("Penetration\nratio", [ax.get_ylabel() for ax in fig.axes])
        self.assertLess(fig.get_size_inches()[1], height)
        plt.close(fig)

    def test_timeline_rms_speed(self):
        k = self._kernels()
        p = self._production(k)
        vbd = self._production(self._kernels("VBD"))
        for run in (p, vbd):
            for step in run["steps"]:
                step["rms_speed"] = 0.05 * step["t"]
        legends = {}
        for mode in RMS_SPEED_MODES:
            fig = plot_timeline([k], [p], ["A"], others=[[vbd]], rms_speed=mode)
            legends[mode] = [t.get_text() for t in fig.legends[0].get_texts()]
            plt.close(fig)
        # The paired run's speed (a fill) either way; the other run's dotted line and its entry only with "all"
        self.assertEqual([t for t in legends["paired"] if t.endswith(SPEED_NAME)], ["AAAVBD RMS speed"])
        self.assertEqual([t for t in legends["all"] if t.endswith(SPEED_NAME)], ["AAAVBD RMS speed", "VBD RMS speed"])


if __name__ == "__main__":
    main()
