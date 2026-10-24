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
        [--budget-factor 1] [--width 5.5] [--no-legend] [--frames PREFIX] [--dataset DATASET] \
        [--title] [--elev 28] [--azim -62] [--dpi 200]

``--frames PREFIX`` adds ``benchmark_frames.py``'s example problems left of each chart as a grid (:func:`add_frames`):
a homogeneous case colored by the error and the heterogeneous case colored by Lame lambda in columns headed by the
cases' labels, their initial iterates above their reference solutions, a colorbar under each column. ``--columns
[BENCHMARK=]PAIRS`` chooses the columns per chart: by default both on the static chart, the error pair on the
h = 1e-2 chart and the material pair on the h = 1e-4 chart, which the paper stacks (the two single-column grids come
out equally wide, so the stacked charts line up). The grid sits left of the chart, its headers in line with the chart's top (the legend's entries, else the 3D
box's top edge) and its colorbar titles on the baseline of the last material label: the chart is exactly as without
``--frames``, and the figure gets wider (the paper scales it). After a benchmark re-run, regenerate the frames first
(``benchmark_frames.py``). The grid's headers are the cases' labels, as on the chart's material axis.

``--dataset`` gives the dataset when the results file's ``dataset`` attribute doesn't resolve (it is relative to the
main checkout's root, so e.g. from a worktree), for the case labels' moduli: without it, the heterogeneous case is
labeled "Heterogeneous" instead of its set of Young's moduli, as in the other benchmark figures.

Writes ``<prefix>_<benchmark>.pdf`` and ``.png`` per benchmark, and with ``--no-legend`` the legend alone as
``<prefix>_legend.pdf``. ``--width`` renders each chart at the width the paper prints it, so its 8 pt labels
print at 8 pt; at a column's width (about 3.4 in) the 3D box gets too small to read, so the paper stacks the
charts at about 0.8 x the text width.

Fonts: the figure text is in Linux Biolinum, the sans serif of ACM's acmart template (SIGGRAPH, TOG), whose
body text is Linux Libertine; from ``--font-dir``, the ``PBAT_FONT_DIR`` environment variable or the installed
fonts (:func:`use_paper_fonts`; the OpenType release:
https://sourceforge.net/projects/linuxlibertine/files/linuxlibertine/5.3.0/); without it, a warning and
Matplotlib's default font. No grid lines (maintainer).
"""

import argparse
import functools
import glob
import json
import os
import re
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

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
FIGURE_HEIGHT = 4.2  # inches; with BOX_ZOOM, sets the box's scale
BOX_ZOOM = 1.35  # the 3D box's scale in its axes (Matplotlib's default 1 leaves wide margins)
FRAME_PAD = 6 / 72  # inches between the plot, its legend and title, and around the saved region
OTHER_COLORS = ["#56B4E9", "#F0E442", "#000000"]
# The Linux Libertine family of ACM's acmart template (SIGGRAPH, TOG): Linux Libertine (serif, body text), Linux
# Biolinum (sans serif, headings) and Linux Libertine Mono (code). OpenType family names first, then other names.
PAPER_SANS = ("Linux Biolinum O", "Linux Biolinum", "Libertinus Sans")
PAPER_SERIF = ("Linux Libertine O", "Linux Libertine", "Libertinus Serif")
PAPER_MONO = ("Linux Libertine Mono O", "Libertinus Mono")


def use_paper_fonts(font_dir: str | None = None) -> str | None:
    """Sets Matplotlib's fonts to the Linux Libertine family: figure text and math in Linux Biolinum (the sans
    serif), with Linux Libertine as the serif and Linux Libertine Mono as the monospace font. The fonts come from
    ``font_dir``, else the ``PBAT_FONT_DIR`` environment variable's directory, else the installed fonts (also ones
    installed after Matplotlib built its font cache). Returns the sans family used; without it, warns and keeps
    Matplotlib's fonts."""
    from matplotlib import font_manager

    def add(paths):
        for path in paths:
            try:
                font_manager.fontManager.addfont(path)
            except (OSError, RuntimeError, ValueError):
                pass

    for directory in (font_dir, os.environ.get("PBAT_FONT_DIR")):
        if directory:
            add(sorted(glob.glob(os.path.join(directory, "*.otf")) + glob.glob(os.path.join(directory, "*.ttf"))))

    def first(families):
        names = {f.name for f in font_manager.fontManager.ttflist}
        return next((name for name in families if name in names), None)

    if first(PAPER_SANS) is None or first(PAPER_SERIF) is None:
        add(p for p in font_manager.findSystemFonts()
            if any(key in os.path.basename(p).lower() for key in ("libertin", "biolin")))
    sans, serif, mono = first(PAPER_SANS), first(PAPER_SERIF), first(PAPER_MONO)
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed the fonts as TrueType/OpenType, editable text in the PDF
    for key, name in (("font.serif", serif), ("font.monospace", mono)):
        if name is not None:
            matplotlib.rcParams[key] = [name, *matplotlib.rcParams[key]]
    if sans is None:
        warnings.warn("Linux Biolinum not found (--font-dir, PBAT_FONT_DIR or an installed font): using Matplotlib's "
                      "default font")
        return None
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [sans, *matplotlib.rcParams["font.sans-serif"]],
                                "mathtext.fontset": "custom", "mathtext.rm": sans, "mathtext.sf": sans,
                                "mathtext.it": f"{sans}:italic", "mathtext.bf": f"{sans}:bold",
                                "mathtext.bfit": f"{sans}:bold"})  # Biolinum has no bold italic
    return sans


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


def benchmark_header(benchmark: str) -> str:
    """A benchmark's column header for the paper (grid column headers, table captions), with the time step as math: ``Elastodynamics, $h = 10^{-2}$ s``."""
    if benchmark.startswith("dynamic_h"):
        h = float(benchmark[len("dynamic_h"):])
        exponent = np.log10(h)
        value = f"10^{{{int(round(exponent))}}}" if abs(exponent - round(exponent)) < 1e-9 else f"{h:g}"
        return f"Elastodynamics, $h = {value}$ s"
    return benchmark_title(benchmark)


def modulus_math(value: float) -> str:
    """A Young's modulus as math: ``10^{6}``, or ``2.5 \\times 10^{6}``."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = float(f"{value / 10.0 ** exponent:.3g}")
    if mantissa == 10:
        mantissa, exponent = 1.0, exponent + 1
    return f"10^{{{exponent}}}" if mantissa == 1 else rf"{mantissa:g} \times 10^{{{exponent}}}"


def material_label(label: str, moduli: np.ndarray | None = None) -> str:
    """A run file's case label for the paper: Young's modulus as math, ``Y = 1e5`` -> ``$E = 10^{5}$ Pa``. Other
    labels (``heterogeneous``) show the scene's ``moduli`` if known, as a set when there are several
    (``$E \\in \\{10^{6}, 10^{8}\\}$ Pa``), else they are capitalized."""
    m = re.fullmatch(r"Y\s*=\s*([0-9.]+[eE][+-]?\d+)", label.strip())
    if m is not None:
        return f"$E = {modulus_math(float(m.group(1)))}$ Pa"
    if moduli is not None and len(moduli):
        if len(moduli) == 1:
            return f"$E = {modulus_math(moduli[0])}$ Pa"
        return rf"$E \in \{{{', '.join(modulus_math(v) for v in moduli)}\}}$ Pa"
    return label[:1].upper() + label[1:]


@functools.lru_cache
def scene_moduli(dataset: str, scene: str) -> tuple[float, ...] | None:
    """The distinct Young's moduli of a benchmark dataset's scene, from its FEM's Lamé parameters (rows mu,
    lambda per quadrature point): E = mu (3 lambda + 2 mu) / (lambda + mu), to 3 significant digits. None if the
    dataset or the parameters can't be read (e.g. the dataset has moved)."""
    try:
        with h5py.File(dataset, "r") as f:
            lame = np.asarray(f[f"{scene}/fem/pbat.sim.dynamics.FemElastoDynamics/lamegU"], dtype=np.float64)
    except (OSError, KeyError):
        return None
    mu, lam = lame[0], lame[1]
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    return tuple(sorted({float(f"{v:.3g}") for v in E}))


def case_label(group: h5py.Group, case: str, dataset: str | None = None) -> str:
    """A case's label for the paper (:func:`material_label`), its moduli from ``dataset`` (default: the results file's
    ``dataset`` attribute, e.g. a path relative to the main checkout's root)."""
    scene = str(group.attrs.get("scene", case))
    dataset = dataset or group.file.attrs.get("dataset")
    moduli = scene_moduli(str(dataset), scene) if dataset is not None else None
    label = material_label(str(group.attrs.get("label", scene)), moduli)
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
    factor: float | None = None, counts: bool = False, title: bool = False, legend: bool = True, scale: float = 1.0,
    dataset: str | None = None,
):
    """One figure: cases x levels x solvers bars of mean normalized increment, with min/max boxes.
    ``solver_order`` (back to front) overrides :func:`depth_order`. ``factor``: the budget factor the bars
    score at (:func:`budget_factor`); below the run's, the T_N cases are re-scored from their logs. ``counts``
    labels the bars with n/N (see the module docstring). ``title`` draws the benchmark's title above the legend,
    ``legend`` the solver legend. ``scale`` multiplies the figure's size but not its fonts (:func:`plot_at_width`).
    ``dataset``: the dataset file for the case labels (:func:`case_label`).
    Returns the figure and the region to save (:func:`frame`)."""
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
    # Depth long enough that the level labels don't crowd under the camera's foreshortening
    aspect = (max(1.6, 1.1 * len(cases)), max(2.4, 0.45 * n_l), 1.0)
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

    fig = plt.figure(figsize=(scale * max(7.0, 1.6 * len(cases) + 4.0), scale * FIGURE_HEIGHT))
    # The 3D axes fill the figure: a 3D axes draws its box inside its area at a fixed scale. The saved region is
    # cropped to the drawn plot (frame), so the figure's size only sets the box's scale
    ax = fig.add_axes((-0.08, 0.05, 1.0, 0.9), projection="3d", computed_zorder=False)
    ax.view_init(elev=elev, azim=azim)
    # Back to front: position along the direction toward the camera, in displayed (box-aspect) units
    scale = np.array([aspect[0] / (x_range[1] - x_range[0]), aspect[1] / (y_range[1] - y_range[0])])
    def depth(x, y):
        return toward_camera @ (np.array([x, y]) * scale)

    bars.sort(key=lambda b: (depth(b[0], b[1]), not b[6]))
    labels = []
    for order, (x, y, z0, dz, color, label, is_box, w, d) in enumerate(bars):
        if is_box:
            collection = ax.bar3d(x - w / 2, y - d / 2, z0, w, d, max(dz, 1e-9), color=color, alpha=0.18,
                                  edgecolor=color, linewidth=0.3, shade=False, zorder=order)
            if label:
                labels.append((x, y, z0 + dz, label, order))
        else:
            collection = ax.bar3d(x - w / 2, y - d / 2, 0.0, w, d, max(dz, 1e-9), color=color, alpha=1.0,
                                  edgecolor="white", linewidth=0.2, shade=True, zorder=order)
        collection.set_clip_on(False)  # the zoomed box (BOX_ZOOM) may reach past the axes' clip box
    z_max = 1.08 * z_top if z_top > 0 else 1.0
    labels += [(x, y, z, label, len(bars)) for x, y, z, label in floating]
    for x, y, z, label, order in labels:
        ax.text(x, y, z + 0.015 * z_max, label, fontsize=6, ha="center", va="bottom", color="#333333",
                zorder=len(bars) + order)

    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([case_label(g, c, dataset) for c, g in cases], fontsize=8)
    ax.set_yticks(range(n_l))
    ax.set_yticklabels([f"{100 * v:g}" for v in levels], fontsize=8, ha="left", va="center")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(0.0, z_max)
    ax.set_xlabel("Material", labelpad=18)
    ax.set_ylabel(r"$1 - \dfrac{e}{e_0}$ (%)", labelpad=0)
    fixed = [("budget" in g.attrs) for _, g in cases]
    norm = "T" if all(fixed) else "T_N" if not any(fixed) else "T_N (or T)"
    ax.zaxis.set_rotate_label(False)  # upright: the title is short enough to read without tilting one's head
    ax.set_zlabel(rf"$\dfrac{{\Delta t}}{{{norm}}}$", labelpad=-2, rotation=0)
    ax.tick_params(axis="z", labelsize=8, pad=0)
    ax.tick_params(axis="y", pad=0)  # tick labels and titles close to their axes (maintainer)
    ax.set_box_aspect(aspect, zoom=BOX_ZOOM)
    ax.grid(False)  # the grid lines carry nothing the axes and bars don't (maintainer)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
    return fig, frame(fig, ax, benchmark_title(benchmark) if title else None, legend_handles(solvers) if legend else None)


def legend_handles(solvers: list[str]) -> list[Patch]:
    """The solver legend's entries, in ``solvers``' order, then the min-max box."""
    handles = [Patch(color=solver_color(s, solvers), label=SOLVER_NAMES.get(s, s)) for s in solvers]
    handles.append(Patch(facecolor=(0.5, 0.5, 0.5, 0.18), edgecolor="grey", label="min-max"))
    return handles


def legend_figure(solvers: list[str]) -> tuple:
    """The solver legend alone, one row, for a figure that stacks several charts drawn without theirs
    (``--no-legend``). Returns the figure and the region to save (inches)."""
    fig = plt.figure(figsize=(8.0, 0.5))
    legend = fig.legend(handles=legend_handles(solvers), loc="center", ncol=len(solvers) + 1, fontsize=8,
                        frameon=False, borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    region = legend.get_window_extent(fig.canvas.get_renderer()).padded(FRAME_PAD * fig.dpi)
    return fig, region.transformed(fig.dpi_scale_trans.inverted())


def plot_at_width(width: float | None, *args, **kwargs) -> tuple:
    """:func:`plot_benchmark` with its saved region ``width`` inches wide (None: the natural size): the figure is
    scaled and its fonts aren't, so the labels print at their nominal size where the paper places the chart at
    that width. The region doesn't scale exactly with the figure (fonts and paddings), so a few redraws converge."""
    scale = 1.0
    for attempt in range(6):
        fig, region = plot_benchmark(*args, **kwargs, scale=scale)
        if width is None or abs(region.width - width) <= 0.01 or attempt == 5:
            break  # the last redraw is kept even if not within 0.01 in: the returned figure must stay open
        plt.close(fig)
        scale *= width / region.width
    return fig, region


def frame(fig, ax, title: str | None, handles: list | None) -> Bbox:
    """Center a one-row legend (if ``handles``), and the title if any, above the drawn plot and return the region
    to save (inches). By default the figures have no title and no caption: the paper's LaTeX figure carries them, as
    sub-figure labels and a caption (maintainer). A 3D axes fills its area and draws its box inside it at a fixed
    scale, so a tight bounding box would keep the whole area's margins: the plot's extent is measured from the
    box's panes, its axes (ticks and titles) and its texts."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = (ax.xaxis, ax.yaxis, ax.zaxis)
    parts = [a.get_tightbbox(renderer) for a in axes] + [a.pane.get_window_extent(renderer) for a in axes]
    parts += [t.get_window_extent(renderer) for t in ax.texts]
    plot = Bbox.union([b for b in parts if b is not None and b.width > 0])
    pad = FRAME_PAD * fig.dpi  # pixels
    to_figure = fig.transFigure.inverted()
    cx = to_figure.transform(((plot.x0 + plot.x1) / 2, 0))[0]
    shown = [plot]
    if handles:
        legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False,
                            bbox_to_anchor=to_figure.transform(((plot.x0 + plot.x1) / 2, plot.y1 + pad)),
                            borderaxespad=0, borderpad=0)
        shown.append(legend.get_window_extent(renderer))
    if title:
        suptitle = fig.suptitle(title, x=cx, y=to_figure.transform((0, shown[-1].y1 + pad))[1], ha="center",
                                va="bottom", fontsize=11)
        shown.append(suptitle.get_window_extent(renderer))
    return Bbox.union(shown).padded(pad).transformed(fig.dpi_scale_trans.inverted())


# Frames (--frames): a 2 x 2 grid of example problems left of the chart, as tall as its saved region
FRAMES_ROWS = {"initial": "Initial", "reference": "Solution"}  # the row labels: the initial iterate, the reference solution
FRAMES_COLUMNS = ("error", "material")  # benchmark_frames.py's pairs, left to right
# --columns' defaults: both pairs on the static chart; one each on the dynamic charts, which the paper stacks (h = 1e-2
# above h = 1e-4), so each shows one example and the stacked charts line up
FRAMES_COLUMNS_DEFAULT = ["error,material", "dynamic_h0.01=error", "dynamic_h0.0001=material"]
FRAMES_TEXT_SIZE = 7  # pt: the grid's text when the chart has none to match (see _chart_text_sizes)
FRAMES_GAP = 2 / 72  # inches between the grid's labels, frames and colorbars
FRAMES_COLUMN_GAP = 0.06  # inches between the grid's columns, at least
FRAMES_MARGIN = 0.08  # inches between the grid and the chart's region
FRAMES_BAR_HEIGHT = 0.055  # inches


def load_frames(prefix: str, benchmark: str) -> dict:
    """``benchmark_frames.py``'s frames of a benchmark (``<prefix>_<benchmark>.json``), with each pair's images
    loaded as arrays."""
    with open(f"{prefix}_{benchmark}.json", encoding="utf-8") as f:
        spec = json.load(f)
    directory = os.path.dirname(os.path.abspath(prefix))
    for pair in spec.values():
        for state in FRAMES_ROWS:
            pair[f"{state}_image"] = plt.imread(os.path.join(directory, pair[state]))
    return spec


def _length_text(meters: float) -> str:
    """A length in m, or in mm below 1 cm: ``0.218 m``, ``1.43 mm``."""
    return f"{meters:.3g} m" if meters >= 0.01 else f"{1e3 * meters:.3g} mm"


def _frames_colorbar(cax, pair: dict, tick_size: float = FRAMES_TEXT_SIZE) -> str:
    """A pair's colorbar from its rendered shades, with its quantity's ticks (labels at ``tick_size`` pt). Returns its
    title (the caller places it)."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    shades = np.asarray(pair["shades"])
    cax.imshow(shades[None], aspect="auto", extent=(pair["vmin"], pair["vmax"], 0, 1), interpolation="bilinear")
    cax.set_yticks([])
    for side in ("top", "right", "left"):
        cax.spines[side].set_visible(False)
    cax.spines["bottom"].set_linewidth(0.5)
    cax.tick_params(axis="x", labelsize=tick_size, length=2, width=0.5, pad=1)
    if pair["quantity"] == "lambda":
        # log10(lambda + 1) axis: decade ticks, the outer decades labeled
        decades = [k for k in range(int(np.ceil(pair["vmin"])), int(np.floor(pair["vmax"])) + 1)]
        at = [np.log10(10.0**k + 1) for k in decades]
        cax.xaxis.set_major_locator(FixedLocator([at[0], at[-1]] if len(at) > 1 else at))
        cax.set_xticklabels([f"$10^{{{decades[0]}}}$", f"$10^{{{decades[-1]}}}$"] if len(at) > 1
                            else [f"$10^{{{decades[0]}}}$"])
        cax.xaxis.set_minor_locator(FixedLocator(at))
        cax.xaxis.set_minor_formatter(NullFormatter())
        cax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
        title = r"$\lambda$ (Pa)"
    else:
        cax.set_xticks([pair["vmin"], pair["vmax"]], ["0", _length_text(pair["vmax"])])
        labels = cax.get_xticklabels()
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
        title = "Error"  # |x - x*|, in the tick labels' unit
    cax.set_xlim(pair["vmin"], pair["vmax"])
    return title


def _text_size(fig, text: str, fontsize: float, rotation: float = 0.0) -> tuple[float, float]:
    """A text's extent (inches) at ``fontsize`` pt, as :func:`add_frames` draws it."""
    t = fig.text(0, 0, text, fontsize=fontsize, rotation=rotation, ha="center", va="center",
                 multialignment="center")
    box = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return box.width / fig.dpi, box.height / fig.dpi


def _wrapped(text: str) -> str:
    """``text`` on two lines, broken at its middle space (unchanged without one)."""
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return text
    i = min(spaces, key=lambda k: abs(k - len(text) / 2))
    return text[:i] + "\n" + text[i + 1:]


def _baseline(text, renderer) -> float:
    """A drawn horizontal text's baseline (display pixels)."""
    return text.get_window_extent(renderer).y0 - text._get_layout(renderer)[0].y0


def _chart_text_sizes(chart) -> tuple[float, float]:
    """The chart's text sizes (pt) for the grid: its material axis' labels (its last x tick label), for the headers,
    row labels and colorbar titles, and its numbers (its first y tick label), for the colorbars' ticks; each
    :data:`FRAMES_TEXT_SIZE` without such a label."""
    def size(labels):
        labels = [t for t in labels if t.get_visible() and t.get_text()]
        return float(labels[0].get_fontsize()) if labels else float(FRAMES_TEXT_SIZE)

    return size(chart.get_xticklabels()[::-1]), size(chart.get_yticklabels())


def _frames_anchors(fig, region: Bbox) -> tuple[str, float, float]:
    """The grid's vertical anchors (inches, figure coordinates) from the drawn chart: the column headers' (alignment,
    y), on the legend entries' baseline, or without a legend with their tops at the plot's top (``region`` less
    :data:`FRAME_PAD`: the 3D box's top edge); and the colorbar titles' baseline, the chart's last x tick label's (the
    material axis' last case), or without one the plot's bottom."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if fig.legends:
        header = ("baseline", float(np.mean([_baseline(t, renderer) for t in fig.legends[0].get_texts()])) / fig.dpi)
    else:
        header = ("top", region.y1 - FRAME_PAD)
    ticks = [t for t in fig.axes[0].get_xticklabels() if t.get_visible() and t.get_text()]
    bottom = _baseline(ticks[-1], renderer) / fig.dpi if ticks else region.y0 + FRAME_PAD
    return header[0], header[1], bottom


def add_frames(fig, region: Bbox, frames: dict, headers: dict, columns: tuple[str, ...] = FRAMES_COLUMNS) -> Bbox:
    """:func:`load_frames`' pairs as a grid left of a chart's saved ``region`` (inches), as tall as it: ``columns`` (of
    :data:`FRAMES_COLUMNS`, left to right) under ``headers`` (e.g. the cases' labels), rows the initial iterate and the
    reference
    solution, labeled at the left (rotated, on two lines if longer than the row), and each column's colorbar at the
    bottom. The headers line up with the chart's top and the colorbars' titles with its last material label
    (:func:`_frames_anchors`, the chart being ``fig.axes[0]``), the texts at the chart's sizes
    (:func:`_chart_text_sizes`); the rows share the height between them, the frames as
    tall as their rows, their width following from their aspect. The chart isn't touched. Returns the region to save,
    widened to the grid. The rows' height is measured with every pair's header and colorbar (those of ``frames``
    and ``headers``), shown or not, so grids with as many columns beside equally tall charts come out equally wide,
    whichever columns they show (the frames' width follows from the height), unless a header is wider than the
    frames."""
    fig_w, fig_h = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    align, header_y, title_y = _frames_anchors(fig, region)
    label_size, tick_size = _chart_text_sizes(fig.axes[0])
    inches = lambda artist: artist.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())  # noqa: E731
    # Every pair's texts and colorbar at their heights first (their extents set the rows'), placed along x once the
    # frames' width is known; the columns not shown are removed after measuring
    measured = [c for c in FRAMES_COLUMNS if c in frames and c in headers]
    header_texts = {c: fig.text(0, header_y / fig_h, headers[c], fontsize=label_size, ha="center", va=align,
                                multialignment="center") for c in measured}
    bars, title_texts = {}, {}
    for c in measured:
        bars[c] = fig.add_axes((0, 0, 1 / fig_w, FRAMES_BAR_HEIGHT / fig_h))
        title_texts[c] = fig.text(0, title_y / fig_h, _frames_colorbar(bars[c], frames[c], tick_size),
                                  fontsize=label_size, ha="center", va="baseline")
    fig.canvas.draw()
    title_top = max(inches(t).y1 for t in title_texts.values())
    labels_bottom = min(inches(t).y0 for c in measured for t in bars[c].get_xticklabels() if t.get_text())
    bar_y = FRAMES_GAP + title_top - labels_bottom  # moves the colorbars up until their tick labels clear the titles
    rows_top = min(inches(t).y0 for t in header_texts.values()) - FRAMES_GAP
    for c in set(measured) - set(columns):
        header_texts.pop(c).remove()
        title_texts.pop(c).remove()
        bars.pop(c).remove()
    rows_bottom = bar_y + FRAMES_BAR_HEIGHT + FRAMES_GAP
    row_h = (rows_top - rows_bottom - (len(FRAMES_ROWS) - 1) * FRAMES_GAP) / len(FRAMES_ROWS)
    width = min(row_h * frames[c][f"{r}_image"].shape[1] / frames[c][f"{r}_image"].shape[0]
                for c in measured for r in FRAMES_ROWS)
    sizes = {t: _text_size(fig, t, label_size, rotation=90) for text in FRAMES_ROWS.values()
             for t in (text, _wrapped(text))}
    rows = {r: text if sizes[text][1] <= row_h else _wrapped(text) for r, text in FRAMES_ROWS.items()}
    label_w = max(sizes[text][0] for text in rows.values())
    # Column pitch: the frames' width, or more if neighboring headers would run into each other
    widths = [inches(header_texts[c]).width for c in columns]
    pitch = max([width] + [(a + b) / 2 for a, b in zip(widths, widths[1:])]) + FRAMES_COLUMN_GAP
    x_frames = region.x0 - FRAMES_MARGIN - (len(columns) - 1) * pitch - width  # the first column's left edge
    for j, c in enumerate(columns):
        x = x_frames + j * pitch
        header_texts[c].set_x((x + width / 2) / fig_w)
        title_texts[c].set_x((x + width / 2) / fig_w)
        bars[c].set_position(((x + 0.06 * width) / fig_w, bar_y / fig_h, 0.88 * width / fig_w, FRAMES_BAR_HEIGHT / fig_h))
    top = rows_top
    for r, text in rows.items():
        fig.text((x_frames - FRAMES_GAP - label_w / 2) / fig_w, (top - row_h / 2) / fig_h, text,
                 fontsize=label_size, rotation=90, ha="center", va="center", multialignment="center")
        for j, c in enumerate(columns):
            image = np.asarray(frames[c][f"{r}_image"], dtype=np.float64)[..., :3]
            h = width * image.shape[0] / image.shape[1]
            alpha = np.clip((1.0 - image.min(axis=2)) / 0.04, 0.0, 1.0)  # white transparent, antialiased edges partly
            ax = fig.add_axes(((x_frames + j * pitch) / fig_w, (top - (row_h + h) / 2) / fig_h, width / fig_w, h / fig_h))
            # Unresampled: vector outputs (PDF) embed the frame at its own resolution, not at the save dpi
            ax.imshow(np.dstack([image, alpha]), interpolation="none")
            ax.set_axis_off()
        top -= row_h + FRAMES_GAP
    return Bbox.from_extents(x_frames - FRAMES_GAP - label_w, region.y0, region.x1, region.y1)


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
    parser.add_argument("--title", action="store_true",
                        help="Draw the benchmark's title (default: none, the paper's sub-figure labels name them).")
    parser.add_argument("--width", type=float, default=None,
                        help="Printed width of each chart in inches (e.g. 5.5 for 0.8 x TOG's 7 in text width): the "
                        "figure is scaled, its fonts aren't, so the labels print at their nominal size when the paper "
                        "places it at this width (default: the natural size, about 6.2 in).")
    parser.add_argument("--no-legend", action="store_false", dest="legend",
                        help="Draw the charts without their solver legend, and write the legend alone to "
                        "<prefix>_legend.pdf, for a figure that stacks several charts under one legend.")
    parser.add_argument("--frames", default=None,
                        help="Prefix of benchmark_frames.py's output: adds its example problems' frames as a grid left "
                        "of each chart (see add_frames).")
    parser.add_argument("--columns", action="append", default=list(FRAMES_COLUMNS_DEFAULT),
                        help="[BENCHMARK=]PAIRS, the comma-separated frame pairs (error, material) of the grid, "
                        "repeatable; a benchmark's own entry wins (default: error,material, dynamic_h0.01=error, "
                        "dynamic_h0.0001=material).")
    parser.add_argument("--dataset", default=None,
                        help="The dataset file the results were run on, for the case labels' moduli (default: the "
                        "results file's dataset attribute, relative to the main checkout's root).")
    parser.add_argument("--elev", type=float, default=28.0, help="Camera elevation (degrees).")
    parser.add_argument("--azim", type=float, default=-62.0, help="Camera azimuth (degrees).")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution.")
    parser.add_argument("--font-dir", default=None, dest="font_dir",
                        help="Directory with the Linux Libertine OpenType files (default: PBAT_FONT_DIR, then the "
                        "installed fonts).")
    args = parser.parse_args()
    use_paper_fonts(args.font_dir)
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
            fig, region = plot_at_width(args.width, f, benchmark, solvers, args.elev, args.azim, order, factor,
                                        args.counts, args.title, args.legend, dataset=args.dataset)
            if args.frames:
                frames = load_frames(args.frames, benchmark)
                columns = tuple(results.per_benchmark(args.columns, benchmark).split(","))
                headers = {c: case_label(f[benchmark][frames[c]["case"]], frames[c]["case"], args.dataset)
                           for c in FRAMES_COLUMNS}  # all of them: the rows' height is measured with each
                region = add_frames(fig, region, frames, headers, columns)
            for ext in ("pdf", "png"):
                path = f"{args.output}_{benchmark}.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches=region)
                print(f"Wrote {path}")
            plt.close(fig)
        if not args.legend:
            # One legend for all the charts: the file's solver order, since each chart orders its own by depth
            fig, region = legend_figure([s for s in solvers if any(s in g for _, _, g in results.iter_cases(f))])
            for ext in ("pdf", "png"):
                path = f"{args.output}_legend.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches=region)
                print(f"Wrote {path}")
            plt.close(fig)


import unittest


class TestFrames(unittest.TestCase):

    def test_add_frames(self):
        import tempfile

        self.assertEqual((_length_text(0.257), _length_text(0.00143)), ("0.257 m", "1.43 mm"))
        self.assertEqual((_wrapped("Initial iterate"), _wrapped("Reference")), ("Initial\niterate", "Reference"))
        with tempfile.TemporaryDirectory() as d:
            prefix = os.path.join(d, "frames")
            spec = {"error": dict(case="Y1e7", quantity="error", vmin=0.0, vmax=0.25),
                    "material": dict(case="het", quantity="lambda", vmin=4.5, vmax=8.5)}
            for kind, pair in spec.items():
                pair["shades"] = np.linspace(0.0, 1.0, 12)[:, None].repeat(3, axis=1).tolist()
                for state in FRAMES_ROWS:
                    pair[state] = f"{kind}_{state}.png"
                    plt.imsave(os.path.join(d, pair[state]), np.full((30, 40, 3), 0.5))
            with open(f"{prefix}_static.json", "w", encoding="utf-8") as f:
                json.dump(spec, f)
            frames = load_frames(prefix, "static")
            self.assertEqual(frames["error"]["initial_image"].shape[:2], (30, 40))
            # A chart with x tick labels and a legend on top: the grid goes left of it, its headers on the legend's
            # baseline, its colorbar titles on the last tick label's
            fig = plt.figure(figsize=(6.0, 3.0))
            chart = fig.add_axes((0.1, 0.2, 0.85, 0.6))
            chart.bar([0, 1], [1, 2], label="bars")
            chart.set_xticks([0, 1], ["A", "$E \\in \\{1\\}$ Pa"])
            fig.legend(loc="upper center")
            region = Bbox.from_extents(0.0, 0.0, 6.0, 3.0)
            headers = {"error": "$E = 10^{7}$ Pa", "material": "$E \\in \\{10^{6}, 10^{8}\\}$ Pa"}
            texts_before = len(fig.texts)
            saved = add_frames(fig, region, frames, headers)
            self.assertEqual((saved.x1, saved.y0, saved.y1), (6.0, 0.0, 3.0))
            self.assertLess(saved.x0, -1.0)
            self.assertEqual(len(fig.axes), 1 + 2 + 4)  # the chart, 2 colorbars, 4 frames
            bars, frame_axes = fig.axes[1:3], fig.axes[3:]
            for ax in fig.axes[1:]:
                self.assertLessEqual(ax.get_position().x1 * 6.0, -FRAMES_MARGIN + 1e-6)  # left of the chart
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            header, title = fig.texts[texts_before], fig.texts[texts_before + 2]  # the first header and title
            # The chart's text sizes: its material labels' for the texts, its numbers' for the colorbar ticks
            chart.tick_params(axis="y", labelsize=6)
            self.assertEqual((header.get_fontsize(), title.get_fontsize()), (10.0, 10.0))  # matplotlib's default
            self.assertEqual(_chart_text_sizes(chart), (10.0, 6.0))
            self.assertAlmostEqual(_baseline(header, renderer), _baseline(fig.legends[0].get_texts()[0], renderer),
                                   delta=1.0)
            self.assertAlmostEqual(_baseline(title, renderer), _baseline(chart.get_xticklabels()[-1], renderer),
                                   delta=1.0)
            # Two rows sharing the height between: each frame a good part of it, the colorbars below them
            self.assertGreater(min(ax.get_position().height * 3.0 for ax in frame_axes), 0.6)
            self.assertLess(max(ax.get_position().y1 for ax in bars), min(ax.get_position().y0 for ax in frame_axes))
            self.assertEqual([t.get_text() for t in bars[1].get_xticklabels()], ["$10^{5}$", "$10^{8}$"])
            path = os.path.join(d, "out.pdf")
            fig.savefig(path, bbox_inches=saved)
            plt.close(fig)
            self.assertGreater(os.path.getsize(path), 1000)
            # Single columns (the stacked dynamic charts): one colorbar and two frames each, equally wide grids
            widths = []
            for column in FRAMES_COLUMNS:
                fig = plt.figure(figsize=(6.0, 3.0))
                fig.add_axes((0.1, 0.2, 0.85, 0.6)).set_xticks([0, 1], ["A", "$E \\in \\{1\\}$ Pa"])
                one = add_frames(fig, region, frames, headers, (column,))
                self.assertEqual(len(fig.axes), 1 + 1 + 2)
                widths.append(region.x0 - one.x0)
                plt.close(fig)
            self.assertAlmostEqual(widths[0], widths[1])
            self.assertLess(widths[0], region.x0 - saved.x0)


if __name__ == "__main__":
    main()
