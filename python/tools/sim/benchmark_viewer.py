"""
Polyscope viewer of a benchmark results file's trajectories (``benchmark.py --snapshots``): what each solver
does to the shape during a benchmark solve, side by side and in sync.

Pick a benchmark, case and sample. Every solver with a trajectory for that sample is drawn as an offset copy
of the body's boundary surface, labeled with its name. Translucent overlays in light colors, each with its own
checkbox and a color legend in the panel, show the problem at every copy: for dynamic problems the reference
solution x* (the set's ``x_ref``, on by default), the inertial target x~ (``xtilde``) and the start state x
(``samples/x``); for static problems the rest shape X, which is the solution x* (off by default). "Show solver
meshes" hides every solver mesh at once (the per-solver checkboxes still choose the copies), to see the
overlays alone, which are then drawn more opaque (opaque when only one is shown).

- **Frames.** The controls are the CPU editor's trajectory viewer's (``python/tools/vbd/ui/trajectory.py``):
  Load, a frame slider, Autoplay and Screenshot. Frame k stands for the solver time t_k = k x budget / F,
  the same for every solver (the frames are evenly spaced in solver time over the problem's budget), so the
  solvers stay synchronized: each copy shows its last frame at or before t_k, and a solve that has ended
  holds its last frame. Autoplay plays F / ``video_seconds`` frames per second, i.e. the solve slowed down by
  ``playback_factor`` = video_seconds / budget; Screenshot saves one ``<frame:08d>.png`` per frame instead,
  to the working directory. The solver time shows in seconds and as a fraction of the problem's T_N
  (Newton's time to its highest level, the benchmark's normalization), with the slowdown factor.
- **Per solver:** e/e0 at the shown frame, the highest level its state has crossed, its iterations and
  sweeps, how its solve ended, and buttons that jump to each level's crossing (``level_frames``): the
  shared time becomes that frame's, between the grid frames when the crossing isn't on one.
- **Colors:** the per-vertex error |x_i - x*_i| or the displacement from the start |x_i - x_i(0)|, on one
  color scale shared by the shown solvers (by default the largest value among them at the current time, so
  the colors show where each solver's error concentrates while e/e0 gives its size), each solver's color, or
  a material parameter (the dataset FEM's Lame mu, lambda or Young's modulus, per boundary face from its
  tetrahedron), named and colored as in the CPU editor, by default in log10(value + 1), so heterogeneous
  scenes show their regions.
- **Plot:** log10(e/e0) against solver time for every shown solver, with the levels and a marker at t.

The mesh comes from the dataset the results were run on (the root attribute ``dataset``, or ``--dataset``
when that file has moved): the FEM of the case's problem set, its tetrahedra reduced to their boundary
triangles. Needs h5py, numpy and polyscope only (and ``benchmark_plot``'s solver colors, with matplotlib), not
Warp.

Usage (without a results file, open one with Load):
    python -m python.tools.sim.benchmark_viewer [path/to/results.h5] \
        [--dataset path/to/PhysicsBasedAnimationToolkit/data/benchmarks/trampolines-test.h5] \
        [--benchmark static] [--case Y1e6] [--sample 0]

Tests: ``python -m unittest python.tools.sim.benchmark_viewer``.
"""

import argparse
from dataclasses import dataclass, field
import math
import os
import time

import h5py
import numpy as np

from . import benchmark_results as results
from .benchmark_plot import SOLVER_NAMES, solver_color
from ..vbd.ui import material

FEM_GROUP = "pbat.sim.dynamics.FemElastoDynamics"  # the dataset's stored FemElastoDynamics (pbat Archive)
MESH_GROUP = "pbat.fem.Mesh"
ERROR_FLOOR = 1e-12  # e/e0 below this plots at the floor (log scale)


def boundary_triangles(X: np.ndarray, T: np.ndarray, return_elements: bool = False):
    """The boundary triangles of a tetrahedral mesh, counter-clockwise seen from outside.

    Args:
        X (np.ndarray): (N, 3) vertex positions.
        T (np.ndarray): (M, 4) tetrahedra, of either orientation.
        return_elements (bool): Also return the tetrahedron each triangle belongs to.

    Returns:
        np.ndarray: (K, 3) the faces that belong to a single tetrahedron; with ``return_elements``,
        (faces, (K,) their tetrahedra).
    """
    T = np.asarray(T, dtype=np.int64).copy()
    a, b, c, d = (X[T[:, i]] for i in range(4))
    negative = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
    T[np.ix_(negative, [1, 2])] = T[np.ix_(negative, [2, 1])]
    # Outward faces of a positively oriented tetrahedron, each opposite one of its vertices
    faces = T[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]].reshape(-1, 3)
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inverse.ravel()] == 1
    if return_elements:
        return faces[boundary], np.repeat(np.arange(len(T)), 4)[boundary]
    return faces[boundary]


# The CPU editor's names for the material quantities (python/tools/vbd/ui/simulation.py, tetrahedral_elastodynamics_body.py)
MATERIAL_MODES = ("Lame mu", "Lame lambda", "Young's Modulus")


def element_materials(lame: np.ndarray, quadrature_elements: np.ndarray, n_elements: int) -> dict[str, np.ndarray]:
    """Per-element material parameters of a FemElastoDynamics (Pa).

    Args:
        lame (np.ndarray): (2, Q) Lame mu and lambda per quadrature point (``lamegU``).
        quadrature_elements (np.ndarray): (Q,) the element of each quadrature point (``egU``).
        n_elements (int): M.

    Returns:
        dict[str, np.ndarray]: ``MATERIAL_MODES`` -> (M,): mu and lambda (the mean over the element's quadrature
        points) and Young's modulus mu (3 lambda + 2 mu) / (lambda + mu).
    """
    e = np.asarray(quadrature_elements, dtype=np.int64).ravel()
    count = np.maximum(np.bincount(e, minlength=n_elements), 1)
    mu = np.bincount(e, np.asarray(lame[0], dtype=np.float64), n_elements) / count
    lam = np.bincount(e, np.asarray(lame[1], dtype=np.float64), n_elements) / count
    young = mu * (3 * lam + 2 * mu) / np.maximum(lam + mu, 1e-30)
    return dict(zip(MATERIAL_MODES, (mu, lam, young)))


def problem_fem_group(dataset: h5py.File, set_name: str) -> str:
    """The FEM group a problem set is posed on (as ``benchmark.py``): its ``fem`` attribute, else its scene's
    ``fem`` group, else (single-scene version-1 datasets) the root's."""
    attrs = dataset[set_name].attrs
    scene = attrs.get("scene")
    if scene is None:
        return "fem"
    return str(attrs.get("fem", f"{scene}/fem"))


def solver_order(f: h5py.File, case: h5py.Group) -> list[str]:
    """The case's solvers in the run's order (the root ``solvers`` attribute), then any others."""
    run = [str(s) for s in f.attrs.get("solvers", [])]
    return [s for s in run if s in case] + sorted(s for s in case if s not in run and "sample" in case[s])


def trajectory_samples(case: h5py.Group, solvers: list[str]) -> list[int]:
    """The samples some solver of the case has a trajectory for, sorted."""
    samples = set()
    for solver in solvers:
        if "trajectories" in case[solver]:
            samples.update(int(s) for s in case[solver]["trajectories"])
    return sorted(samples)


def list_cases(path: str) -> list[tuple[str, str, list[str], list[int]]]:
    """(benchmark, case, solvers, samples) of every case with trajectories, in the file's order."""
    with h5py.File(path, "r") as f:
        results.check_format(f, path)
        out = []
        for benchmark, case, g in results.iter_cases(f):
            solvers = solver_order(f, g)
            samples = trajectory_samples(g, solvers)
            if samples:
                out.append((benchmark, case, solvers, samples))
    return out


def _time_tolerance(t: float) -> float:
    return 1e-9 * max(abs(t), 1e-12)


@dataclass
class Trajectory:
    """One solver's frames of one solve, with its log."""

    solver: str
    x: np.ndarray  # (F, N, 3) float32 positions
    iteration: np.ndarray  # (F,) log entries of the frames
    time: np.ndarray  # (F,) the solver time each frame stands for: its target time, else its own
    t: np.ndarray  # (F,) the frames' own solver times
    relative_error: np.ndarray  # (F,) e/e0 of the frames
    sweeps: np.ndarray  # (F,) cumulative sweeps of the frames
    level_frames: np.ndarray  # (L,) frame of each level's crossing, -1 if not within the frames
    level_exact: np.ndarray  # (L,) bool: the frame is the crossing's own state
    crossing_time: np.ndarray  # (L,) solver time of each level's crossing, NaN if not reached
    crossing_iteration: np.ndarray  # (L,) log entry of each level's crossing, -1 if not reached
    log_time: np.ndarray  # (n,) the full log's solver times
    log_relative_error: np.ndarray  # (n,) the full log's e/e0
    termination: str
    e0: float
    _order: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._order = np.argsort(self.time, kind="stable")

    @property
    def end_time(self) -> float:
        """The solver time of the solve's last log entry."""
        return float(self.log_time[-1])

    def frame_at(self, t: float) -> int:
        """The frame shown at solver time t: the one standing for the latest time at or before t (the first
        frame before it)."""
        times = self.time[self._order]
        i = int(np.searchsorted(times, t + _time_tolerance(t), side="right")) - 1
        return int(self._order[max(i, 0)])

    def level_of_frame(self, frame: int) -> int:
        """The highest level the frame's state has crossed (index into the levels), -1 if none."""
        crossed = np.nonzero((self.crossing_iteration >= 0) & (self.crossing_iteration <= self.iteration[frame]))[0]
        return int(crossed[-1]) if crossed.size else -1


def load_trajectory(g: h5py.Group, solver: str, sample: int) -> Trajectory | None:
    """A solver's trajectory of one sample (its solver group ``g``), or None without one."""
    key = f"trajectories/{sample:06d}"
    if key not in g:
        return None
    position = int(np.nonzero(g["sample"][()] == sample)[0][0])
    tr = g[key]
    log = g[f"logs/{sample:06d}"]
    e0 = float(g["e0"][position])
    error = log["error"][()].astype(np.float64)
    log_time = log["time"][()].astype(np.float64)
    sweeps = log["sweeps"][()]
    iteration = tr["iteration"][()].astype(np.int64)
    t = tr["t"][()].astype(np.float64)
    t_target = tr["t_target"][()].astype(np.float64)
    relative = error / e0 if e0 > 0 else np.full_like(error, math.nan)
    return Trajectory(
        solver=solver,
        x=tr["x"][()],
        iteration=iteration,
        time=np.where(np.isfinite(t_target), t_target, t),
        t=t,
        relative_error=relative[iteration],
        sweeps=sweeps[iteration],
        level_frames=tr["level_frames"][()].astype(np.int64),
        level_exact=tr["level_exact"][()].astype(bool),
        crossing_time=g["crossing_time"][position].astype(np.float64),
        crossing_iteration=g["crossing_iteration"][position].astype(np.int64),
        log_time=log_time,
        log_relative_error=relative,
        termination=str(g["termination"].asstr()[position]),
        e0=e0,
    )


@dataclass
class Overlay:
    """A shape of the problem drawn translucent at every solver's copy."""

    name: str  # the checkbox and legend label
    positions: np.ndarray  # (N, 3)
    color: tuple[float, float, float]  # light, so the overlays don't hide the copies
    enabled: bool  # shown by default


# Light colors, distinct from each other and from the solvers' (Okabe-Ito) colors
REFERENCE_COLOR = (0.55, 0.85, 0.95)
TARGET_COLOR = (1.0, 0.8, 0.55)
START_COLOR = (0.72, 0.92, 0.62)
REST_COLOR = (0.8, 0.8, 0.8)
# Overlay opacity over the solver meshes, and without them (several overlays; a single one is opaque)
OVERLAY_OPACITY, OVERLAY_OPACITY_ALONE = 0.3, 0.6


def problem_overlays(problem: str, X: np.ndarray, samples: h5py.Group | None, sample: int) -> list[Overlay]:
    """The overlays of a problem: x*, x~ and x for a dynamic problem (from its set's ``samples``), X for a
    static one (whose solution it is)."""
    if problem == "static":
        return [Overlay("Rest shape X = x*", X, REST_COLOR, False)]
    load = lambda key: samples[key][sample].astype(np.float64)
    return [
        Overlay("Reference x* (x_ref)", load("x_ref"), REFERENCE_COLOR, True),
        Overlay("Inertial target x~ (xtilde)", load("xtilde"), TARGET_COLOR, False),
        Overlay("Start state x", load("x"), START_COLOR, False),
    ]


@dataclass
class Selection:
    """One sample of one case: the mesh, the reference and every solver's trajectory."""

    benchmark: str
    case: str
    sample: int
    problem: str  # "static" or "dynamic"
    label: str
    X: np.ndarray  # (N, 3) rest positions
    triangles: np.ndarray  # (K, 3) boundary triangles
    face_materials: dict[str, np.ndarray]  # MATERIAL_MODES -> (K,): each boundary triangle's tetrahedron's value
    reference: np.ndarray  # (N, 3) x*: X (static) or x_ref (dynamic)
    overlays: list[Overlay]
    levels: np.ndarray  # (L,) error reductions
    T_N: float  # the problem's normalization time (Newton's time to its highest level, or the fixed budget)
    span: float  # the solver time the frames span (the problem's budget)
    video_seconds: float
    n_frames: int  # F: frame k (0 <= k <= F) stands for the solver time k x span / F
    trajectories: list[Trajectory]

    @property
    def playback_factor(self) -> float:
        """Video seconds per second of solver time."""
        return self.video_seconds / self.span if self.span > 0 else math.nan

    @property
    def frame_seconds(self) -> float:
        """The video time between frames (1 / fps)."""
        return self.video_seconds / self.n_frames

    def frame_time(self, k: int) -> float:
        """The solver time of frame k."""
        return self.span * k / self.n_frames

    def frame_of_time(self, t: float) -> int:
        """The last frame at or before solver time t."""
        return min(int(math.floor(t / self.span * self.n_frames + 1e-9)), self.n_frames) if self.span > 0 else 0


def load_selection(path: str, benchmark: str, case: str, sample: int, dataset: str | None = None) -> Selection:
    """Loads one sample of a case: its trajectories, the dataset's mesh and the reference solution.

    Args:
        path (str): Results file (``benchmark.py --snapshots``).
        benchmark (str): E.g. ``static``, ``dynamic_h0.01``.
        case (str): The case (scene) name.
        sample (int): Problem index in the case's set.
        dataset (str | None): The dataset file, instead of the results file's ``dataset`` attribute.

    Returns:
        Selection: The case's sample, with one trajectory per solver that has one.
    """
    with h5py.File(path, "r") as f:
        results.check_format(f, path)
        g = f[benchmark][case]
        dataset = dataset or str(f.attrs["dataset"])
        if not os.path.isfile(dataset):
            raise FileNotFoundError(f"Dataset {dataset!r} not found; pass its path with --dataset")
        trajectories = []
        T_N = math.nan
        video_seconds, span, n_frames = math.nan, math.nan, 1
        for solver in solver_order(f, g):
            tr = load_trajectory(g[solver], solver, sample)
            if tr is None:
                continue
            trajectories.append(tr)
            if "normalization" in g[solver]:
                position = int(np.nonzero(g[solver]["sample"][()] == sample)[0][0])
                T_N = float(g[solver]["normalization"][position])
            attrs = g[solver][f"trajectories/{sample:06d}"].attrs
            video_seconds = float(attrs["video_seconds"])
            n_frames = max(1, int(attrs["requested_frames"]))
            if attrs["playback_factor"] > 0:
                span = video_seconds / float(attrs["playback_factor"])
        if not trajectories:
            raise ValueError(f"{path}: {benchmark}/{case} has no trajectory of sample {sample}")
        if not math.isfinite(span):
            span = max(float(tr.time.max()) for tr in trajectories)
        levels = results.benchmark_levels(f, benchmark)
        set_name, problem = str(g.attrs["set"]), str(g.attrs["problem"])
        label = str(g.attrs.get("label", case))
    with h5py.File(dataset, "r") as d:
        fem = d[problem_fem_group(d, set_name)][FEM_GROUP]
        X = fem[f"{MESH_GROUP}/X"][()].T.astype(np.float64)
        T = fem[f"{MESH_GROUP}/E"][()].T
        materials = element_materials(fem["lamegU"][()], fem["egU"][()], len(T))
        overlays = problem_overlays(problem, X, d[set_name]["samples"] if problem != "static" else None, sample)
    reference = X if problem == "static" else overlays[0].positions
    triangles, elements = boundary_triangles(X, T, return_elements=True)
    face_materials = {name: values[elements] for name, values in materials.items()}
    for tr in trajectories:
        if tr.x.shape[1] != X.shape[0]:
            raise ValueError(f"{tr.solver}'s frames have {tr.x.shape[1]} nodes, the dataset's FEM {X.shape[0]}: "
                             f"wrong --dataset?")
    return Selection(benchmark, case, sample, problem, label, X, triangles, face_materials, reference, overlays, levels, T_N,
                     span, video_seconds, n_frames, trajectories)


def log_curve(tr: Trajectory, times: np.ndarray) -> np.ndarray:
    """log10(e/e0) of the solve's state at each of ``times`` (its last log entry at or before it; NaN after
    the solve's end), floored at ``ERROR_FLOOR``."""
    i = np.searchsorted(tr.log_time, times, side="right") - 1
    r = tr.log_relative_error[np.clip(i, 0, None)]
    y = np.log10(np.clip(np.where(np.isfinite(r), r, 1.0), ERROR_FLOOR, None))
    y[times > tr.end_time + _time_tolerance(tr.end_time)] = math.nan
    return y


def display_name(solver: str) -> str:
    return SOLVER_NAMES.get(solver, solver)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def _level_text(level: float) -> str:
    return f"{100 * level:.10g}%"


COLOR_MODES = ("Error |x - x*|", "Displacement |x - x(0)|", "Solver color") + MATERIAL_MODES
SCALE_MODES = ("Shared max at the frame", "Shared max at the start", "Fixed max")
_WARNING_COLOR = (1.0, 0.75, 0.2, 1.0)  # as the CPU editor's trajectory viewer


def slowdown_text(playback_factor: float) -> str:
    """``slowed down 200x`` (playback factor above 1), ``real time`` or ``sped up 4x``."""
    if not math.isfinite(playback_factor) or playback_factor <= 0:
        return "unknown speed"
    if abs(playback_factor - 1.0) < 5e-3:
        return "real time"
    if playback_factor > 1.0:
        return f"slowed down {playback_factor:.3g}x"
    return f"sped up {1.0 / playback_factor:.3g}x"


class Viewer:
    """The Polyscope GUI: the meshes, the panel and the playback state."""

    def __init__(self, dataset: str | None):
        self.path: str | None = None
        self.dataset = dataset
        self.cases: list = []
        self.case_index, self.sample_index = 0, 0
        self.selection: Selection | None = None
        self.error: str | None = None
        # The shared solver time: frame k's, or a level crossing's (jump) between frames
        self.frame, self.t = 0, 0.0
        self.autoplay, self.screenshot = False, False
        self.clock_time, self.clock_lag = time.time(), 0.0
        self.color_mode, self.scale_mode, self.fixed_max, self.vmax = 0, 0, 1.0, 0.0
        self.spacing = 1.15
        self.show_solvers = True  # every solver mesh, above the per-solver checkboxes
        self.log_material = True  # material colors in log10(value + 1), as the CPU editor
        self.show_overlay: dict[str, bool] = {}  # by overlay name, kept across selections
        self.visible: dict[str, bool] = {}
        self.meshes, self.offsets = {}, {}
        self.ghosts = {}  # (solver, overlay name) -> the overlay's mesh at the solver's copy
        self.vertices: np.ndarray | None = None  # boundary vertex indices
        self.faces: np.ndarray | None = None  # boundary triangles over them
        self.shown: dict[str, int] = {}
        self.dirty = True
        self.plot_times = np.zeros(0)
        self.plot_curves: dict[str, np.ndarray] = {}

    # --- Loading ---

    def load(self, path: str, benchmark: str | None = None, case: str | None = None, sample: int | None = None):
        """Opens a results file at its first case with trajectories (or the given one)."""
        try:
            cases = list_cases(path)
            if not cases:
                raise ValueError(f"{path} has no trajectories: run benchmark.py with --snapshots")
            case_index = next((i for i, (b, c, _, _) in enumerate(cases)
                               if benchmark in (None, b) and case in (None, c)), None)
            if case_index is None:
                raise ValueError(f"{path}: no case with trajectories matches benchmark {benchmark}, case {case}")
        except (OSError, KeyError, ValueError) as e:
            self.error = str(e)
            return
        self.path, self.cases = path, cases
        samples = cases[case_index][3]
        self.select(case_index, samples.index(sample) if sample in samples else 0)

    def load_dialog(self):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Benchmark Results File",
            filetypes=[("HDF5 benchmark results files", "*.h5"), ("All Files", "*.*")],
        )
        root.destroy()
        if path:
            self.load(path)

    def select(self, case_index: int, sample_index: int):
        import polyscope as ps

        benchmark, case, _, samples = self.cases[case_index]
        sample_index = min(sample_index, len(samples) - 1)
        try:
            selection = load_selection(self.path, benchmark, case, samples[sample_index], self.dataset)
        except (OSError, KeyError, ValueError) as e:
            self.error = str(e)
            return
        # Frame the copies when the scene changes; another sample keeps the camera
        new_scene = self.selection is None or self.selection.X.shape != selection.X.shape or len(
            self.selection.trajectories) != len(selection.trajectories)
        self.case_index, self.sample_index = case_index, sample_index
        self.error = None
        ps.remove_all_structures()
        self.meshes, self.ghosts, self.offsets, self.shown = {}, {}, {}, {}
        self.selection = s = selection
        self.vertices, self.faces = np.unique(s.triangles, return_inverse=True)
        self.faces = self.faces.reshape(-1, 3)
        self.place()
        names = [tr.solver for tr in s.trajectories]
        for tr in s.trajectories:
            name = display_name(tr.solver)
            self.visible.setdefault(tr.solver, True)
            for overlay in s.overlays:
                self.show_overlay.setdefault(overlay.name, overlay.enabled)
                self.ghosts[tr.solver, overlay.name] = ps.register_surface_mesh(
                    f"{name} {overlay.name}", self.copy_positions(overlay.positions, tr.solver), self.faces,
                    color=overlay.color, transparency=self.overlay_opacity())
            mesh = ps.register_surface_mesh(name, self.copy_positions(tr.x[0], tr.solver), self.faces,
                                            color=_rgb(solver_color(tr.solver, names)))
            self.meshes[tr.solver] = mesh
        if new_scene:
            self.frame_camera()
        self.frame = min(self.frame, s.n_frames)
        self.t = s.frame_time(self.frame)
        self.plot_times = np.linspace(0.0, s.span, 400)
        self.plot_curves = {tr.solver: log_curve(tr, self.plot_times) for tr in s.trajectories}
        self.dirty = True

    def frame_camera(self):
        """Looks at the row of copies from the front and above, all of them in view."""
        import polyscope as ps

        s = self.selection
        lo, hi = s.X.min(axis=0), s.X.max(axis=0)
        last = self.offsets[s.trajectories[-1].solver]
        center = (lo + hi + last) / 2
        width = float(hi[0] - lo[0] + last[0])
        # Horizontal half-angle of polyscope's default 45 degree vertical field of view, at a 16:10 window
        distance = 0.6 * width / math.tan(math.atan(math.tan(math.radians(22.5)) * 1.6))
        ps.look_at(center + distance * np.array([0.0, -0.9, 0.45]), center)

    def copy_positions(self, x: np.ndarray, solver: str) -> np.ndarray:
        """Positions x (N, 3) of the body, at a solver's copy: its boundary vertices, offset."""
        return x[self.vertices] + self.offsets[solver]

    def place(self):
        """The copies' offsets: side by side along x, ``spacing`` body widths apart."""
        s = self.selection
        width = max(float(np.ptp(s.X[:, 0])), 1e-6)
        self.offsets = {tr.solver: np.array([i * self.spacing * width, 0.0, 0.0])
                        for i, tr in enumerate(s.trajectories)}

    # --- Meshes ---

    def vertex_values(self, tr: Trajectory, frame: int) -> np.ndarray:
        x = tr.x[frame][self.vertices].astype(np.float64)
        if self.color_mode == 0:
            return np.linalg.norm(x - self.selection.reference[self.vertices], axis=1)
        return np.linalg.norm(x - tr.x[0][self.vertices], axis=1)

    def overlay_opacity(self) -> float:
        """Translucent over the solver meshes; without them, more opaque, and opaque if it is the only one."""
        if self.show_solvers:
            return OVERLAY_OPACITY
        shown = sum(self.show_overlay.get(o.name, o.enabled) for o in self.selection.overlays)
        return 1.0 if shown <= 1 else OVERLAY_OPACITY_ALONE

    def material_values(self) -> tuple[np.ndarray, tuple[float, float], str]:
        """The color mode's material parameter per boundary face, its color range and colormap: the CPU editor's
        (log10(value + 1), and its fixed range for Young's modulus), or linear over the mesh's values."""
        name = COLOR_MODES[self.color_mode]
        values = self.selection.face_materials[name]
        cmap = material.young_modulus_log10_cmap() if name == "Young's Modulus" else material.lame_parameters_cmap()
        if self.log_material:
            values = np.log10(values + 1)
            if name == "Young's Modulus":
                return values, material.young_modulus_log10_range(), cmap
        lo, hi = float(values.min()), float(values.max())
        return values, (lo, hi if hi > lo else lo + max(abs(lo), 1.0) * 1e-6), cmap

    def update_meshes(self):
        s = self.selection
        frames = {tr.solver: tr.frame_at(self.t) for tr in s.trajectories}
        if not self.dirty and frames == self.shown:
            return
        shown = [tr for tr in s.trajectories if self.visible[tr.solver]]
        for tr in s.trajectories:
            mesh = self.meshes[tr.solver]
            mesh.set_enabled(self.show_solvers and self.visible[tr.solver])
            mesh.update_vertex_positions(self.copy_positions(tr.x[frames[tr.solver]], tr.solver))
            for overlay in s.overlays:
                ghost = self.ghosts[tr.solver, overlay.name]
                ghost.set_enabled(self.show_overlay[overlay.name] and self.visible[tr.solver])
                ghost.set_transparency(self.overlay_opacity())
                ghost.update_vertex_positions(self.copy_positions(overlay.positions, tr.solver))
        if self.dirty:  # the color mode may have changed
            for tr in s.trajectories:
                self.meshes[tr.solver].remove_all_quantities()
        if self.color_mode < 2 and shown:
            values = {tr.solver: self.vertex_values(tr, frames[tr.solver]) for tr in shown}
            if self.scale_mode == 0:
                vmax = max(float(v.max()) for v in values.values())
            elif self.scale_mode == 1:
                vmax = max(float(self.vertex_values(tr, 0).max()) for tr in shown)
            else:
                vmax = self.fixed_max
            self.vmax = max(vmax, 1e-12)
            for solver, v in values.items():
                self.meshes[solver].add_scalar_quantity(COLOR_MODES[self.color_mode], v, vminmax=(0.0, self.vmax),
                                                        cmap="viridis", enabled=True)
        elif self.color_mode >= 3 and self.dirty:  # the material doesn't change with the frames
            values, vminmax, cmap = self.material_values()
            for tr in s.trajectories:
                self.meshes[tr.solver].add_scalar_quantity(COLOR_MODES[self.color_mode], values, defined_on="faces",
                                                           vminmax=vminmax, cmap=cmap, enabled=True)
        self.shown = frames
        self.dirty = False

    # --- Panel ---

    def draw_frames(self):
        """The CPU editor's trajectory controls (``python/tools/vbd/ui/trajectory.py``), over the shared frames."""
        import polyscope as ps
        import polyscope.imgui as imgui

        s = self.selection
        width = imgui.GetWindowWidth()
        imgui.SetNextItemWidth(width * 0.7)
        changed, frame = imgui.SliderInt("Frame", self.frame, 0, s.n_frames)
        if changed:
            self.frame, self.t = frame, s.frame_time(frame)
        autoplay_changed, self.autoplay = imgui.Checkbox("Autoplay", self.autoplay)
        imgui.SameLine()
        _, self.screenshot = imgui.Checkbox("Screenshot", self.screenshot)
        on_frame = abs(self.t - s.frame_time(self.frame)) <= _time_tolerance(self.t)
        imgui.Text(f"Solver time {self.t:.5g} s = {self.t / s.T_N:.4f} T_N"
                   + ("" if on_frame else " (a level crossing)"))
        imgui.Text(f"Playback {s.n_frames / s.video_seconds:.3g} fps: {slowdown_text(s.playback_factor)}")
        now = time.time()
        if self.autoplay and self.frame < s.n_frames:
            if autoplay_changed:
                self.clock_lag = 0.0
            elif self.screenshot:
                ps.screenshot("{:08d}.png".format(self.frame))
                self.frame += 1
            else:
                # Frames due since the last tick at the video's frame rate, keeping the remainder
                self.clock_lag += now - self.clock_time
                advance = int(self.clock_lag / s.frame_seconds)
                self.clock_lag -= advance * s.frame_seconds
                self.frame = min(self.frame + advance, s.n_frames)
            self.t = s.frame_time(self.frame)
        else:
            self.clock_lag = 0.0
        self.clock_time = now

    def draw_overlays(self):
        """One checkbox per overlay, labeled in the overlay's color (the legend)."""
        import polyscope.imgui as imgui

        changed, self.show_solvers = imgui.Checkbox("Show solver meshes", self.show_solvers)
        self.dirty |= changed
        if imgui.IsItemHovered():
            imgui.SetTooltip("Hide every solver mesh to see the overlays alone (the per-solver checkboxes still choose "
                             "the copies); the overlays are then drawn more opaque")
        imgui.TextDisabled("Overlays, at every copy:")
        for overlay in self.selection.overlays:
            imgui.PushID(overlay.name)
            changed, self.show_overlay[overlay.name] = imgui.Checkbox("##overlay", self.show_overlay[overlay.name])
            self.dirty |= changed
            imgui.SameLine()
            imgui.TextColored((*overlay.color, 1.0), overlay.name)
            imgui.PopID()

    def draw_solvers(self):
        """Per solver: visibility, e/e0, the level reached, iterations and jumps to the level crossings."""
        import polyscope.imgui as imgui

        s = self.selection
        names = [tr.solver for tr in s.trajectories]
        for tr in s.trajectories:
            imgui.PushID(tr.solver)
            changed, self.visible[tr.solver] = imgui.Checkbox("##visible", self.visible[tr.solver])
            self.dirty |= changed
            imgui.SameLine()
            frame = self.shown[tr.solver]
            level = tr.level_of_frame(frame)
            imgui.TextColored((*_rgb(solver_color(tr.solver, names)), 1.0), display_name(tr.solver))
            imgui.SameLine()
            sweeps = int(tr.sweeps[frame])
            imgui.Text(f"e/e0 {tr.relative_error[frame]:.2e}, level "
                       f"{_level_text(s.levels[level]) if level >= 0 else '-'}, iteration {tr.iteration[frame]}"
                       + (f" ({sweeps} sweeps)" if sweeps not in (0, int(tr.iteration[frame])) else ""))
            if self.t >= tr.end_time:
                imgui.SameLine()
                imgui.TextDisabled(f"(ended: {tr.termination} at {tr.end_time:.4g} s)")
            imgui.Text("    Jump to")
            for l, level_value in enumerate(s.levels):
                imgui.SameLine()
                k = int(tr.level_frames[l])
                label = _level_text(level_value)
                if k < 0:
                    imgui.TextDisabled(label)
                    continue
                if imgui.SmallButton(f"{label}##level{l}"):
                    self.t = float(tr.time[k])
                    self.frame = s.frame_of_time(self.t)
                if imgui.IsItemHovered():
                    imgui.SetTooltip(f"Crossed at {tr.crossing_time[l]:.5g} s, iteration {tr.crossing_iteration[l]}"
                                     + ("" if tr.level_exact[l] else "; shown: the nearest snapshot"))
            imgui.PopID()

    def draw_plot(self):
        """log10(e/e0) against solver time, per shown solver, with the levels and a marker at the time."""
        import polyscope.imgui as imgui

        s = self.selection
        w = max(imgui.GetContentRegionAvail()[0], 240.0)
        h = 170.0
        x0, y0 = imgui.GetCursorScreenPos()
        imgui.InvisibleButton("##error_plot", (w, h))
        draw = imgui.GetWindowDrawList()
        pad_l, pad_b, pad_t = 44.0, 18.0, 6.0
        curves = {k: v for k, v in self.plot_curves.items() if self.visible.get(k)}
        lows = [math.floor(c[np.isfinite(c)].min()) for c in curves.values() if np.isfinite(c).any()]
        ymin = min(lows + [math.floor(math.log10(1 - s.levels[-1])), -1])
        ymax = 0.0

        def to_screen(t, y):
            return (x0 + pad_l + (w - pad_l - 4) * t / max(s.span, 1e-30),
                    y0 + pad_t + (h - pad_t - pad_b) * (ymax - y) / (ymax - ymin))

        draw.AddRectFilled((x0, y0), (x0 + w, y0 + h), imgui.GetColorU32((0.08, 0.08, 0.1, 1.0)))
        grid = imgui.GetColorU32((0.35, 0.35, 0.4, 1.0))
        text = imgui.GetColorU32((0.8, 0.8, 0.8, 1.0))
        step = max(1, int(math.ceil((ymax - ymin) / 6)))
        for y in range(0, int(ymin) - 1, -step):
            a, b = to_screen(0.0, y), to_screen(s.span, y)
            draw.AddLine(a, b, grid, 1.0)
            draw.AddText((x0 + 4, a[1] - 7), text, f"1e{y}" if y else "1")
        level_color = imgui.GetColorU32(_WARNING_COLOR[:3] + (0.6,))
        for level in s.levels:
            y = math.log10(1 - level)
            if y >= ymin:
                draw.AddLine(to_screen(0.0, y), to_screen(s.span, y), level_color, 1.0)
        draw.AddText((x0 + pad_l, y0 + h - pad_b + 2), text, "0")
        right = f"{s.span:.4g} s"
        draw.AddText((x0 + w - imgui.CalcTextSize(right)[0] - 4, y0 + h - pad_b + 2), text, right)
        names = [tr.solver for tr in s.trajectories]
        t = min(self.t, s.span)
        for tr in s.trajectories:
            if tr.solver not in curves:
                continue
            color = imgui.GetColorU32((*_rgb(solver_color(tr.solver, names)), 1.0))
            c = curves[tr.solver]
            keep = np.isfinite(c)
            points = [to_screen(tt, y) for tt, y in zip(self.plot_times[keep], c[keep])]
            if len(points) > 1:
                draw.AddPolyline(points, len(points), color, 0, 1.5)
            r = tr.relative_error[self.shown.get(tr.solver, 0)]
            y = math.log10(min(max(r, ERROR_FLOOR), 1.0)) if math.isfinite(r) else 0.0
            draw.AddCircleFilled(to_screen(t, max(y, ymin)), 3.5, color)
        draw.AddLine(to_screen(t, ymax), to_screen(t, ymin), imgui.GetColorU32((1.0, 1.0, 1.0, 0.8)), 1.0)
        imgui.TextDisabled("log10(e/e0) against solver time; orange: the levels")

    def draw_labels(self):
        """Each solver's name under its copy, projected to the screen (perspective camera)."""
        import polyscope as ps
        import polyscope.imgui as imgui

        s = self.selection
        params = ps.get_view_camera_parameters()
        V = np.asarray(params.get_view_mat(), dtype=np.float64)
        f = 1.0 / math.tan(math.radians(params.get_fov_vertical_deg()) / 2)
        aspect = params.get_aspect()
        width, height = ps.get_window_size()
        lo, hi = s.X.min(axis=0), s.X.max(axis=0)
        below = np.array([(lo[0] + hi[0]) / 2, lo[1], lo[2]])  # the front bottom edge's middle
        draw = imgui.GetBackgroundDrawList()
        names = [tr.solver for tr in s.trajectories]
        for tr in s.trajectories:
            if not self.visible[tr.solver]:
                continue
            p = V @ np.append(below + self.offsets[tr.solver], 1.0)
            if p[2] >= 0:  # behind the camera
                continue
            x = (f / aspect * p[0] / -p[2] + 1) / 2 * width
            y = (1 - f * p[1] / -p[2]) / 2 * height
            name = display_name(tr.solver)
            color = imgui.GetColorU32((*_rgb(solver_color(tr.solver, names)), 1.0))
            draw.AddText((x - imgui.CalcTextSize(name)[0] / 2, y + 8), color, name)

    def callback(self):
        import polyscope.imgui as imgui

        imgui.PushID("BenchmarkViewer")
        width = imgui.GetWindowWidth()
        if imgui.Button("Load Results", [width / 2.1, 0]):
            self.load_dialog()
        if self.path is not None:
            imgui.SameLine()
            imgui.TextDisabled(os.path.basename(self.path))
        if self.error:
            imgui.PushStyleColor(imgui.ImGuiCol_Text, _WARNING_COLOR)
            imgui.TextWrapped(self.error)
            imgui.PopStyleColor()
        s = self.selection
        if s is None:
            imgui.PopID()
            return
        imgui.SetNextItemWidth(width * 0.5)
        changed, i = imgui.Combo("Benchmark/case", self.case_index, [f"{b}/{c}" for b, c, _, _ in self.cases])
        if changed:
            self.select(i, 0)
        imgui.SetNextItemWidth(width * 0.5)
        changed, j = imgui.Combo("Sample", self.sample_index, [str(k) for k in self.cases[self.case_index][3]])
        if changed:
            self.select(self.case_index, j)
        s = self.selection
        imgui.TextWrapped(f"{s.label}, {s.problem}: T_N = {s.T_N:.4g} s; {s.n_frames} frames over "
                          f"{s.span:.4g} s ({s.span / s.T_N:.3g} T_N), {s.video_seconds:g} s of video")
        imgui.Separator()
        self.draw_frames()
        imgui.Separator()
        imgui.SetNextItemWidth(width * 0.5)
        changed, self.color_mode = imgui.Combo("Color", self.color_mode, list(COLOR_MODES))
        self.dirty |= changed
        if self.color_mode < 2:
            imgui.SetNextItemWidth(width * 0.5)
            changed, self.scale_mode = imgui.Combo("Scale", self.scale_mode, list(SCALE_MODES))
            self.dirty |= changed
            if self.scale_mode == 2:
                imgui.SetNextItemWidth(width * 0.3)
                changed, self.fixed_max = imgui.InputFloat("Max (m)", self.fixed_max, format="%.4g")
                self.dirty |= changed
            imgui.TextDisabled(f"Color scale 0 to {self.vmax:.4g} m")
        elif self.color_mode >= 3:
            changed, self.log_material = imgui.Checkbox("Log10 scale", self.log_material)
            self.dirty |= changed
            raw = s.face_materials[COLOR_MODES[self.color_mode]]
            imgui.SameLine()
            imgui.TextDisabled(f"boundary faces: {raw.min():.4g} to {raw.max():.4g} Pa")
        imgui.SetNextItemWidth(width * 0.3)
        changed, self.spacing = imgui.SliderFloat("Spacing", self.spacing, 1.0, 3.0, format="%.2f")
        if changed:
            self.place()
            self.dirty = True
        self.draw_overlays()
        self.update_meshes()
        imgui.Separator()
        self.draw_solvers()
        imgui.Separator()
        self.draw_plot()
        imgui.PopID()
        self.draw_labels()


def main():
    parser = argparse.ArgumentParser(prog="Benchmark trajectory viewer",
                                     description="Polyscope viewer of benchmark.py --snapshots trajectories.")
    parser.add_argument("results", nargs="?", default=None,
                        help="Results file (.h5) written by benchmark.py --snapshots (default: open one with Load).")
    parser.add_argument("--dataset", default=None,
                        help="The dataset file the results were run on (default: the results file's dataset attribute).")
    parser.add_argument("--benchmark", default=None, help="Benchmark to open first, e.g. static or dynamic_h0.01.")
    parser.add_argument("--case", default=None, help="Case (scene) to open first, e.g. Y1e6.")
    parser.add_argument("--sample", type=int, default=None, help="Sample to open first.")
    args = parser.parse_args()

    import polyscope as ps

    ps.set_program_name("Benchmark trajectories")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.init()
    viewer = Viewer(args.dataset)
    if args.results is not None:
        viewer.load(args.results, args.benchmark, args.case, args.sample)
    ps.set_user_callback(viewer.callback)
    ps.show()


import unittest


class TestBenchmarkViewer(unittest.TestCase):
    @staticmethod
    def cube():
        """A unit cube split into 6 tetrahedra, some of them negatively oriented."""
        X = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.float64)
        T = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
        return X, T

    def test_boundary_triangles(self):
        X, T = self.cube()
        F = boundary_triangles(X, T)
        self.assertEqual(len(F), 12)
        # Outward: the divergence theorem gives the cube's volume, +1
        a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        self.assertAlmostEqual(np.einsum("ij,ij->", a, np.cross(b, c)) / 6, 1.0)
        F = boundary_triangles(X, T[:, [1, 0, 2, 3]])  # flipped tetrahedra
        a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
        self.assertAlmostEqual(np.einsum("ij,ij->", a, np.cross(b, c)) / 6, 1.0)

    @staticmethod
    def trajectory(time, iteration, crossing_iteration):
        n = len(time)
        log_time = np.linspace(0.0, 1.0, int(max(iteration)) + 1)
        return Trajectory(
            solver="VBD", x=np.zeros((n, 1, 3), dtype=np.float32), iteration=np.asarray(iteration),
            time=np.asarray(time, dtype=np.float64), t=np.asarray(time, dtype=np.float64),
            relative_error=np.ones(n), sweeps=np.asarray(iteration), level_frames=np.full(2, -1),
            level_exact=np.zeros(2, dtype=bool), crossing_time=np.full(2, math.nan),
            crossing_iteration=np.asarray(crossing_iteration), log_time=log_time,
            log_relative_error=np.ones_like(log_time), termination="target", e0=1.0,
        )

    def test_frame_at(self):
        # Targets 0, 0.25, 0.5 and a level-crossing frame at 0.3 between them
        tr = self.trajectory([0.0, 0.25, 0.3, 0.5], [0, 10, 12, 20], [12, -1])
        self.assertEqual([tr.frame_at(t) for t in (0.0, 0.1, 0.25, 0.29, 0.3, 0.49, 0.5, 9.0)],
                         [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual([tr.level_of_frame(k) for k in range(4)], [-1, -1, 0, 0])

    def test_log_curve(self):
        tr = self.trajectory([0.0, 1.0], [0, 4], [-1, -1])
        tr.log_relative_error = np.array([1.0, 0.1, 0.01, 0.0, np.nan])
        y = log_curve(tr, np.array([0.0, 0.3, 0.5, 0.75, 1.0, 2.0]))
        np.testing.assert_allclose(y[:5], [0.0, -1.0, -2.0, math.log10(ERROR_FLOOR), 0.0])
        self.assertTrue(math.isnan(y[5]))  # after the solve's end

    def test_overlays(self):
        """A dataset and a results file of one cube scene: each problem's overlays load from its set and sit
        at every solver's copy, as the solvers' frames."""
        import tempfile

        X, T = self.cube()
        rng = np.random.default_rng(0)
        shapes = {key: rng.standard_normal((2, 8, 3)).astype(np.float32) for key in ("x", "xtilde", "x_ref")}
        levels = (0.5,)
        sample = 1  # not the first, so the sample (not its position) must select the shapes
        with tempfile.TemporaryDirectory() as root:
            dataset, path = os.path.join(root, "dataset.h5"), os.path.join(root, "results.h5")
            with h5py.File(dataset, "w") as d:
                mesh = d.create_group(f"S/fem/{FEM_GROUP}/{MESH_GROUP}")
                mesh.create_dataset("X", data=X.T.astype(np.float32))
                mesh.create_dataset("E", data=T.T)
                # Two materials: the first three tetrahedra 100x stiffer (mu, lambda per quadrature point)
                lame = np.array([[1e7] * 3 + [1e5] * 3, [4e7] * 3 + [4e5] * 3], dtype=np.float32)
                d[f"S/fem/{FEM_GROUP}"].create_dataset("lamegU", data=lame)
                d[f"S/fem/{FEM_GROUP}"].create_dataset("egU", data=np.arange(6)[:, None])
                for name in ("S/static/E1", "S/dynamic_h0.01/E1"):
                    d.create_group(name).attrs["scene"] = "S"
                for key, value in shapes.items():
                    d.create_dataset(f"S/dynamic_h0.01/E1/samples/{key}", data=value)
            with h5py.File(path, "w") as f:
                f.attrs.update(format=results.FORMAT, version=results.VERSION, dataset=dataset,
                               solvers=["VBD", "Newton"])
                for benchmark, problem in (("static", "static"), ("dynamic_h0.01", "dynamic")):
                    case = results.require_case(f, benchmark, "S", {"set": f"S/{benchmark}/E1", "problem": problem},
                                                levels=levels)
                    for solver in ("VBD", "Newton"):
                        c = results.level_crossings(np.array([1.0, 0.4]), np.array([0.0, 1.0]), levels)
                        summary = dict(sample=sample, e0=1.0, iterations=1, sweeps=1, solver_time=1.0,
                                       final_relative_error=0.4, termination="target", deformation_energy=0.0,
                                       crossings=c, budget=1.0, normalization=1.0)
                        trajectory = dict(iteration=np.array([0, 1]), t=np.array([0.0, 1.0]),
                                          t_target=np.array([0.0, 1.0]), level_frames=np.array([1]),
                                          level_exact=np.array([True]), x=np.stack([X, X]), requested_frames=1,
                                          ring_slots=2, stride=1, device_bytes=0, video_seconds=2.0,
                                          playback_factor=2.0)
                        log = (np.array([1.0, 0.4]), np.array([0.0, 1.0]), np.array([0, 1]))
                        results.write_solver(case, solver, levels, [summary], [log], [trajectory])
            self.assertEqual([(b, c, n) for b, c, _, n in list_cases(path)], [("static", "S", [1]),
                                                                                ("dynamic_h0.01", "S", [1])])
            static = load_selection(path, "static", "S", sample)
            dynamic = load_selection(path, "dynamic_h0.01", "S", sample)
        self.assertEqual([(o.name, o.enabled) for o in static.overlays], [("Rest shape X = x*", False)])
        np.testing.assert_array_equal(static.overlays[0].positions, X)
        self.assertEqual([o.enabled for o in dynamic.overlays], [True, False, False])
        for overlay, key in zip(dynamic.overlays, ("x_ref", "xtilde", "x")):
            np.testing.assert_array_equal(overlay.positions, shapes[key][sample])
        np.testing.assert_array_equal(dynamic.reference, shapes["x_ref"][sample])
        # Each boundary face has its tetrahedron's material: 2 faces per tetrahedron on the cube's boundary
        _, elements = boundary_triangles(X, T, return_elements=True)
        stiff = elements < 3
        self.assertEqual(int(stiff.sum()), 6)
        mu = dynamic.face_materials["Lame mu"]
        np.testing.assert_allclose(mu[stiff], 1e7)
        np.testing.assert_allclose(mu[~stiff], 1e5)
        np.testing.assert_allclose(dynamic.face_materials["Lame lambda"][stiff], 4e7)
        young = dynamic.face_materials["Young's Modulus"]
        np.testing.assert_allclose(young[stiff], 1e7 * (3 * 4e7 + 2e7) / 5e7, rtol=1e-6)  # nu = 0.4: E = 2.8 mu
        self.assertEqual(dynamic.span, 1.0)
        # Positioned like the copies: the second solver's overlays and frames share its offset
        viewer = Viewer(None)
        viewer.selection = dynamic
        viewer.vertices, faces = np.unique(dynamic.triangles, return_inverse=True)
        viewer.place()
        self.assertEqual(len(viewer.vertices), 8)
        for overlay in dynamic.overlays:
            shift = viewer.copy_positions(overlay.positions, "Newton") - overlay.positions[viewer.vertices]
            np.testing.assert_allclose(shift, np.tile([viewer.spacing * 1.0, 0.0, 0.0], (8, 1)))
            np.testing.assert_allclose(viewer.copy_positions(overlay.positions, "VBD"), overlay.positions)
        frame = dynamic.trajectories[1].x[1]
        np.testing.assert_allclose(viewer.copy_positions(frame, "Newton") - frame, shift)
        # Overlays alone: more opaque without the solver meshes, opaque when only one is shown
        viewer.show_overlay = {o.name: o.enabled for o in dynamic.overlays}
        self.assertEqual(viewer.overlay_opacity(), OVERLAY_OPACITY)
        viewer.show_solvers = False
        self.assertEqual(viewer.overlay_opacity(), 1.0)
        viewer.show_overlay["Start state x"] = True
        self.assertEqual(viewer.overlay_opacity(), OVERLAY_OPACITY_ALONE)
        # Material colors: log10(value + 1) over the mesh's range, Young's modulus on the CPU editor's range
        viewer.color_mode = COLOR_MODES.index("Lame mu")
        values, vminmax, cmap = viewer.material_values()
        np.testing.assert_allclose(vminmax, (math.log10(1e5 + 1), math.log10(1e7 + 1)))
        self.assertEqual(len(values), len(dynamic.triangles))
        viewer.color_mode = COLOR_MODES.index("Young's Modulus")
        self.assertEqual(viewer.material_values()[1], material.young_modulus_log10_range())
        viewer.log_material = False
        np.testing.assert_allclose(viewer.material_values()[1], (young.min(), young.max()))


if __name__ == "__main__":
    main()
