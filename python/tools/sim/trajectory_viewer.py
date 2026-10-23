"""
Polyscope viewer of ``simulate.py`` outputs (or the GPU GUI's Serialization): several trajectories of one scene,
e.g. one per solver, played side by side or overlaid, in sync. The CPU editor's trajectory viewer
(``python/tools/vbd/ui/trajectory.py``) plays one trajectory in the editor's scene; this one needs only the output
files, which store the FEM at the root (``FemElastoDynamics/{X,E,m}``) and each frame's positions ``sim/<frame>/x``.

- **Same scene.** Every file (and the reference) must have the first one's node count and rest positions, else
  it is rejected with an error naming it. Frames are read from the files when shown, a few kept in memory.
- **Sync by simulated time.** A frame's time is its time step count (the attribute ``sim/<frame>.attrs["t"]``,
  or for older outputs the frame schedule of ``params.attrs["fps"]``) times ``params/Integration.attrs["dt"]``, so
  trajectories with other frame rates, time steps or start times (``--resume``) align. The frame slider steps
  through every trajectory's frame times; each copy shows its last frame at or before the shared time (its first
  frame before it starts, its last after it ends). Play advances the shared time at the playback rate (simulated
  seconds per second), looping at the end.
- **Layout.** Side by side along x, ``spacing`` scene widths apart, labeled left to right in the panel, or
  overlaid, with an opacity for all copies.
- **Colors.** The node speed |x_k - x_{k-1}| / (t_k - t_{k-1}) or, with ``--reference`` (e.g. a converged Newton
  run), the distance |x - x_ref| per node to the reference's frame at the shown frame's time, both on one scale
  shared by the copies (a fixed max, or the largest value among them at the frame); each body in its own color
  (the FEM's connected components); or each trajectory's color.
- **Per trajectory**, at its shown frame: the frame and its time, the solve statuses of the time steps since the
  previous frame (``sim/<frame>/solve_status``: how the solves ended, and their worst ratios of the residual and
  the penetration to their tolerances), and with a reference the mass-weighted RMS distance
  sqrt(sum_i m_i |x_i - x_ref,i|^2 / sum_i m_i) (the M-norm, over the bodies' nodes), with a plot of it over
  time, filled in progressively in the background.
- Static colliders (element-less nodes after the bodies') are drawn as translucent points; their triangles aren't
  stored. The view is framed on the bodies: huge plane colliders would otherwise make it tiny.

Usage (from the repo root):

    python -m python.tools.sim.trajectory_viewer a.h5 b.h5 ... [--labels A B ...] [--reference ref.h5]

Labels default to each file's solver (``params.attrs["solver"]``) when they differ, else the file names. Needs
h5py, numpy, scipy and polyscope (and ``benchmark_plot``'s solver colors, with matplotlib), not Warp.

Tests: ``python -m unittest python.tools.sim.trajectory_viewer``.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import math
import os
import time

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .benchmark_plot import solver_color
from .benchmark_viewer import boundary_triangles, display_name, _rgb
from .profiling.trajectory import frame_steps

CACHED_FRAMES = 3  # per trajectory: the shown frame, the one before it (speeds) and one more
REST_TOLERANCE = 1e-6  # rest positions match within this fraction of the scene's size
RMS_BUDGET = 0.015  # seconds per GUI tick spent filling in the RMS curves
_WARNING_COLOR = (1.0, 0.75, 0.2, 1.0)  # as the CPU editor's trajectory viewer
_COLLIDER_COLOR = (0.55, 0.55, 0.6)
# Colors of trajectories that aren't told apart by their solver (the same solver twice, or none recorded)
_PALETTE = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")


def _time_tolerance(t: float) -> float:
    return 1e-9 * max(abs(t), 1e-6)


@dataclass
class Trajectory:
    """One output file's frames, read when needed."""

    path: str
    label: str
    solver: str  # params.attrs["solver"], or ""
    file: h5py.File
    frames: list[str]  # frame group names, in order
    times: np.ndarray  # (F,) simulated time (s) of each frame
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _status: dict = field(default_factory=dict)

    def read(self, k: int) -> np.ndarray:
        """Frame k's positions (N, 3), uncached."""
        return np.asarray(self.file["sim"][self.frames[k]]["x"], dtype=np.float64)

    def positions(self, k: int) -> np.ndarray:
        """Frame k's positions (N, 3), keeping the last :data:`CACHED_FRAMES` read."""
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        x = self.read(k)
        self._cache[k] = x
        if len(self._cache) > CACHED_FRAMES:
            self._cache.popitem(last=False)
        return x

    def frame_at(self, t: float) -> int:
        """The last frame at or before time t (the first before the trajectory starts)."""
        return max(int(np.searchsorted(self.times, t + _time_tolerance(t), side="right")) - 1, 0)

    def speeds(self, k: int) -> np.ndarray:
        """Per node, |x_k - x_{k-1}| / (t_k - t_{k-1}) (m/s); zero at the first frame."""
        x = self.positions(k)
        elapsed = self.times[k] - self.times[k - 1] if k > 0 else 0.0
        if elapsed <= 0:
            return np.zeros(len(x))
        return np.linalg.norm(x - self.positions(k - 1), axis=1) / elapsed

    def status(self, k: int) -> list[str]:
        """Frame k's solve statuses as text lines (see :func:`status_lines`)."""
        if k not in self._status:
            g = self.file["sim"][self.frames[k]]
            self._status[k] = status_lines(g["solve_status"]) if "solve_status" in g else []
        return self._status[k]

    def close(self):
        self.file.close()


def open_trajectory(path: str, label: str | None = None) -> Trajectory:
    """Opens a ``simulate.py`` output (read-only, kept open)."""
    f = h5py.File(path, "r")
    try:
        if "FemElastoDynamics/X" not in f or "sim" not in f:
            raise ValueError(f"{path} is not a simulate.py output (no FemElastoDynamics/X or sim group)")
        frames = sorted(k for k in f["sim"].keys() if k.isdigit())
        if not frames:
            raise ValueError(f"{path} has no frames")
        dt = float(f["params/Integration"].attrs["dt"])
        times = frame_steps(f).astype(np.float64) * dt
        solver = str(f["params"].attrs.get("solver", ""))
    except (KeyError, ValueError):
        f.close()
        raise
    return Trajectory(path, label or "", solver, f, frames, times)


def default_labels(trajectories: list[Trajectory]) -> list[str]:
    """Each file's solver when they all differ, else the file names (numbered if those repeat too)."""
    solvers = [tr.solver for tr in trajectories]
    if all(solvers) and len(set(solvers)) == len(solvers):
        return [display_name(s) for s in solvers]
    names = [os.path.splitext(os.path.basename(tr.path))[0] for tr in trajectories]
    if len(set(names)) == len(names):
        return names
    return [f"{i}: {n}" for i, n in enumerate(names)]


def trajectory_colors(trajectories: list[Trajectory]) -> list[tuple[float, float, float]]:
    """The benchmark plots' solver colors when the solvers all differ, else a palette by position."""
    solvers = [tr.solver for tr in trajectories]
    if all(solvers) and len(set(solvers)) == len(solvers):
        return [_rgb(solver_color(s, solvers)) for s in solvers]
    return [_rgb(_PALETTE[i % len(_PALETTE)]) for i in range(len(trajectories))]


def body_ids(n_nodes: int, E: np.ndarray) -> np.ndarray:
    """Per node, its body: the connected component of the tetrahedra, numbered in order of their first node
    (element-less nodes are components of their own)."""
    rows = np.repeat(E, 4, axis=1).ravel()
    cols = np.tile(E, (1, 4)).ravel()
    graph = sp.coo_matrix((np.ones_like(rows, dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
    _, component = connected_components(graph, directed=False)
    _, first = np.unique(component, return_index=True)
    rank = np.empty_like(first)
    rank[np.argsort(first)] = np.arange(first.size)
    return rank[component]


def mass_rms(x: np.ndarray, x_ref: np.ndarray, m: np.ndarray) -> float:
    """sqrt(sum_i m_i |x_i - x_ref,i|^2 / sum_i m_i), the M-norm distance per unit mass (m)."""
    d2 = np.einsum("ij,ij->i", x - x_ref, x - x_ref)
    return float(np.sqrt(np.dot(m, d2) / max(float(m.sum()), 1e-30)))


def status_lines(g: h5py.Group) -> list[str]:
    """A frame's solve statuses (``status.StatusLog.write_h5``: the time steps since the previous frame), per
    solver family: how the solves ended, then their most outer iterations and worst ratios to tolerance. Enum
    names come from the group's attributes, so the viewer needs no Warp."""
    lines = []
    if "vbd" in g:
        v = g["vbd"][()]
        names = [str(n) for n in g.attrs["solve_end"]]
        counts = np.bincount(v["reason"], minlength=len(names))
        ends = ", ".join(f"{names[r].lower().replace('_', ' ')} {c}" for r, c in enumerate(counts) if c)
        lines.append(f"{len(v)} solve{'s' if len(v) > 1 else ''}: {ends}")
        lines.append(f"  outer max {v['outer'].max()} / {v['max_outer'][-1]}, "
                     f"residual / tol {v['gradient_ratio'].max():.3g}, "
                     f"penetration / tol {v['feasibility_ratio'].max():.3g}")
    if "newton_solves" in g:
        n = g["newton_solves"][()]
        lines.append(f"{len(n)} Newton solve{'s' if len(n) > 1 else ''}: {int(n['converged'].sum())} converged")
        lines.append(f"  outer max {n['outer'].max()} / {n['max_outer'][-1]}, "
                     f"residual / tol {np.nanmax(n['gradient_ratio'], initial=0.0):.3g}, "
                     f"penetration / tol {np.nanmax(n['feasibility_ratio'], initial=0.0):.3g}")
    if int(g.attrs.get("lost", 0)):
        lines.append(f"{int(g.attrs['lost'])} solves lost (status ring overwritten)")
    return lines or ["no solves"]


@dataclass
class Scene:
    """The shared FEM: rest positions, masses, the boundary surface and the static colliders' nodes."""

    X: np.ndarray  # (N, 3)
    m: np.ndarray  # (N,) lumped masses
    body_nodes: np.ndarray  # the nodes of some tetrahedron
    colliders: np.ndarray  # the element-less nodes
    vertices: np.ndarray  # boundary vertex indices
    faces: np.ndarray  # boundary triangles over them
    bodies: np.ndarray  # per boundary vertex, its body
    width: float  # along x


def load_scene(f: h5py.File) -> Scene:
    X = np.asarray(f["FemElastoDynamics/X"], dtype=np.float64)
    E = np.asarray(f["FemElastoDynamics/E"], dtype=np.int64)
    m = np.asarray(f["FemElastoDynamics/m"], dtype=np.float64).ravel() if "FemElastoDynamics/m" in f else (
        np.ones(len(X)))
    body_nodes = np.unique(E)
    vertices, faces = np.unique(boundary_triangles(X, E), return_inverse=True)
    width = float(np.ptp(X[body_nodes, 0])) if len(body_nodes) else 0.0
    return Scene(X, m, body_nodes, np.setdiff1d(np.arange(len(X)), body_nodes), vertices, faces.reshape(-1, 3),
                 body_ids(len(X), E)[vertices], max(width, 1e-6))


def check_same_scene(scene: Scene, tr: Trajectory, first: str):
    """Raises a ValueError naming ``tr``'s file unless its node count and rest positions are the scene's."""
    X = np.asarray(tr.file["FemElastoDynamics/X"], dtype=np.float64)
    if X.shape != scene.X.shape:
        raise ValueError(f"{tr.path} has {len(X)} nodes, {first} {len(scene.X)}: not the same scene")
    size = max(float(np.ptp(scene.X, axis=0).max()), 1e-6)
    gap = float(np.abs(X - scene.X).max(initial=0.0))
    if gap > REST_TOLERANCE * size:
        raise ValueError(f"{tr.path}'s rest positions differ from {first}'s by up to {gap:.3g} m: not the same scene")


def load(paths: list[str], labels: list[str] | None = None, reference: str | None = None):
    """Opens the trajectories and the reference, checked against the first file's scene.

    Returns:
        tuple[Scene, list[Trajectory], Trajectory | None]
    """
    if labels is not None and len(labels) != len(paths):
        raise ValueError(f"{len(labels)} labels, for {len(paths)} trajectory file(s): give one label per file")
    opened = []
    try:
        for path in paths:
            opened.append(open_trajectory(path))
        ref = open_trajectory(reference, "Reference") if reference else None
        if ref is not None:
            opened.append(ref)
        scene = load_scene(opened[0].file)
        for tr in opened[1:]:
            check_same_scene(scene, tr, paths[0])
    except Exception:
        for tr in opened:
            tr.close()
        raise
    trajectories = opened[: len(paths)]
    for tr, label in zip(trajectories, labels or default_labels(trajectories)):
        tr.label = label
    if len({tr.label for tr in trajectories}) != len(trajectories):
        raise ValueError(f"labels must differ: {[tr.label for tr in trajectories]}")
    return scene, trajectories, ref


def time_grid(trajectories: list[Trajectory]) -> np.ndarray:
    """Every trajectory's frame times, sorted, times closer than the tolerance merged."""
    times = np.sort(np.concatenate([tr.times for tr in trajectories]))
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = np.diff(times) > np.array([_time_tolerance(t) for t in times[1:]])
    return times[keep]


COLOR_MODES = ("Speed |v|", "Distance |x - x_ref|", "Body", "Trajectory color")


class Viewer:
    """The Polyscope GUI: the copies, the panel and the playback state."""

    def __init__(self, scene: Scene, trajectories: list[Trajectory], reference: Trajectory | None = None):
        self.scene, self.trajectories, self.reference = scene, trajectories, reference
        self.colors = trajectory_colors(trajectories)
        self.grid = time_grid(trajectories)
        self.frame, self.t = 0, float(self.grid[0])
        self.play, self.rate = False, 1.0
        self.clock_time = time.time()
        self.overlay, self.spacing, self.opacity = False, 1.15, 0.5
        self.collider_opacity = 0.3
        self.color_mode = 0
        self.auto_scale = {0: False, 1: True}  # per color mode: the largest value at the frame, else the fixed max
        self.fixed_max = {0: 10.0, 1: 0.01}  # m/s, m
        self.vmax = 0.0
        self.meshes, self.points = [], []
        self.shown: list[int] | None = None
        self.dirty = True
        # RMS distance to the reference per (trajectory, frame), filled in by fill_rms in time order
        self.rms: list[np.ndarray] = [np.full(len(tr.times), np.nan) for tr in trajectories]
        self._rms_queue = sorted(((tr.times[k], i, k) for i, tr in enumerate(trajectories)
                                  for k in range(len(tr.times))), reverse=True) if reference else []
        self._rms_ref: tuple[int, np.ndarray] | None = None

    # --- Data ---

    def offset(self, i: int) -> np.ndarray:
        return np.zeros(3) if self.overlay else np.array([i * self.spacing * self.scene.width, 0.0, 0.0])

    def reference_positions(self, t: float) -> tuple[int, np.ndarray]:
        k = self.reference.frame_at(t)
        return k, self.reference.positions(k)

    def frame_rms(self, i: int, k: int) -> float:
        """Trajectory i's frame k's mass-weighted RMS distance to the reference (cached)."""
        if np.isnan(self.rms[i][k]):
            tr, s = self.trajectories[i], self.scene
            _, x_ref = self.reference_positions(tr.times[k])
            self.rms[i][k] = mass_rms(tr.positions(k)[s.body_nodes], x_ref[s.body_nodes], s.m[s.body_nodes])
        return float(self.rms[i][k])

    def fill_rms(self, budget: float = RMS_BUDGET) -> bool:
        """Computes RMS values in time order for up to ``budget`` seconds, reading each reference frame once and
        not disturbing the shown frames' cache. Returns True when all are done."""
        s, start = self.scene, time.perf_counter()
        while self._rms_queue and time.perf_counter() - start < budget:
            t, i, k = self._rms_queue.pop()
            if not np.isnan(self.rms[i][k]):
                continue
            j = self.reference.frame_at(t)
            if self._rms_ref is None or self._rms_ref[0] != j:
                self._rms_ref = (j, self.reference.read(j))
            x = self.trajectories[i].read(k)
            self.rms[i][k] = mass_rms(x[s.body_nodes], self._rms_ref[1][s.body_nodes], s.m[s.body_nodes])
        if not self._rms_queue:
            self._rms_ref = None
        return not self._rms_queue

    def vertex_values(self, i: int, k: int) -> np.ndarray:
        """The color mode's value per boundary vertex: speed (m/s) or distance to the reference (m)."""
        tr, v = self.trajectories[i], self.scene.vertices
        if self.color_mode == 0:
            return tr.speeds(k)[v]
        _, x_ref = self.reference_positions(tr.times[k])
        return np.linalg.norm(tr.positions(k)[v] - x_ref[v], axis=1)

    def set_time(self, t: float):
        self.t = t
        self.frame = max(int(np.searchsorted(self.grid, t + _time_tolerance(t), side="right")) - 1, 0)

    # --- Meshes ---

    def body_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The copies' bounding box at rest, over the bodies' nodes only (colliders such as planes can be huge)."""
        X = self.scene.X[self.scene.body_nodes] if len(self.scene.body_nodes) else self.scene.X
        lo, hi = X.min(axis=0), X.max(axis=0)
        return lo, hi + self.offset(len(self.trajectories) - 1)

    def update_extents(self):
        """With colliders, polyscope's scene extents (length scale and bounding box, which set its home view and
        clipping) are the copies' bodies', not its automatic ones over every structure."""
        import polyscope as ps

        if len(self.scene.colliders):
            lo, hi = self.body_bounds()
            ps.set_automatically_compute_scene_extents(False)
            ps.set_bounding_box(lo, hi)
            ps.set_length_scale(max(float(np.linalg.norm(hi - lo)), 1e-6))

    def frame_camera(self):
        """Looks at the copies' bodies from the front and above, all of them in view (as the benchmark viewer)."""
        import polyscope as ps

        self.update_extents()
        last = self.offset(len(self.trajectories) - 1)
        lo, hi = self.body_bounds()
        hi = hi - last  # one copy's box
        center = (lo + hi + last) / 2
        # The row's width at the horizontal half-angle of polyscope's default 45 degree vertical field of view (16:10
        # window), and at least one copy's bounding sphere in the vertical field of view (overlaid copies)
        distance = max(0.6 * float(hi[0] - lo[0] + last[0]) / math.tan(math.atan(math.tan(math.radians(22.5)) * 1.6)),
                       0.5 * float(np.linalg.norm(hi - lo)) / math.tan(math.radians(22.5)))
        ps.look_at(center + distance * np.array([0.0, -0.9, 0.45]), center)

    def register(self):
        import polyscope as ps

        s = self.scene
        for i, tr in enumerate(self.trajectories):
            x = tr.positions(0)
            mesh = ps.register_surface_mesh(tr.label, x[s.vertices] + self.offset(i), s.faces, color=self.colors[i],
                                            smooth_shade=True)
            self.meshes.append(mesh)
            if len(s.colliders):
                self.points.append(ps.register_point_cloud(f"{tr.label} colliders", x[s.colliders] + self.offset(i),
                                                           color=_COLLIDER_COLOR, radius=0.002))
        self.frame_camera()

    def update_meshes(self):
        s = self.scene
        frames = [tr.frame_at(self.t) for tr in self.trajectories]
        if not self.dirty and frames == self.shown:
            return
        for i, (tr, k) in enumerate(zip(self.trajectories, frames)):
            x = tr.positions(k)
            mesh = self.meshes[i]
            mesh.update_vertex_positions(x[s.vertices] + self.offset(i))
            mesh.set_transparency(self.opacity if self.overlay else 1.0)
            if self.points:
                self.points[i].update_point_positions(x[s.colliders] + self.offset(i))
                self.points[i].set_enabled(not self.overlay or i == 0)
                self.points[i].set_transparency(self.collider_opacity)
            if self.dirty:  # the color mode may have changed
                mesh.remove_all_quantities()
        if self.color_mode in (0, 1) and (self.color_mode == 0 or self.reference is not None):
            values = [self.vertex_values(i, k) for i, k in enumerate(frames)]
            vmax = max(float(v.max(initial=0.0)) for v in values) if self.auto_scale[self.color_mode] else (
                self.fixed_max[self.color_mode])
            self.vmax = max(vmax, 1e-12)
            for mesh, v in zip(self.meshes, values):
                mesh.add_scalar_quantity(COLOR_MODES[self.color_mode], v, vminmax=(0.0, self.vmax), cmap="viridis",
                                         enabled=True)
        elif self.color_mode == 2 and self.dirty:  # the bodies don't change with the frames
            for mesh in self.meshes:
                mesh.add_scalar_quantity("Body", s.bodies, datatype="categorical", enabled=True)
        self.shown = frames
        self.dirty = False

    # --- Panel ---

    def draw_frames(self):
        import polyscope as ps
        import polyscope.imgui as imgui

        width = imgui.GetWindowWidth()
        imgui.SetNextItemWidth(width * 0.6)
        changed, frame = imgui.SliderInt("Frame", self.frame, 0, len(self.grid) - 1)
        if changed:
            self.frame, self.t = frame, float(self.grid[frame])
        imgui.SameLine()
        if imgui.Button("Screenshot"):
            ps.screenshot("{:08d}.png".format(self.frame))
        if imgui.IsItemHovered():
            imgui.SetTooltip("Saves the view as <frame:08d>.png in the working directory")
        play_changed, self.play = imgui.Checkbox("Play", self.play)
        imgui.SameLine()
        imgui.SetNextItemWidth(width * 0.35)
        _, self.rate = imgui.SliderFloat("Playback rate", self.rate, 0.05, 2.0, format="%.2fx")
        imgui.Text(f"t = {self.t:.4f} s of {self.grid[-1]:.4f} s")
        now = time.time()
        if self.play and not play_changed:
            t = self.t + (now - self.clock_time) * self.rate
            self.set_time(t if t <= self.grid[-1] + _time_tolerance(t) else float(self.grid[0]))
        self.clock_time = now

    def draw_display(self):
        import polyscope.imgui as imgui

        width = imgui.GetWindowWidth()
        changed, self.overlay = imgui.Checkbox("Overlay", self.overlay)
        if changed:
            self.frame_camera()
            self.dirty = True
        imgui.SameLine()
        imgui.SetNextItemWidth(width * 0.3)
        if self.overlay:
            changed, self.opacity = imgui.SliderFloat("Opacity", self.opacity, 0.05, 1.0, format="%.2f")
        else:
            changed, self.spacing = imgui.SliderFloat("Spacing", self.spacing, 1.0, 3.0, format="%.2f")
            if changed:
                self.update_extents()  # the row's
        self.dirty |= changed
        if self.points:
            imgui.SetNextItemWidth(width * 0.3)
            changed, self.collider_opacity = imgui.SliderFloat("Collider opacity", self.collider_opacity, 0.0, 1.0,
                                                               format="%.2f")
            self.dirty |= changed
        modes = list(COLOR_MODES) if self.reference is not None else [m for i, m in enumerate(COLOR_MODES) if i != 1]
        index = modes.index(COLOR_MODES[self.color_mode])
        imgui.SetNextItemWidth(width * 0.5)
        changed, index = imgui.Combo("Color", index, modes)
        if changed:
            self.color_mode = COLOR_MODES.index(modes[index])
            self.dirty = True
        if self.color_mode in (0, 1):
            unit = "m/s" if self.color_mode == 0 else "m"
            changed, self.auto_scale[self.color_mode] = imgui.Checkbox("Scale to the frame's max",
                                                                       self.auto_scale[self.color_mode])
            self.dirty |= changed
            if not self.auto_scale[self.color_mode]:
                imgui.SameLine()
                imgui.SetNextItemWidth(width * 0.3)
                changed, value = imgui.InputFloat(f"Max ({unit})", self.fixed_max[self.color_mode], format="%.4g")
                if changed:
                    self.fixed_max[self.color_mode] = max(value, 1e-12)
                    self.dirty = True
            imgui.TextDisabled(f"Shared color scale 0 to {self.vmax:.4g} {unit}")

    def draw_trajectories(self):
        import polyscope.imgui as imgui

        if not self.overlay:
            imgui.TextDisabled("Left to right:")
        for i, (tr, k) in enumerate(zip(self.trajectories, self.shown)):
            imgui.PushID(f"trajectory{i}")
            imgui.TextColored((*self.colors[i], 1.0), tr.label)
            imgui.SameLine()
            note = " (not started)" if self.t < tr.times[0] - _time_tolerance(tr.times[0]) else (
                " (ended)" if self.t > tr.times[-1] + _time_tolerance(tr.times[-1]) else "")
            imgui.Text(f"frame {k} / {len(tr.times) - 1}, t = {tr.times[k]:.4f} s{note}")
            if imgui.IsItemHovered():
                imgui.SetTooltip(tr.path)
            if self.reference is not None:
                j, _ = self.reference_positions(tr.times[k])
                gap = abs(self.reference.times[j] - tr.times[k])
                imgui.Text(f"    RMS distance to the reference {self.frame_rms(i, k):.4g} m"
                           + (f" (reference frame {gap:.3g} s earlier)" if gap > _time_tolerance(tr.times[k]) else ""))
            for line in tr.status(k):
                imgui.TextDisabled(f"    {line}")
            imgui.PopID()

    def draw_plot(self):
        """The RMS distance to the reference against time, per trajectory, with a marker at the shown frames."""
        import polyscope.imgui as imgui

        done = self.fill_rms()
        w = max(imgui.GetContentRegionAvail()[0], 240.0)
        h = 150.0
        x0, y0 = imgui.GetCursorScreenPos()
        imgui.InvisibleButton("##rms_plot", (w, h))
        draw = imgui.GetWindowDrawList()
        pad_l, pad_b, pad_t = 54.0, 18.0, 6.0
        known = [r[np.isfinite(r)] for r in self.rms]
        ymax = max([float(r.max()) for r in known if len(r)] + [1e-12])
        t0, t1 = float(self.grid[0]), max(float(self.grid[-1]), float(self.grid[0]) + 1e-12)

        def to_screen(t, y):
            return (x0 + pad_l + (w - pad_l - 4) * (t - t0) / (t1 - t0),
                    y0 + pad_t + (h - pad_t - pad_b) * (1 - y / ymax))

        draw.AddRectFilled((x0, y0), (x0 + w, y0 + h), imgui.GetColorU32((0.08, 0.08, 0.1, 1.0)))
        text = imgui.GetColorU32((0.8, 0.8, 0.8, 1.0))
        draw.AddLine(to_screen(t0, 0.0), to_screen(t1, 0.0), imgui.GetColorU32((0.35, 0.35, 0.4, 1.0)), 1.0)
        draw.AddText((x0 + 4, y0 + pad_t - 2), text, f"{ymax:.3g} m")
        draw.AddText((x0 + 4, y0 + h - pad_b - 12), text, "0")
        draw.AddText((x0 + pad_l, y0 + h - pad_b + 2), text, f"{t0:.3g} s")
        right = f"{t1:.3g} s"
        draw.AddText((x0 + w - imgui.CalcTextSize(right)[0] - 4, y0 + h - pad_b + 2), text, right)
        for i, (tr, k) in enumerate(zip(self.trajectories, self.shown)):
            color = imgui.GetColorU32((*self.colors[i], 1.0))
            keep = np.isfinite(self.rms[i])
            points = [to_screen(t, y) for t, y in zip(tr.times[keep], self.rms[i][keep])]
            if len(points) > 1:
                draw.AddPolyline(points, len(points), color, 0, 1.5)
            if keep[k]:
                draw.AddCircleFilled(to_screen(tr.times[k], self.rms[i][k]), 3.5, color)
        draw.AddLine(to_screen(self.t, 0.0), to_screen(self.t, ymax), imgui.GetColorU32((1.0, 1.0, 1.0, 0.8)), 1.0)
        imgui.TextDisabled("RMS distance to the reference (M-norm) against time"
                           + ("" if done else f", computing ({sum(int(k.sum()) for k in map(np.isfinite, self.rms))}"
                              f" / {sum(len(r) for r in self.rms)} frames)"))

    def callback(self):
        import polyscope.imgui as imgui

        imgui.PushID("TrajectoryViewer")
        if self.reference is not None:
            imgui.TextDisabled(f"Reference: {os.path.basename(self.reference.path)}")
        self.draw_frames()
        imgui.Separator()
        self.draw_display()
        self.update_meshes()
        imgui.Separator()
        self.draw_trajectories()
        if self.reference is not None:
            imgui.Separator()
            self.draw_plot()
        imgui.PopID()


def main():
    parser = argparse.ArgumentParser(
        prog="Trajectory viewer", description="Polyscope viewer of several simulate.py outputs of one scene, in sync.")
    parser.add_argument("trajectories", nargs="+", help="simulate.py outputs (.h5) of the same scene.")
    parser.add_argument("--labels", nargs="+", default=None, help="One label per trajectory (default: the solvers, "
                        "else the file names).")
    parser.add_argument("--reference", default=None, help="A trajectory (.h5) of the same scene to measure the "
                        "distance to, e.g. a converged Newton run.")
    args = parser.parse_args()
    try:
        scene, trajectories, reference = load(args.trajectories, args.labels, args.reference)
    except (OSError, KeyError, ValueError) as e:
        parser.error(str(e))

    import polyscope as ps

    ps.set_program_name("Trajectories")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    # Data exploration: no ground plane (it only gives simulations a floor reference), best anti-aliasing
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)
    ps.set_transparency_mode("pretty")  # the overlay's translucent copies
    ps.init()
    viewer = Viewer(scene, trajectories, reference)
    viewer.register()
    ps.set_user_callback(viewer.callback)
    ps.show()


import unittest


class TestTrajectoryViewer(unittest.TestCase):
    @staticmethod
    def write(path, X, E, frames, dt=0.01, fps=50.0, solver="VBD", m=None, with_t=True, statuses=None):
        """A simulate.py output: frames is a list of (time step count, positions)."""
        with h5py.File(path, "w") as f:
            g = f.create_group("FemElastoDynamics")
            g.create_dataset("X", data=X)
            g.create_dataset("E", data=E)
            g.create_dataset("m", data=np.ones(len(X)) if m is None else m)
            f.create_group("params").attrs.update(fps=fps, solver=solver)
            f["params"].create_group("Integration").attrs["dt"] = dt
            f.create_group("sim")
            for k, (t, x) in enumerate(frames):
                frame = f.create_group(f"sim/{k:08d}")
                frame.create_dataset("x", data=x)
                if with_t:
                    frame.attrs["t"] = t
                if statuses is not None and k in statuses:
                    statuses[k](frame.create_group("solve_status"))

    @staticmethod
    def scene():
        """Two tetrahedra (separate bodies) and one element-less collider node."""
        X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1], [0, 0, -1]], dtype=np.float64)
        E = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        return X, E

    @staticmethod
    def vbd_status(group):
        # The fields of status.VBD_DTYPE that status_lines reads (status.py imports Warp)
        dtype = [("outer", np.int32), ("reason", np.int32), ("gradient_ratio", np.float32),
                 ("feasibility_ratio", np.float32), ("max_outer", np.int32)]
        rows = np.zeros(2, dtype=dtype)
        rows["reason"] = [1, 4]
        rows["outer"] = [3, 10]
        rows["max_outer"] = 10
        rows["gradient_ratio"] = [0.5, 3.0]
        rows["feasibility_ratio"] = [0.2, 0.1]
        group.create_dataset("vbd", data=rows)
        group.attrs["solve_end"] = ["RUNNING", "CONVERGED", "GRADIENT_UNMET", "FEASIBILITY_UNMET", "BOTH_UNMET"]
        group.attrs["lost"] = 0

    def files(self, root):
        """VBD: dt 0.01, a frame every 2 steps from step 0; Newton: resumed at step 4, a frame every 4 steps, moving
        twice as fast; a reference at rest, without frame step counts (the fps schedule)."""
        X, E = self.scene()
        moved = lambda k, v: X + np.outer(np.arange(len(X)) < 8, [v * k, 0.0, 0.0])  # noqa: E731
        paths = [os.path.join(root, f"{n}.h5") for n in ("a", "b", "ref")]
        self.write(paths[0], X, E, [(2 * k, moved(k, 0.1)) for k in range(5)],
                   statuses={1: self.vbd_status, 2: lambda g: g.attrs.update(lost=0)})
        self.write(paths[1], X, E, [(4 * k + 4, moved(k + 1, 0.4)) for k in range(2)], fps=25.0, solver="Newton")
        self.write(paths[2], X, E, [(2 * k, X) for k in range(5)], solver="Newton", with_t=False)
        return paths

    def test_load_and_sync(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            paths = self.files(root)
            scene, (a, b), ref = load(paths[:2], reference=paths[2])
            try:
                self.assertEqual([a.label, b.label], ["VBD", "Newton"])
                np.testing.assert_allclose(a.times, [0.0, 0.02, 0.04, 0.06, 0.08])
                np.testing.assert_allclose(b.times, [0.04, 0.08])
                np.testing.assert_allclose(ref.times, a.times)
                np.testing.assert_allclose(time_grid([a, b]), a.times)
                self.assertEqual([b.frame_at(t) for t in (0.0, 0.0799, 0.08 - 1e-12, 1.0)], [0, 0, 1, 1])
                self.assertEqual(scene.colliders.tolist(), [8])
                self.assertEqual(sorted(set(scene.bodies.tolist())), [0, 1])
                self.assertEqual(len(scene.faces), 8)
                np.testing.assert_allclose(a.speeds(0), 0.0)
                speeds = a.speeds(2)
                np.testing.assert_allclose(speeds[:8], 0.1 / 0.02)
                np.testing.assert_allclose(speeds[8], 0.0)
                self.assertEqual(a.status(1), ["2 solves: converged 1, both unmet 1",
                                               "  outer max 10 / 10, residual / tol 3, penetration / tol 0.2"])
                self.assertEqual(a.status(2), ["no solves"])
                self.assertEqual(a.status(0), [])  # no solve_status group
                # The distances to the reference (at rest): every body node moved by 0.1 k; the collider is excluded
                viewer = Viewer(scene, [a, b], ref)
                self.assertAlmostEqual(viewer.frame_rms(0, 3), 0.3)
                while not viewer.fill_rms(budget=1.0):
                    pass
                np.testing.assert_allclose(viewer.rms[0], 0.1 * np.arange(5))
                np.testing.assert_allclose(viewer.rms[1], [0.4, 0.8])
                viewer.set_time(0.05)
                self.assertEqual((viewer.frame, [tr.frame_at(viewer.t) for tr in (a, b)]), (2, [2, 0]))
            finally:
                for tr in (a, b, ref):
                    tr.close()
            # Masses weight the distance
            x = np.zeros((2, 3))
            self.assertAlmostEqual(mass_rms(x + [[1, 0, 0], [0, 0, 0]], x, np.array([3.0, 1.0])), np.sqrt(0.75))
            self.assertEqual(mass_rms(x, x, np.zeros(2)), 0.0)

    def test_scene_mismatch(self):
        import tempfile

        X, E = self.scene()
        with tempfile.TemporaryDirectory() as root:
            paths = self.files(root)
            fewer, moved = os.path.join(root, "fewer.h5"), os.path.join(root, "moved.h5")
            self.write(fewer, X[:8], E, [(0, X[:8])])
            self.write(moved, X + [0.0, 0.0, 0.01], E, [(0, X)])
            with self.assertRaisesRegex(ValueError, "fewer.h5 has 8 nodes"):
                load([paths[0], fewer])
            with self.assertRaisesRegex(ValueError, "moved.h5's rest positions differ"):
                load([paths[0]], reference=moved)
            with self.assertRaisesRegex(ValueError, "labels"):
                load(paths[:2], labels=["x"])
            # The same solver twice: the file names
            _, trs, _ = load([paths[0], paths[0]])
            self.assertEqual([tr.label for tr in trs], ["0: a", "1: a"])
            for tr in trs:
                tr.close()

    def test_gui_stubbed(self):
        """Every panel and color mode with polyscope stubbed: balanced IDs, unique widget labels per ID scope."""
        import sys
        import tempfile
        from unittest import mock

        stack, labels = [], []

        def widget(label, *args, **kwargs):
            labels.append((tuple(stack), label))
            return False, (args[0] if args else None)

        imgui = mock.MagicMock()
        imgui.PushID.side_effect = lambda i: stack.append(i)
        imgui.PopID.side_effect = lambda: stack.pop()
        for name in ("Checkbox", "SliderInt", "SliderFloat", "InputFloat", "Combo"):
            getattr(imgui, name).side_effect = widget
        imgui.Button.side_effect = lambda label, *a: labels.append((tuple(stack), label)) or False
        imgui.GetWindowWidth.return_value = 400.0
        imgui.GetContentRegionAvail.return_value = (400.0, 300.0)
        imgui.GetCursorScreenPos.return_value = (0.0, 0.0)
        imgui.CalcTextSize.return_value = (20.0, 10.0)
        ps = mock.MagicMock()
        ps.imgui = imgui
        ps.register_surface_mesh.side_effect = lambda *a, **k: mock.MagicMock()
        ps.register_point_cloud.side_effect = lambda *a, **k: mock.MagicMock()
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(sys.modules, {"polyscope": ps,
                                                                                  "polyscope.imgui": imgui}):
            paths = self.files(root)
            scene, trs, ref = load(paths[:2], reference=paths[2])
            try:
                viewer = Viewer(scene, trs, ref)
                viewer.register()
                self.assertEqual(ps.register_point_cloud.call_count, 2)  # the collider node, at each copy
                # The scene extents are the bodies' (not the collider's), over the row of copies
                ps.set_automatically_compute_scene_extents.assert_called_with(False)
                lo, hi = ps.set_bounding_box.call_args.args
                bodies = scene.X[scene.body_nodes]
                np.testing.assert_allclose(lo, bodies.min(axis=0))
                np.testing.assert_allclose(hi, bodies.max(axis=0) + viewer.offset(len(trs) - 1))
                viewer.callback()
                for points in viewer.points:  # translucent colliders
                    points.set_transparency.assert_called_with(viewer.collider_opacity)
                self.assertEqual(viewer.collider_opacity, 0.3)
                for overlay in (False, True):
                    viewer.overlay = overlay
                    for mode in range(len(COLOR_MODES)):
                        viewer.color_mode, viewer.dirty = mode, True
                        labels.clear()
                        viewer.callback()
                        self.assertEqual(stack, [])
                        self.assertEqual(len(labels), len(set(labels)), labels)
                viewer.overlay = False
                viewer.color_mode = 0
                viewer.dirty = True
                viewer.set_time(0.04)
                viewer.callback()
                values = viewer.meshes[0].add_scalar_quantity.call_args.args[1]
                np.testing.assert_allclose(values, 0.1 / 0.02)
                self.assertEqual(viewer.vmax, 10.0)  # the fixed max
                # The second copy sits one spaced scene width to the right
                shift = viewer.meshes[1].update_vertex_positions.call_args.args[0][0] - (
                    trs[1].positions(trs[1].frame_at(viewer.t))[scene.vertices[0]])
                np.testing.assert_allclose(shift, [viewer.spacing * scene.width, 0.0, 0.0])
                viewer.play, viewer.clock_time = True, time.time() - 1.0  # past the end: back to the start
                viewer.callback()
                self.assertEqual(viewer.frame, 0)
                # Without colliders, polyscope keeps its automatic scene extents
                ps.set_automatically_compute_scene_extents.reset_mock()
                ps.set_bounding_box.reset_mock()
                plain = Viewer(replace(scene, colliders=np.zeros(0, dtype=np.int64)), trs, ref)
                plain.register()
                plain.callback()
                ps.set_automatically_compute_scene_extents.assert_not_called()
                ps.set_bounding_box.assert_not_called()
                self.assertEqual(plain.points, [])
            finally:
                for tr in (*trs, ref):
                    tr.close()
        self.assertEqual(imgui.PushID.call_count, imgui.PopID.call_count)


if __name__ == "__main__":
    main()
