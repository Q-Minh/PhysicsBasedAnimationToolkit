# type: ignore

import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
from pbatoolkit import pbat
import h5py as h5
import typing
import gc
import time
import numpy as np

# C++ FemElastoDynamics archive group of a frame in the CPU simulators' trajectories
_FEM_GROUP = "pbat.sim.dynamics.FemElastoDynamics"
_WARNING_COLOR = (1.0, 0.75, 0.2, 1.0)


class Trajectory:
    """Plays a trajectory file frame by frame: a CPU simulator's, where each frame is a C++
    FemElastoDynamics archive, or a field trajectory, where each frame holds some of the FEM's nodal fields
    as |# nodes| x 3 datasets (python.tools.sim.simulate and its GUI's Serialization write ``x`` only) and
    the FEM is stored once, in the root ``FemElastoDynamics`` group.
    """

    _archive: pbat.io.Archive
    _path: str | None
    _field_frames: bool
    _rest_positions: np.ndarray | None
    _n_element_nodes: int
    _t: int
    _n_frames: int
    _dirty: bool
    _group: str
    _tmin: int
    _tmax: int
    _autoplay: bool
    _dt: float
    _screenshot: bool
    _clock_time: float

    def __init__(self):
        self._archive = None
        self._path = None
        self._field_frames = False
        self._rest_positions = None
        self._n_element_nodes = 0
        self._t = 0
        self._n_frames = 0
        self._dirty = False
        self._group = "sim"
        self._tmin = 0
        self._tmax = 0
        self._autoplay = False
        self._dt = 1e-2
        self._clock_time = time.time()
        self._screenshot = False

    def draw(self):
        imgui.PushID("Trajectory")
        width = imgui.GetWindowWidth()
        button_size = [width / 2.1, 0]
        if imgui.Button("Load Trajectory", button_size):
            self._load_trajectory()
        imgui.SameLine()
        imgui.SetNextItemWidth(button_size[0] * 0.8)
        _, self._group = imgui.InputText("Group", self._group)
        if self._path is not None:
            imgui.SetNextItemWidth(width * 0.7)
            _, t = imgui.SliderInt("Frame", self._t, self._tmin, self._tmax)
            imgui.SameLine()
            imgui.SetNextItemWidth(width * 0.2)
            sync = imgui.Button("Sync")
            autoplay_changed, self._autoplay = imgui.Checkbox(
                "Autoplay", self._autoplay
            )
            imgui.SameLine()
            _, self._screenshot = imgui.Checkbox("Screenshot", self._screenshot)
            if self._field_frames:
                imgui.PushStyleColor(imgui.ImGuiCol_Text, _WARNING_COLOR)
                imgui.TextWrapped(
                    "Partial frames: only some FemElastoDynamics members are stored. The others are left "
                    "as they are, except velocities, which are zero when absent."
                )
                imgui.PopStyleColor()

            if t != self._t or sync:
                self._t = t
                self._dirty = True
            if self._autoplay:
                self._dirty = self._t < self._tmax or self._dirty
                if autoplay_changed:
                    self._clock_time = time.time()
                else:
                    now = time.time()
                    elapsed = now - self._clock_time
                    self._clock_time = now
                    if self._screenshot:
                        ps.screenshot("{:08d}.png".format(self._t))
                        n_frames_advance = 1
                    else:
                        n_frames_advance = round(elapsed / self._dt)
                    self._t = min(self._t + n_frames_advance, self._tmax)


        imgui.PopID()

    def set_timestep(self, dt: float):
        self._dt = dt

    @property
    def rest_positions(self) -> np.ndarray | None:
        """|# nodes| x 3 rest positions of a field trajectory's FEM (None otherwise, or if the file has no
        root FemElastoDynamics group). The GPU simulators' nodes are the bodies' nodes, in scene order, then
        the static colliders' (element-less) nodes."""
        return self._rest_positions

    @property
    def n_element_nodes(self) -> int:
        """1 + the largest node index of a field trajectory's elements (0 if unknown)."""
        return self._n_element_nodes

    def close(self):
        """Forget the loaded trajectory."""
        self._archive = None
        self._path = None
        self._field_frames = False
        self._rest_positions = None
        self._n_element_nodes = 0
        self._dirty = False
        self._autoplay = False
        gc.collect()

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def t(self) -> int:
        return self._t

    @t.setter
    def t(self, value: int):
        self._t = min(max(value, self._tmin), self._tmax)

    def undirty(
        self,
        callback: typing.Callable[[pbat.io.Archive], None],
        fields_callback: typing.Callable[[dict[str, np.ndarray]], None] | None = None,
    ):
        """Load the current frame: pass its archive to `callback`, or, for a field trajectory, all of its
        datasets by name to `fields_callback` (subgroups, e.g. ``solve_status``, are skipped)."""
        if self._dirty:
            frame = f"{self._group}/{self._t:08d}"
            if self._field_frames:
                with h5.File(self._path, "r") as f:
                    fields = {
                        name: obj[()]
                        for name, obj in f[frame].items()
                        if isinstance(obj, h5.Dataset)
                    }
                if fields_callback is not None:
                    fields_callback(fields)
            else:
                callback(self._archive.get(frame))
            self._dirty = False

    def _load_trajectory(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Trajectory File",
            filetypes=[
                ("HDF5 simulation trajectory files", "*.h5"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            with h5.File(file_path, "r") as f:
                # Collect child group names under the selected parent group
                child_names = [
                    name
                    for name, obj in f[self._group].items()
                    if isinstance(obj, h5.Group)
                ]
                frame_indices: list[int] = [int(name) for name in child_names]
                if len(frame_indices) == 0:
                    raise RuntimeError(
                        f"No integer-named frame groups found under '{self._group}'"
                    )
                self._tmin = min(frame_indices)
                self._tmax = max(frame_indices)
                self._n_frames = self._tmax - self._tmin + 1
                first = f[self._group][f"{self._tmin:08d}"]
                field_frames = _FEM_GROUP not in first
                rest_positions, n_element_nodes = None, 0
                if field_frames and "FemElastoDynamics/X" in f:
                    rest_positions = f["FemElastoDynamics/X"][()]
                    n_element_nodes = int(f["FemElastoDynamics/E"][()].max()) + 1
            self._field_frames = field_frames
            self._rest_positions = rest_positions
            self._n_element_nodes = n_element_nodes
            self._archive = (
                None
                if field_frames
                else pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
            )
            self._path = file_path
            gc.collect()
            self._t = self._tmin
            self._dirty = True
        except Exception as e:
            ps.error(f"Failed to load trajectory file: {e}")
        finally:
            root.destroy()
