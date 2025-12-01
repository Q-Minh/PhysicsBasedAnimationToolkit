# type: ignore

import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
from pbatoolkit import pbat
import h5py as h5
import typing
import gc


class Trajectory:
    _archive: pbat.io.Archive
    _t: int
    _n_frames: int
    _dirty: bool
    _group: str
    _tmin: int
    _tmax: int

    def __init__(self):
        self._archive = None
        self._t = 0
        self._n_frames = 0
        self._dirty = False
        self._group = "sim"
        self._tmin = 0
        self._tmax = 0

    def draw(self):
        imgui.PushID("Trajectory")
        button_size = [imgui.GetWindowWidth() / 2.1, 0]
        if imgui.Button("Load Trajectory", button_size):
            self._load_trajectory()
        imgui.SameLine()
        imgui.SetNextItemWidth(button_size[0] * 0.8)
        _, self._group = imgui.InputText("Group", self._group)
        if self._archive is not None:
            # TODO: Display trajectory file path
            _, t = imgui.SliderInt("Frame", self._t, self._tmin, self._tmax)
            if t != self._t:
                self._t = t
                self._dirty = True
        imgui.PopID()

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def t(self) -> int:
        return self._t

    @t.setter
    def t(self, value: int):
        self._t = min(max(value, self._tmin), self._tmax)

    def undirty(self, callback: typing.Callable[[pbat.io.Archive], None]):
        if self._dirty:
            callback(self._archive.get(f"{self._group}/{self._t:08d}"))
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
            self._archive = pbat.io.Archive(
                file_path, flags=pbat.io.AccessMode.ReadOnly
            )
            gc.collect()
            self._t = self._tmin
            self._dirty = True
        except Exception as e:
            ps.error(f"Failed to load trajectory file: {e}")
        finally:
            root.destroy()
