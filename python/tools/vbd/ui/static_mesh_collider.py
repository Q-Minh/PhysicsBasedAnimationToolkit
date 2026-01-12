# type: ignore

import polyscope as ps
import polyscope.imgui as imgui
import numpy as np
from .utils.ps_helper import PsHelper
from .utils import styles
import h5py as h5


class StaticMeshCollider:
    """Class representing a static mesh collider in the scene."""

    name: str
    _V: np.ndarray
    _F: np.ndarray
    _sm: ps.SurfaceMesh
    _ps_helper: PsHelper

    def __init__(self):
        self.name = None
        self._V = None
        self._F = None
        self._sm = None
        self._ps_helper = None

    def draw(self):
        imgui.PushID(self.name)
        tab_flags = styles.default_tab_flags()
        styles.set_style_subtle()
        if imgui.BeginTabBar("Mesh options", tab_flags):
            self._ps_helper.draw()
            if imgui.BeginTabItem("Hide", True, tab_flags)[0]:
                # This is intentionally left empty
                imgui.EndTabItem()
            imgui.EndTabBar()
        styles.pop_most_recent_style()
        imgui.PopID()

    def set_visible(self, visible: bool):
        if self._sm is not None:
            self._sm.set_enabled(visible)

    def on_mesh_added(self, name: str, V: np.ndarray, F: np.ndarray):
        self.name = name
        self._V = V
        self._F = F
        self._sm = ps.register_surface_mesh(
            name,
            self._V,
            self._F,
        )
        self._sm.set_transform(np.eye(4))
        self._ps_helper = PsHelper(self._sm)

    def on_mesh_removed(self):
        if self._sm is not None:
            ps.remove_surface_mesh(self._sm.get_name())
            self._sm = None

    def serialize(self, grp: h5.Group):
        grp = grp.create_group("tools.vbd.ui.StaticMeshCollider")
        grp.attrs["name"] = self.name
        grp["V"] = self.VT
        grp["F"] = self._F

    def deserialize(self, grp: h5.Group, headless: bool = False):
        grp = grp["tools.vbd.ui.StaticMeshCollider"]
        self.name = grp.attrs["name"]
        self._V = grp["V"][:]
        self._F = grp["F"][:]
        if not headless:
            self.on_mesh_added(self.name, self._V, self._F)

    @property
    def F(self) -> np.ndarray:
        return self._F

    @property
    def VT(self) -> np.ndarray:
        if self._V is None:
            return None
        if self._sm is None:
            return self._V
        T = self._sm.get_transform()
        VH = np.vstack([self._V.T, np.ones((1, self._V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT
