# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from enum import Enum
from igl import boundary_facets
import numpy as np
from ..tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
from . import styles
from .ps_helper import PsHelper
import typing


class SelectionTargets(Enum):
    VERTEX = 1
    CELL = 2


class Selection:
    name: str
    _prop_name: str
    _prop_value: typing.Any
    _callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]]

    def __init__(
        self,
        name: str,
        prop_name: str,
        prop_value: typing.Any,
        target: SelectionTargets,
        callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]],
    ):
        self.name = name
        self._prop_name = prop_name
        self._prop_value = prop_value
        self._target = target
        self._callback = callback

    @classmethod
    def get_subclasses(self):
        return [cls.__name__ for cls in Selection.__subclasses__()]


    def on_added(self):
        pass


    def on_removed(self):
        pass

    def specific_draw(self):
        if isinstance(self._prop_value, float):
            _, self._prop_value = imgui.InputFloat(self._prop_name, self._prop_value)
        elif isinstance(self._prop_value, int):
            _, self._prop_value = imgui.InputInt(self._prop_name, self._prop_value)
        elif (
            isinstance(self._prop_value, np.ndarray)
            and self._prop_value.shape[0] == 3
            and type(self._prop_value[0]) in [float, np.float32, np.float64]
        ):
            _, self._prop_value = imgui.InputFloat3(self._prop_name, self._prop_value)
            self._prop_value = np.array(self._prop_value)

    def draw(self):
        pass

    def set_visible(self, visible: bool):
        pass

class BoxSelection(Selection):
    """Class to manage box selection of vertices or cells in Polyscope."""

    name: str
    _prop_name: str
    _prop_value: typing.Any
    _target: SelectionTargets
    _callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]]
    _vertices: np.ndarray
    _faces: np.ndarray
    _min: np.ndarray
    _max: np.ndarray
    _scale: np.ndarray
    _surface_only: bool
    _ps_mesh: ps.SurfaceMesh
    _ps_helper: PsHelper

    def __init__(
        self,
        name: str,
        prop_name: str,
        prop_value: typing.Any,
        target: SelectionTargets,
        callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]],
    ):
        super().__init__(name, prop_name, prop_value, target, callback)

        # Cube vertices
        self._vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        self._min = np.min(self._vertices, axis=0)
        self._max = np.max(self._vertices, axis=0)
        # Cube faces
        self._faces = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 7, 4],
                [1, 2, 6, 5],
            ]
        )
        self._scale = np.ones(3, dtype=np.float32)
        self._surface_only = False
        

    def on_added(self):
        self._ps_mesh = ps.register_surface_mesh(self.name, self._vertices, self._faces)
        self._ps_mesh.set_transparency(0.5)
        self._ps_helper = PsHelper(self._ps_mesh)


    def on_removed(self):
        ps.remove_surface_mesh(self._ps_mesh.get_name())
        self._ps_mesh = None
        self._ps_helper = None


    def _inverse_transform_points(self, V) -> np.ndarray:
        T = self._ps_mesh.get_transform()
        Tinv = np.diag(
            [1 / self._scale[0], 1 / self._scale[1], 1 / self._scale[2], 1]
        ) @ np.linalg.inv(T)
        VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
        VT = (Tinv @ VH).T[:, :3]
        return VT

    def inside_test(self, verts: np.ndarray, cells: np.ndarray):
        """Test which structures are inside the box defined by the cube mesh."""
        # Get axis-aligned bounding box of the cube
        VT = self._inverse_transform_points(verts)
        if self._target == SelectionTargets.VERTEX:
            # Test vertices
            inside = np.all((VT >= self._min) & (VT <= self._max), axis=1)
            indices = np.where(inside)[0]
            if self._surface_only:
                boundary_indices = np.unique(boundary_facets(cells))
                indices = np.intersect1d(boundary_indices, indices)
            return indices
        if self._target == SelectionTargets.CELL:
            # Test cell centroids
            centroids = np.mean(VT[cells], axis=1)
            inside = np.all((centroids >= self._min) & (centroids <= self._max), axis=1)
            indices = np.where(inside)[0]
            return indices

    def draw(self, meshes: list[TetrahedralElastodynamicsBody]):
        imgui.PushID(self.name)
        tab_flags = styles.default_tab_flags()
        styles.set_style_subtle()
        if imgui.BeginTabBar("Mode bar", tab_flags):
            if imgui.BeginTabItem("Setup", True, tab_flags)[0]:
                _, self._scale = imgui.SliderFloat3("Size", self._scale, 0, 2)
                self._scale = np.array(self._scale)
                self._ps_mesh.update_vertex_positions(self._vertices * self._scale)
                # Input field for specific property that we're manipulating
                self.specific_draw()
                if self._target == SelectionTargets.VERTEX:
                    _, self._surface_only = imgui.Checkbox("Surface Only", self._surface_only)
                imgui.EndTabItem()

            self._ps_helper.draw()

            if imgui.BeginTabItem("Hide", True, tab_flags)[0]:
                # This is intentionally left empty
                imgui.EndTabItem()
            imgui.EndTabBar()
        styles.pop_most_recent_style()
        if imgui.Button("Apply", styles.default_button_size()):
            for b, m in enumerate(meshes):
                VT, C = m.VT, m.T
                # Get indices inside box
                indices = self.inside_test(VT, C)
                # Apply in callback that depends on property that we selected
                self._callback(b, self._prop_value, indices)
        
        imgui.PopID()

    def set_visible(self, visible: bool):
        if self._ps_mesh is not None:
            self._ps_mesh.set_enabled(visible and self._ps_helper.get_show_mesh())
            self._ps_mesh.set_transform_gizmo_enabled(visible and self._ps_helper.get_show_gizmo())



class RegionSelection(Selection):
    """Class to manage selection of regions (defined over cells) in Polyscope."""

    name: str
    _prop_name: str
    _prop_value: typing.Any
    _region_selection: dict[str,str]
    _callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]]

    def __init__(
        self,
        name: str,
        prop_name: str,
        prop_value: typing.Any,
        callback: typing.Callable[None, [int, typing.Any, np.ndarray[int]]],
    ):
        super().__init__(name, prop_name, prop_value, SelectionTargets.CELL, callback)
        self._region_selection = {}
        

    def region_test(self, cells: np.ndarray, regions: np.ndarray, region_selection: str):
        """Test which cells are inside the regions that have been selected."""
        if region_selection == "":
            return np.zeros_like(regions)
        if region_selection == "---":
            return np.ones_like(regions, dtype=bool)
        region_list = region_selection.split("-")
        region_list = [int(value) for value in region_list if value.isdigit() ]
        region_list = np.array(region_list)
        
        
        indices = np.isin(regions, region_list)
        return indices


    def draw(self, meshes: list[TetrahedralElastodynamicsBody]):
        imgui.PushID(self.name)
        tab_flags = styles.default_tab_flags()
        
        styles.set_style_subtle()
        if imgui.BeginTabBar("Mode bar", tab_flags):
            if imgui.BeginTabItem("Setup", True, tab_flags)[0]:
                # Input field for specific property that we're manipulating
                self.specific_draw()
                for mesh in meshes:
                    if self._region_selection.get(mesh.name) is None:
                        self._region_selection[mesh.name] = ""
                for mesh in meshes:
                    if imgui.TreeNode(mesh.name):
                        _, self._region_selection[mesh.name] = imgui.InputText("Region1-R2-R3-...", self._region_selection[mesh.name], imgui.ImGuiInputTextFlags_CharsDecimal)
                        imgui.TreePop()
                imgui.EndTabItem()

            if imgui.BeginTabItem("Hide", True, tab_flags)[0]:
                # This is intentionally left empty
                imgui.EndTabItem()
            imgui.EndTabBar()
        styles.pop_most_recent_style()

        if imgui.Button("Apply", styles.default_button_size()):
            for b, m in enumerate(meshes):
                C, R = m.T, m.R
                regions = self._region_selection[mesh.name]
                # Get indices inside box
                indices = self.region_test(C, R, regions)
                # Apply in callback that depends on property that we selected
                self._callback(b, self._prop_value, indices)
        imgui.SameLine()
        if imgui.Button("Clear All", styles.half_button_size()):
            for mesh_name in self._region_selection:
                self._region_selection[mesh_name] = ""
                
        imgui.PopID()

    