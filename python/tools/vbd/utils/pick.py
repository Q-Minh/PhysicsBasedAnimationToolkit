# type: ignore
import polyscope as ps
import polyscope.imgui as imgui
from enum import Enum
import numpy as np
from ..ui.tetrahedral_elastodynamics_body import TetrahedralElastodynamicsBody
import typing


class SelectionTargets(Enum):
    VERTEX = 1
    CELL = 2


class BoxSelection:
    """Class to manage box selection of vertices in Polyscope."""

    name: str
    _prop_name: str
    _prop_value: typing.Any
    _target: SelectionTargets
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
        # Cube vertices
        self.vertices = np.array(
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
        self._min = np.min(self.vertices, axis=0)
        self._max = np.max(self.vertices, axis=0)
        # Cube faces
        self.faces = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 7, 4],
                [1, 2, 6, 5],
            ]
        )
        self.scale = np.ones(3, dtype=np.float32)

    def on_added(self):
        self.ps_mesh = ps.register_surface_mesh(self.name, self.vertices, self.faces)
        self.ps_mesh.set_transparency(0.5)

    def on_removed(self):
        ps.remove_surface_mesh(self.ps_mesh.get_name())
        self.ps_mesh = None

    def _inverse_transform_points(self, V) -> np.ndarray:
        T = self.ps_mesh.get_transform()
        Tinv = np.diag(
            [1 / self.scale[0], 1 / self.scale[1], 1 / self.scale[2], 1]
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
            return indices
        if self._target == SelectionTargets.CELL:
            # Test cell centroids
            centroids = np.mean(verts[cells], axis=1)
            inside = np.all((centroids >= self._min) & (centroids <= self._max), axis=1)
            indices = np.where(inside)[0]
            return indices

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

    def draw(self, meshes: list[TetrahedralElastodynamicsBody]):
        imgui.PushID(self.name)
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        _, self.scale = imgui.SliderFloat3("Size", self.scale, 0, 10)
        self.scale = np.array(self.scale)
        self.ps_mesh.update_vertex_positions(self.vertices * self.scale)
        # Input field for specific property that we're manipulating
        self.specific_draw()
        if imgui.Button("Apply", default_button_size):
            for b, m in enumerate(meshes):
                VT, C = m.VT, m.T
                # Get indices inside box
                indices = self.inside_test(VT, C)
                # Apply in callback that depends on property that we selected
                self._callback(b, self._prop_value, indices)
        imgui.PopID()


class DirichletSelection(BoxSelection):

    def __init__(self, name, dirichlet_library, transform_library):
        super().__init__(name, self.callback, SelectionTargets.VERTEX)
        self.transform_index = 0
        self.dirichlet_library = dirichlet_library
        self.transform_library = transform_library

    def specific_draw(self):
        if self.transform_library.transforms:
            _, self.transform_index = imgui.Combo(
                "Picked Group",
                self.transform_index,
                [t.name for t in self.transform_library.transforms],
            )

    def callback(self, mesh, indices):
        # Vertices only affected by Dirichlet
        transform = self.transform_library.transforms[self.transform_index]
        dgroup = self.dirichlet_library[transform.name].get_mesh_indices(mesh)
        for i in indices:
            dgroup.update_indices(i)
        self.dirichlet_library[transform.name].build_point_cloud()


class CellSelection(BoxSelection):
    def __init__(
        self, name, callback: typing.Callable[[int, typing.Any, np.ndarray[int]], None]
    ):
        super().__init__(name, callback, SelectionTargets.CELL)

    def specific_draw(self):
        _, self.stored = imgui.InputFloat(self.property_tag, self.stored)


class VertexSelection(BoxSelection):
    def __init__(self, name, stored, property_tag, mesh_property_name):
        super().__init__(name, self.callback, SelectionTargets.CELL)
        self.stored = stored
        self.property_tag = property_tag
        self.mesh_property_name = mesh_property_name

    def specific_draw(self):
        _, self.stored = imgui.InputFloat3(self.property_tag, self.stored)

    def callback(self, mesh, indices):
        arr = getattr(mesh, self.mesh_property_name)
        arr[indices] = self.stored
        # mesh.handle.add_scalar_quantity(self.property_tag, arr, defined_on='nodes')


class YoungSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name)


class RhoSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name, 1e3, "MassDensity", "rho")


class NuSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name, 0.45, "Poisson ratio", "nu")


class V0Selection(VertexSelection):

    def __init__(self, name):
        super().__init__(name, np.array[0, 0, 0], "Initial velocity", "v0")
