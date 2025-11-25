import polyscope as ps
import polyscope.imgui as imgui
from enum import Enum
import numpy as np
from scene_mesh import SceneMesh

class SelectionTargets(Enum):
    VERTEX = 1
    CELL = 2

class BoxSelection:
    """Class to manage box selection of vertices in Polyscope."""
    def __init__(self, name: str, callback, target: SelectionTargets = SelectionTargets.VERTEX):
        self.name = name
        self.target = target
        self.callback = callback
        # Cube vertices
        self.vertices = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ]) * 5
        # Cube faces
        self.faces = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 3, 7, 4],
            [1, 2, 6, 5],
        ])
        self.scale = [1.0, 1.0, 1.0]
        self.ps_mesh = ps.register_surface_mesh(name, self.vertices, self.faces)
        self.ps_mesh.set_transparency(0.5)
        
    def transformed_vertices(self, V) -> np.ndarray:
        T = self.ps_mesh.get_transform()
        VH = np.vstack([V.T, np.ones((1, V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT

    def inside_test(self, verts: np.ndarray, cells: np.ndarray):
        """Test which structures are inside the box defined by the cube mesh."""
        # Get axis-aligned bounding box of the cube
        transformed_verts = self.transformed_vertices(self.vertices * self.scale)
        min_bounds = np.min(transformed_verts, axis=0)
        max_bounds = np.max(transformed_verts, axis=0)

        if self.target == SelectionTargets.VERTEX:
            # Test vertices
            inside = np.all((verts >= min_bounds) & (verts <= max_bounds), axis=1)
            indices = np.where(inside)[0]
            return indices
        if self.target == SelectionTargets.CELL:
            # Test cell centroids
            centroids = np.mean(verts[cells], axis=1)
            inside = np.all((centroids >= min_bounds) & (centroids <= max_bounds), axis=1)
            indices = np.where(inside)[0]
            return indices

    def specific_draw(self):
        raise NotImplementedError("specific_draw must be implemented in subclasses")

    def draw(self, meshes: list[SceneMesh]):
        imgui.PushID(self.name)
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        
        _, self.scale = imgui.SliderFloat3("Size", self.scale, 0, 10)

        self.ps_mesh.update_vertex_positions(self.vertices * self.scale)
        # Input field for specific property that we're manipulating
        self.specific_draw()
        if imgui.Button("Apply Box Selection", default_button_size):
            for m in meshes:
                VT = m.transformed_vertices()
                C = m.C
                # Get indices inside box
                indices = self.inside_test(VT, C)
                # Apply in callback that depends on property that we selected
                self.callback(m, indices)
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
                        "Picked Group", self.transform_index,
                        [t.name for t in self.transform_library.transforms]
                    )
        

    def callback(self, mesh, indices):
        # Vertices only affected by Dirichlet
        transform = self.transform_library.transforms[self.transform_index]
        dgroup = self.dirichlet_library[transform.name].get_mesh_indices(mesh)
        for i in indices:
            dgroup.update_indices(i)
        self.dirichlet_library[transform.name].build_point_cloud()


class CellSelection(BoxSelection):
    def __init__(self, name, stored, property_tag, mesh_property_name):
        super().__init__(name, self.callback, SelectionTargets.CELL)
        self.stored = stored
        self.property_tag = property_tag
        self.mesh_property_name = mesh_property_name

    def specific_draw(self):
        _, self.stored = imgui.InputFloat(self.property_tag, self.stored)

    def callback(self, mesh, indices):
        arr = getattr(mesh, self.mesh_property_name)
        arr[indices] = self.stored
        mesh.handle.add_scalar_quantity(self.property_tag, arr, defined_on='cells')


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
        #mesh.handle.add_scalar_quantity(self.property_tag, arr, defined_on='nodes')

class YoungSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name, 1e9, "Young's modulus", "Y")


class RhoSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name, 1e3, "MassDensity", "rho")
    

class NuSelection(CellSelection):

    def __init__(self, name):
        super().__init__(name, 0.45, "Poisson ratio", "nu")


class V0Selection(VertexSelection):

    def __init__(self, name):
        super().__init__(name, np.array[0,0,0], "Initial velocity", "v0")