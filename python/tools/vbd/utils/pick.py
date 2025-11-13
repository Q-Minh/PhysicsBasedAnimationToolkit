import polyscope as ps
import polyscope.imgui as imgui
from enum import Enum
import numpy as np

class SelectionTargets(Enum):
    VERTEX = 1
    CELL = 2

class BoxSelection:
    """Class to manage box selection of vertices in Polyscope."""
    def __init__(self, name: str, target: SelectionTargets = SelectionTargets.VERTEX):
        self.name = name
        self.target = target
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
        self.Y = 1e6 # Young's modulus (cell selection)
        self.scale = [1.0, 1.0, 1.0]
        self.pos = [0.0, 0.0, 0.0]
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
        transformed_verts = self.transformed_vertices(self.vertices)* self.scale  + self.pos
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


# def select_one(hit_pos, name, cur_dirichlet):
#   pick_result = ps.pick(screen_coords=hit_pos)
#   #print(pick_result.is_hit, pick_result.structure_name == vm.get_name(), pick_result.structure_data['element_type'])
#   if pick_result.is_hit and pick_result.structure_name == name and pick_result.structure_data['element_type'] == "vertex":
#       # print(pick_result)
#       i = pick_result.local_index
#       cur_dirichlet, _ = update_vdbc(cur_dirichlet, i)
#   return cur_dirichlet
