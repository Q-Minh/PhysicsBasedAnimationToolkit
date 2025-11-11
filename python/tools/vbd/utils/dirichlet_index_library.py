import polyscope as ps
import numpy as np

class DirichletIndices:
    """Class to manage Dirichlet boundary condition indices and their visualization in Polyscope. 
    The indices refer to one specific mesh in the scene AND one specific transform"""
    def __init__(self, V: np.ndarray):
        self.indices = np.array([], dtype=np.int32)
        self.V = V

    def update_vdbc(self, i):
        # Nothing selected yet
        if self.indices is None:
            self.indices = np.array([i], dtype=np.int32)
            return self.indices, self.indices.shape[0]
        # Toggle selection
        found = np.where(self.indices == i)[0]
        if found.shape[0] > 0:
            # Deselect if found
            self.indices = np.delete(self.indices, found)
        else:
            # Select if not found
            self.indices = np.hstack([self.indices, i])
        return self.indices, self.indices.shape[0]

class DirichletLibrary:
    """Class to manage multiple Dirichlet boundary condition groups for a transform."""
    def __init__(self, name: str, scene_meshes = []):
        self.mesh_to_indices = {} # Mesh to DirichletIndices
        self.name = name # Name of the transform
        self.point_cloud = None
        for mesh in scene_meshes:
            self.add_mesh(mesh.name, mesh.V)

    def add_mesh(self, name: str, V: np.ndarray):
        if name not in self.mesh_to_indices:
            self.mesh_to_indices[name] = DirichletIndices(V)

    def get_mesh_indices(self, name: str):
        return self.mesh_to_indices.get(name, None)
    
    def build_point_cloud(self):
        stack = np.array([], dtype=np.int32)
        for mesh in self.mesh_to_indices:
            val = self.mesh_to_indices[mesh]
            positions = val.V[val.indices]
            if positions.shape[0] > 0:
                stack = np.vstack([stack, positions]) if stack.shape[0] > 0 else positions
        if stack.shape[0] == 0:
            if ps.has_point_cloud(self.name):
                ps.remove_point_cloud(self.name)
            self.point_cloud = None
            return
        self.point_cloud = ps.register_point_cloud(self.name, stack)