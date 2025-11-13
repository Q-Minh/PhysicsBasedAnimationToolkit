import numpy as np
import polyscope as ps

class SceneMesh:
    def __init__(self, name: str, V: np.ndarray, C: np.ndarray):
        self.name = name
        self.V = V  # (n,3)
        self.C = C  # (m,4)
        # Per-mesh settings (future extension: heterogeneous materials, loads, constraints)
        self.Y = np.full((C.shape[0]), 1e6)  # Young's modulus per element
        self.nu = 0.45
        self.rho = 1e3
        self.v0 = np.array([0.0, 0.0, 0.0])
        self.b = np.array(
            [0.0, 0.0, 0.0]
        )  # per-mesh body force (e.g. wind, extra load)
        #self.dirichlet_indices = {}  # Array of vertex indices with Dirichlet boundary conditions
        self.handle = ps.register_volume_mesh(name, V, C)
        self.handle.add_scalar_quantity("Young's modulus", self.Y, defined_on='cells',
                           cmap='blues', enabled=True)

    def transformed_vertices(self) -> np.ndarray:
        T = self.handle.get_transform()
        VH = np.vstack([self.V.T, np.ones((1, self.V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT

