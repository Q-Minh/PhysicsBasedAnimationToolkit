import numpy as np
import polyscope as ps

class MyData1:
    m: float = 1e3
    name: str = "Melissa"

class SceneMesh:
    def __init__(self, name: str, V: np.ndarray, C: np.ndarray):
        self.name = name
        self.V = V  # (n,3)
        self.C = C  # (m,4)
        # Per-mesh settings (future extension: heterogeneous materials, loads, constraints)
        self.Y_default = 1e8
        self.nu_default = 0.45
        self.rho_default = 1e3
        self.Y = np.full((C.shape[0]), self.Y_default)  # Young's modulus per element
        self.nu = np.full((C.shape[0]), self.nu_default)  # Poisson's ratio per element
        self.rho = np.full((C.shape[0]), self.rho_default)  # Density per element
        self.v0 = np.array([0.0, 0.0, 0.0])
        self.b = np.array(
            [0.0, 0.0, 0.0]
        )  # per-mesh body force (e.g. wind, extra load)
        #self.dirichlet_indices = {}  # Array of vertex indices with Dirichlet boundary conditions
        self.handle = ps.register_volume_mesh(name, V, C)
        self.handle.add_scalar_quantity("Young's modulus", self.Y, defined_on='cells',
                           cmap='blues', enabled=False)
        self.handle.add_scalar_quantity("Poisson ratio", self.nu, defined_on='cells',
                           cmap='reds', enabled=False)
        self.handle.add_scalar_quantity("Density", self.rho, defined_on='cells',
                           cmap='gray', enabled=False)

    def transformed_vertices(self) -> np.ndarray:
        T = self.handle.get_transform()
        VH = np.vstack([self.V.T, np.ones((1, self.V.shape[0]))])
        VT = (T @ VH).T[:, :3]
        return VT

