import os
import tempfile
import unittest

import meshio
import numpy as np
from pbatoolkit import pbat

import modalrom


class TestLoadModel(unittest.TestCase):
    def test_mesh_input(self):
        """A tetrahedral mesh file loads with its own material: a lumped (diagonal) mass matrix of the body's mass."""
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
        C = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        volume = 1.0 / 6.0 + abs(np.linalg.det(V[[2, 3, 4]] - V[1])) / 6.0
        rho = 1000.0
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "two_tets.msh")
            meshio.write(path, meshio.Mesh(V, [("tetra", C)]))
            E, X, M, egU, wgU, GNegU, mug, lambdag, freedofs, gravity = modalrom.load_model(
                path, rho, 1e6, 0.45, pbat.fem.Element.Tetrahedron, pbat.fem.HyperElasticEnergy.StableNeoHookean
            )
        self.assertEqual(M.shape, (15, 15))
        self.assertEqual(M.nnz, 15)  # diagonal
        m = M.diagonal()
        np.testing.assert_allclose(m.reshape(-1, 3), np.repeat(m[0::3, None], 3, axis=1))  # same per coordinate
        np.testing.assert_allclose(m[0::3].sum(), rho * volume, rtol=1e-5)
        self.assertEqual(len(freedofs), 15)


if __name__ == "__main__":
    unittest.main()
