import unittest

import numpy as np
from pbatoolkit import pbat

import modalrom


class TestElasticEnergyFloat64(unittest.TestCase):
    def setUp(self):
        # Two tetrahedra sharing a face, with heterogeneous material
        V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        C = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        element = pbat.fem.Element.Tetrahedron
        self.X, self.E = pbat.fem.mesh(V.T, C.T, element=element)
        egU, wgU, GNegU, mug, lambdag = modalrom.potential(
            self.E, self.X, element, np.array([1e6, 3e6]), np.array([0.3, 0.45])
        )
        args = (self.E, self.X, egU, wgU, GNegU, mug, lambdag, pbat.fem.HyperElasticEnergy.StableNeoHookean, element)
        self.psi32 = modalrom.elastic_energy(*args)
        self.psi64 = modalrom.elastic_energy_float64(*args)
        rng = np.random.default_rng(0)
        self.x = self.X.flatten(order="F").astype(np.float64) + 0.1 * rng.standard_normal(self.X.size)

    def test_matches_pbat(self):
        e64, g64 = self.psi64(self.x, True)
        e32, g32 = self.psi32(self.x, True)
        self.assertAlmostEqual(e64 / e32, 1.0, delta=1e-5)
        np.testing.assert_allclose(g64, g32, rtol=1e-4, atol=1e-4 * np.abs(g64).max())

    def test_gradient_matches_finite_differences(self):
        _, g = self.psi64(self.x, True)
        h = 1e-6
        fd = np.array([(self.psi64(self.x + h * ei) - self.psi64(self.x - h * ei)) / (2 * h) for ei in np.eye(self.x.size)])
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6 * np.abs(g).max())

    def test_hessian_comes_from_pbat(self):
        _, _, H = self.psi64(self.x, True, True)
        _, _, H32 = self.psi32(self.x, True, True)
        np.testing.assert_array_equal(H.toarray(), H32.toarray())


if __name__ == "__main__":
    unittest.main()
