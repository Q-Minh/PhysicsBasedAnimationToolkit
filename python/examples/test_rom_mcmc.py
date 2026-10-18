import unittest

import numpy as np

import rom_mcmc


class QuadraticEnergy:
    """Psi_hat(q) = 1/2 q^T diag(a) q. Its level set is an ellipsoid, and under the "uniform"
    weight z = sqrt(a) * q is uniformly distributed on a sphere."""

    def __init__(self, a: np.ndarray, scale: np.ndarray):
        self.a = a
        self.scale = scale
        self.dims = a.shape[0]
        self.weight = "uniform"

    def energy(self, q):
        return 0.5 * q @ (self.a * q)

    def energy_and_gradient(self, q):
        return self.energy(q), self.a * q

    def log_weight(self, q, gq):
        return 0.0


class TestRadialRoot(unittest.TestCase):
    def test_hits_level_set(self):
        model = QuadraticEnergy(np.array([1.0, 4.0, 9.0]), np.ones(3))
        rng = np.random.default_rng(0)
        for r0 in [1e-3, 1.0, 1e3]:
            d = rng.standard_normal(3)
            r, monotone = rom_mcmc.radial_root(model, d, 2.5, r0)
            self.assertTrue(monotone)
            self.assertAlmostEqual(model.energy(r * d), 2.5, delta=1e-5 * 2.5)


class TestSamplers(unittest.TestCase):
    def _check_uniform_on_sphere(self, kind):
        a = np.array([1.0, 4.0, 9.0])
        # Deliberately not whitened, so the sampler has to correct for the anisotropy
        model = QuadraticEnergy(a, np.ones(3))
        sampler = rom_mcmc.make_sampler(kind, model, 1.0, np.random.default_rng(1))
        batch = rom_mcmc.run_chain(sampler, n_samples=3000, burn_in=500, pilot=1000)
        np.testing.assert_allclose(batch.energy, 1.0, rtol=1e-4)
        z = np.sqrt(a) * batch.q
        z2 = (z / np.linalg.norm(z, axis=1, keepdims=True)) ** 2
        # Var((z_i/|z|)^2) = 4/45 for m = 3, so the standard error of the mean is ~0.0055
        np.testing.assert_allclose(z2.mean(axis=0), 1.0 / 3.0, atol=0.03)
        self.assertGreater(batch.stats["acceptance"], 0.1)
        return batch

    def test_polar_metropolis(self):
        batch = self._check_uniform_on_sphere("polar")
        self.assertEqual(batch.stats["star_violations"], 0)

    def test_constrained_random_walk(self):
        self._check_uniform_on_sphere("crw")

    def test_ray_sampler_is_biased(self):
        a = np.array([1.0, 100.0])
        model = QuadraticEnergy(a, np.ones(2))
        sampler = rom_mcmc.make_sampler("ray", model, 1.0, np.random.default_rng(2))
        batch = sampler.sample(4000)
        z = np.sqrt(a) * batch.q
        z2 = (z / np.linalg.norm(z, axis=1, keepdims=True)) ** 2
        # Uniform directions in q concentrate z along the stiff axis
        self.assertGreater(z2[:, 1].mean(), 0.8)


class TestAutocorrelation(unittest.TestCase):
    def test_ar1(self):
        rng = np.random.default_rng(3)
        rho, n = 0.9, 200000
        x = np.empty(n)
        x[0] = 0.0
        for k in range(1, n):
            x[k] = rho * x[k - 1] + rng.standard_normal()
        tau = rom_mcmc.integrated_autocorrelation_time(x)
        self.assertAlmostEqual(tau, (1 + rho) / (1 - rho), delta=1.5)

    def test_white_noise(self):
        x = np.random.default_rng(4).standard_normal(10000)
        self.assertLess(rom_mcmc.integrated_autocorrelation_time(x), 1.3)


if __name__ == "__main__":
    unittest.main()
