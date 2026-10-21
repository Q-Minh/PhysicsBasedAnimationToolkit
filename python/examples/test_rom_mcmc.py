import unittest

import numpy as np

import rom_mcmc


class QuadraticEnergy:
    """Psi_hat(q) = 1/2 q^T diag(a) q. Its level set is an ellipsoid, and under the "uniform"
    weight z = sqrt(a) * q is uniformly distributed on a sphere."""

    def __init__(self, a: np.ndarray, scale: np.ndarray, mass: np.ndarray | None = None):
        self.a = a
        self.scale = scale
        self.dims = a.shape[0]
        self.weight = "uniform"
        self.mass = np.ones(self.dims) if mass is None else mass

    def reduced_mass(self, q):
        return np.diag(self.mass)

    def energy(self, q):
        return 0.5 * q @ (self.a * q)

    def energy_and_gradient(self, q):
        return self.energy(q), self.a * q

    def log_weight(self, q, gq):
        return 0.0


class NoisyQuadraticEnergy(QuadraticEnergy):
    """QuadraticEnergy plus absolute Gaussian noise, like pbat's float32 energy round-off."""

    def __init__(self, a, scale, noise, seed=0):
        super().__init__(a, scale)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def energy(self, q):
        return super().energy(q) + self.noise * self.rng.standard_normal()

    def energy_and_gradient(self, q):
        return self.energy(q), self.a * q


class TestEnergyNoise(unittest.TestCase):
    def test_estimate(self):
        model = NoisyQuadraticEnergy(np.array([1.0, 4.0, 9.0]), np.ones(3), noise=1e-4)
        noise = rom_mcmc.estimate_energy_noise(model, np.array([0.5, 0.2, 0.1]))
        self.assertAlmostEqual(noise, 1e-4, delta=0.3e-4)

    def test_microcanonical_measures_noise_at_its_samples(self):
        # A stiffening energy U = 1/2 |q|^2 + b |q|^8 with float32-like round-off relative to U. The probe on the
        # linearized level set |q| = sqrt(2 E) sits at U ~ b 2^4 E^4 >> E, like soft trampolines at 1e4 J, while the
        # chain stays at U < E, where the noise is tiny
        b, rel, E = 10.0, 1e-6, 2.0

        class Stiffening:
            dims, scale, weight = 3, np.ones(3), "uniform"

            def energy(self, q):
                r2 = q @ q
                u = 0.5 * r2 + b * r2**4
                return u * (1 + rel * np.sin(1e6 * q.sum()))  # deterministic pseudo-random round-off

            def reduced_mass(self, q):
                return np.eye(3)

        model = Stiffening()
        probe = np.sqrt(2 * E) * np.ones(3) / np.sqrt(3)
        self.assertTrue(rom_mcmc.energy_noise_stats(rom_mcmc.estimate_energy_noise(model, probe), E)["energy_noise_warning"])
        batch = rom_mcmc.run_chains(
            lambda rng: rom_mcmc.make_sampler("microcanonical", model, E, rng),
            n_chains=2,
            n_samples=40,
            burn_in=200,
            pilot=200,
            seed=3,
        )
        self.assertFalse(batch.stats["energy_noise_warning"])
        self.assertLess(batch.stats["energy_noise"], rel * E)
        # Recomputing from the stored pool reproduces the recorded statistics
        self.assertEqual(rom_mcmc.pool_energy_noise(model, batch.q, 2), batch.stats["energy_noise"])

    def test_no_false_star_violations(self):
        model = NoisyQuadraticEnergy(np.array([1.0, 4.0, 9.0]), np.ones(3), noise=1e-4)
        sampler = rom_mcmc.make_sampler("polar", model, 1.0, np.random.default_rng(5))
        sampler.burn_in(300)
        self.assertEqual(sampler.star_violations, 0)


class TestExternalForce(unittest.TestCase):
    def test_energy_gradient_and_ray_agree(self):
        # A 2-node quadratic map in R^4 with energy 1/2 |x|^2 and a constant external force f, so that
        # U(q) = 1/2 |x0 + u|^2 - 1/2 |x0|^2 - f^T u
        rng = np.random.default_rng(9)
        n, m = 4, 2
        U, Q = rng.standard_normal((n, m)), rng.standard_normal((n, m, m))
        Q = 0.5 * (Q + Q.transpose(0, 2, 1))
        x0, f = rng.standard_normal(n), rng.standard_normal(n)

        def potential(x, gradient=False):
            return (0.5 * x @ x, x) if gradient else 0.5 * x @ x

        model = rom_mcmc.QuadraticManifold(U, Q, np.ones(n), x0, potential, np.ones(m), fext=f)
        q = rng.standard_normal(m)
        u = U @ q + 0.5 * np.einsum("nij,i,j->n", Q, q, q)
        expected = 0.5 * (x0 + u) @ (x0 + u) - 0.5 * x0 @ x0 - f @ u
        self.assertAlmostEqual(model.energy(q), expected, places=12)
        e, g = model.energy_and_gradient(q)
        h = 1e-6
        fd = [(model.energy(q + h * ei) - model.energy(q - h * ei)) / (2 * h) for ei in np.eye(m)]
        np.testing.assert_allclose(g, fd, rtol=1e-6)
        self.assertAlmostEqual(model.ray_energy(q)(1.7), model.energy(1.7 * q), places=12)


class TestRadialRoot(unittest.TestCase):
    def test_energy_dip_is_not_a_star_violation(self):
        # Like gravity: U = 1/2 q^T A q - b^T q first dips below zero along rays with b^T d > 0, but still
        # crosses each positive level once
        b = np.array([2.0, 0.0, 0.0])

        class Tilted(QuadraticEnergy):
            def energy(self, q):
                return super().energy(q) - b @ q

        model = Tilted(np.array([1.0, 4.0, 9.0]), np.ones(3))
        d = np.array([1.0, 0.2, 0.0])
        for r0 in [1e-2, 1.0, 10.0]:
            r, single_crossing = rom_mcmc.radial_root(model, d, 0.5, r0)
            self.assertTrue(single_crossing)
            self.assertAlmostEqual(model.energy(r * d), 0.5, delta=1e-5)

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


class TestMicrocanonical(unittest.TestCase):
    """With U = 1/2 q^T A q and a constant mass matrix, the microcanonical (q, p) state is uniform on a
    sphere of dimension 2d - 1 in whitened phase-space coordinates. Hence U / E ~ Beta(d/2, d/2), and
    M^(1/2) qdot points in a uniformly random direction."""

    def test_equipartition_and_exact_energy(self):
        a, mass, E = np.array([1.0, 4.0, 9.0]), np.array([1.0, 2.0, 3.0]), 2.0
        model = QuadraticEnergy(a, np.ones(3), mass)
        batch = rom_mcmc.run_chains(
            lambda rng: rom_mcmc.make_sampler("microcanonical", model, E, rng),
            n_chains=4,
            n_samples=4000,
            burn_in=500,
            pilot=1000,
            seed=6,
        )
        kinetic = 0.5 * np.sum(mass * batch.qdot**2, axis=1)
        np.testing.assert_allclose(kinetic + batch.energy, E, rtol=1e-12)
        # Beta(3/2, 3/2) has mean 1/2 and standard deviation 1/4
        self.assertAlmostEqual(np.mean(batch.energy / E), 0.5, delta=0.03)
        self.assertAlmostEqual(np.std(batch.energy / E), 0.25, delta=0.03)
        w = np.sqrt(mass) * batch.qdot
        w2 = (w / np.linalg.norm(w, axis=1, keepdims=True)) ** 2
        np.testing.assert_allclose(w2.mean(axis=0), 1.0 / 3.0, atol=0.03)
        self.assertLess(batch.stats["rhat"], 1.05)

    def test_configuration_dependent_mass(self):
        # The harmonic system U = 1/2 |z|^2 with unit mass, written in coordinates z = q + c q^3. Then
        # M_r(q) = diag((1 + 3 c q^2)^2), and the ensemble is coordinate-independent, so z must show the
        # same equipartition as above. Without the sqrt(det M_r) factor, mean(U) / E drops to ~0.4.
        c = 0.5

        class Reparametrized:
            dims, scale, weight = 3, np.ones(3), "uniform"

            def z(self, q):
                return q + c * q**3

            def energy(self, q):
                return 0.5 * self.z(q) @ self.z(q)

            def reduced_mass(self, q):
                return np.diag((1 + 3 * c * q**2) ** 2)

        model, E = Reparametrized(), 2.0
        batch = rom_mcmc.run_chains(
            lambda rng: rom_mcmc.make_sampler("microcanonical", model, E, rng),
            n_chains=4,
            n_samples=4000,
            burn_in=500,
            pilot=1000,
            seed=8,
        )
        self.assertAlmostEqual(np.mean(batch.energy / E), 0.5, delta=0.03)
        zdot = (1 + 3 * c * batch.q**2) * batch.qdot
        np.testing.assert_allclose(0.5 * np.sum(zdot**2, axis=1) + batch.energy, E, rtol=1e-12)

    def test_split_rhat_detects_disagreeing_chains(self):
        rng = np.random.default_rng(7)
        same = [rng.standard_normal((2000, 2)) for _ in range(4)]
        shifted = same[:3] + [same[3] + 1.0]
        self.assertLess(rom_mcmc.split_rhat(same), 1.01)
        self.assertGreater(rom_mcmc.split_rhat(shifted), 1.1)


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


class TestWriteSamples(unittest.TestCase):
    def test_groups_share_rest_positions_and_basis(self):
        import os
        import tempfile

        import h5py

        rng = np.random.default_rng(5)
        N, m = 4, 2
        X = rng.standard_normal((3, N))
        x0 = X.flatten(order="F")
        U = rng.standard_normal((3 * N, m))
        Q = np.zeros((3 * N, m, m))
        mass = np.ones(3 * N)
        model = rom_mcmc.QuadraticManifold(U, Q, mass, x0, lambda x, *_: 0.0, np.ones(m), weight="uniform")
        q = rng.standard_normal((3, m))
        batch = rom_mcmc.SampleBatch(q, np.zeros(3), np.zeros(3), {})
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "dataset.h5")
            with h5py.File(path, "w") as f:
                f.create_dataset("X", data=X.T)
                rom_mcmc.write_basis(f.create_group("basis"), model, np.ones(m))
                for name in ["a/E1", "b/E1"]:
                    rom_mcmc.write_samples(f.create_group(name), X, model, np.ones(m), batch, {}, shared=f)
            with h5py.File(path, "r") as f:
                for name in ["a/E1", "b/E1"]:
                    g = f[name]
                    self.assertEqual(g["X"].id, f["X"].id)
                    self.assertEqual(g["basis/U"].id, f["basis/U"].id)
                    np.testing.assert_allclose(g["samples/x"][1], (x0 + U @ q[1]).reshape(N, 3), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
