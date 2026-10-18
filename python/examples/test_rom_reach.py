import unittest

import numpy as np
import scipy.sparse

import rom_mcmc
import rom_reach


def quadratic_potential(K: scipy.sparse.csr_matrix, x0: np.ndarray):
    """Psi(x) = 1/2 (x - x0)^T K (x - x0), with the potential(x, gradient, hessian) interface."""

    def potential(x, gradient=False, hessian=False):
        d = x - x0
        e = 0.5 * d @ (K @ d)
        if hessian:
            return e, K @ d, K
        return (e, K @ d) if gradient else e

    return potential


def random_spd(n: int, rng: np.random.Generator) -> scipy.sparse.csr_matrix:
    A = rng.standard_normal((n, n))
    return scipy.sparse.csr_matrix(A @ A.T + n * np.eye(n))


class TestStaticEquilibrium(unittest.TestCase):
    def test_linear_with_load_and_handles(self):
        rng = np.random.default_rng(0)
        n = 12
        K, x0 = random_spd(n, rng), rng.standard_normal(n)
        free = np.arange(3, n)
        handles, hu = np.array([4, 7]), np.array([0.3, -0.2])
        f = rng.standard_normal(n)
        u, info = rom_reach.static_equilibrium(
            quadratic_potential(K, x0), x0, free, f, handles, hu, steps=3
        )
        unknown = np.setdiff1d(free, handles)
        Kd = K.toarray()
        expected = np.linalg.solve(Kd[np.ix_(unknown, unknown)], f[unknown] - Kd[np.ix_(unknown, handles)] @ hu)
        np.testing.assert_allclose(u[unknown], expected, rtol=1e-8)
        np.testing.assert_allclose(u[handles], hu)
        np.testing.assert_array_equal(u[:3], 0.0)

    def test_implicit_euler_step(self):
        # One BDF1 step with Psi = 1/2 u^T K u solves (K + M/h^2) u = M/h^2 ut on the free DOFs
        rng = np.random.default_rng(2)
        n, h = 10, 1e-2
        K, x0 = random_spd(n, rng), rng.standard_normal(n)
        mass, ut = rng.uniform(0.5, 2.0, n), rng.standard_normal(n)
        free = np.arange(2, n)
        k = mass / h**2
        u, _ = rom_reach.static_equilibrium(
            quadratic_potential(K, x0), x0, free, inertia=(k, ut), u0=ut, steps=1
        )
        # The fixed DOFs keep their initial displacement u0 = ut and couple to the free ones through K
        fixed = np.arange(2)
        Kd = K.toarray()
        rhs = k[free] * ut[free] - Kd[np.ix_(free, fixed)] @ ut[fixed]
        np.testing.assert_allclose(u[free], np.linalg.solve(Kd[np.ix_(free, free)] + np.diag(k[free]), rhs), rtol=1e-8)
        np.testing.assert_array_equal(u[fixed], ut[fixed])

    def test_nonlinear_separable(self):
        # Psi = sum 1/2 d^2 + 1/4 d^4, so equilibrium solves d + d^3 = f per DOF
        n, x0 = 6, np.zeros(6)
        f = np.linspace(-3.0, 3.0, n)

        def potential(x, gradient=False, hessian=False):
            d = x - x0
            e = np.sum(0.5 * d**2 + 0.25 * d**4)
            if hessian:
                return e, d + d**3, scipy.sparse.diags(1 + 3 * d**2).tocsr()
            return (e, d + d**3) if gradient else e

        u, info = rom_reach.static_equilibrium(potential, x0, np.arange(n), f, steps=2)
        np.testing.assert_allclose(u + u**3, f, atol=1e-7)
        self.assertEqual(info["last_correction"].shape, (n,))
        self.assertAlmostEqual(np.abs(info["last_correction"]).max(), info["correction"])


class TestRegions(unittest.TestCase):
    def test_box_and_region_load(self):
        X = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        nodes = rom_reach.box_region(X, np.array([0.6, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(nodes, [2, 3])
        m = np.repeat([1.0, 1.0, 1.0, 3.0], 3)
        f = rom_reach.region_load(m, nodes, np.array([0.0, 0.0, -8.0]))
        np.testing.assert_allclose(f.reshape(-1, 3).sum(axis=0), [0.0, 0.0, -8.0])
        np.testing.assert_allclose(f.reshape(-1, 3)[:, 2], [0.0, 0.0, -2.0, -6.0])
        np.testing.assert_array_equal(rom_reach.node_dofs(np.array([1])), [3, 4, 5])


class TestFits(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.n, self.m = 30, 3
        U = rng.standard_normal((self.n, self.m))
        Q = 0.3 * rng.standard_normal((self.n, self.m, self.m))
        Q = 0.5 * (Q + Q.transpose(0, 2, 1))
        self.mass = rng.uniform(0.5, 2.0, self.n)
        self.x0 = rng.standard_normal(self.n)
        self.K = random_spd(self.n, rng)
        self.model = rom_mcmc.QuadraticManifold(
            U, Q, self.mass, self.x0, quadratic_potential(self.K, self.x0), np.ones(self.m), weight="uniform"
        )
        self.rng = rng

    def test_linear_fit_is_m_orthogonal_projection(self):
        ut = self.rng.standard_normal(self.n)
        u, _ = rom_reach.linear_fit(self.model.U, self.mass, ut)
        np.testing.assert_allclose(self.model.U.T @ (self.mass * (ut - u)), 0.0, atol=1e-10)

    def test_manifold_fit_recovers_reachable_target(self):
        qt = np.array([0.4, -0.7, 0.2])
        ut = self.model.displacement(qt)
        u, q = rom_reach.manifold_fit(self.model, ut)
        np.testing.assert_allclose(q, qt, atol=1e-6)
        self.assertLess(rom_reach.fit_metrics(u, ut, self.mass)["relative_error"], 1e-6)

    def test_energy_regularization_trades_fit_for_energy(self):
        ut = self.model.displacement(np.array([0.8, -0.5, 0.6])) + 0.1 * self.rng.standard_normal(self.n)
        scale = rom_reach.deformation_energy(self.model.potential, self.x0, ut)
        errors, energies = [], []
        for alpha in [0.0, 0.1, 1.0]:
            u, q = rom_reach.manifold_fit(self.model, ut, alpha, scale)
            errors.append(rom_reach.fit_metrics(u, ut, self.mass)["relative_error"])
            energies.append(self.model.energy(q))
        self.assertTrue(np.all(np.diff(errors) > 0) and np.all(np.diff(energies) < 0))

    def test_rom_equilibrium_of_linear_model(self):
        # With Psi = 1/2 u^T K u and u = U q, equilibrium solves (U^T K U) q = U^T f
        model = rom_mcmc.QuadraticManifold(
            self.model.U, None, self.mass, self.x0, quadratic_potential(self.K, self.x0), np.ones(self.m)
        )
        f = self.rng.standard_normal(self.n)
        _, q = rom_reach.rom_equilibrium(model, f, steps=2)
        U = model.U
        np.testing.assert_allclose(q, np.linalg.solve(U.T @ (self.K @ U), U.T @ f), rtol=1e-5)

    def test_rom_equilibrium_follows_handles(self):
        handles = np.arange(0, 9)
        target = self.model.displacement(np.array([0.3, 0.1, -0.2]))[handles]
        u, _ = rom_reach.rom_equilibrium(self.model, handle_dofs=handles, handle_u=target, kappa=1e4)
        self.assertLess(np.abs(u[handles] - target).max(), 0.05 * np.abs(target).max())


if __name__ == "__main__":
    unittest.main()
