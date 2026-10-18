import os
import tempfile
import unittest

import h5py
import numpy as np

import benchmark_browser


def write_problem_file(path, energy, U, Q, q, qdot, extra=None):
    """A minimal problem file: rest positions, samples and basis, as make_benchmark.py writes them."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n, m = U.shape
    N, S = n // 3, q.shape[0]
    X = np.arange(n, dtype=np.float32).reshape(N, 3)
    with h5py.File(path, "w") as f:
        f.attrs.update(dict(target_energy=energy, input="scene.h5:fem", pool_file=os.path.join("pool", os.path.basename(path)),
                            pool_indices=np.arange(S), rhat=1.01, ess=100.0))
        f.create_dataset("X", data=X)
        u = q @ U.T + 0.5 * np.einsum("nij,si,sj->sn", Q, q, q)
        f.create_dataset("samples/x", data=(X.ravel() + u).reshape(S, N, 3).astype(np.float32))
        f.create_dataset("samples/energy", data=np.linspace(0.2, 0.6, S) * energy)
        f.create_dataset("samples/q", data=q)
        f.create_dataset("samples/qdot", data=qdot)
        f.create_dataset("basis/U", data=U)
        f.create_dataset("basis/Q", data=Q)
        for name, data in (extra or {}).items():
            f.create_dataset(f"samples/{name}", data=data)


class TestBrowserData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        rng = np.random.default_rng(0)
        n, m, S = 12, 2, 3
        self.U, self.Q = rng.standard_normal((n, m)), rng.standard_normal((n, m, m))
        self.Q = 0.5 * (self.Q + self.Q.transpose(0, 2, 1))
        self.q, self.qdot = rng.standard_normal((S, m)), rng.standard_normal((S, m))
        for energy in [50.0, 5000.0]:
            write_problem_file(os.path.join(self.root, "static", f"E{energy:g}.h5"), energy, self.U, self.Q, self.q, self.qdot)
            write_problem_file(os.path.join(self.root, "pool", f"E{energy:g}.h5"), energy, self.U, self.Q, self.q, self.qdot)
        self.v = rng.standard_normal((S, n // 3, 3))
        for h in ["0.01", "0.0001"]:
            write_problem_file(
                os.path.join(self.root, f"dynamic_h{h}", "E50.h5"), 50.0, self.U, self.Q, self.q, self.qdot,
                dict(v=self.v, xtilde=np.zeros((S, n // 3, 3)), x_ref=np.ones((S, n // 3, 3)),
                     ref_newton_iterations=np.full(S, 4), ref_newton_correction=np.full(S, 1e-6), ref_min_J=np.ones(S)),
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_orders_sets_and_energies(self):
        sets = benchmark_browser.scan_dataset(self.root)
        self.assertEqual(list(sets), ["static", "dynamic_h0.0001", "dynamic_h0.01"])
        self.assertEqual([energy for energy, _ in sets["static"]], [50.0, 5000.0])

    def test_static_velocities_from_the_shared_sample(self):
        data = benchmark_browser.load_problems(os.path.join(self.root, "static", "E50.h5"))
        n, m = self.U.shape
        for s in range(self.q.shape[0]):
            J = self.U + (self.Q.reshape(n * m, m) @ self.q[s]).reshape(n, m)
            np.testing.assert_allclose(data["v"][s].ravel(), J @ self.qdot[s], rtol=1e-12)
        self.assertNotIn("xtilde", data)

    def test_dynamic_files_use_their_stored_arrays(self):
        data = benchmark_browser.load_problems(os.path.join(self.root, "dynamic_h0.01", "E50.h5"))
        np.testing.assert_allclose(data["v"], self.v)
        self.assertEqual(data["x_ref"].shape, data["x"].shape)
        np.testing.assert_array_equal(data["ref_newton_iterations"], 4)

    def test_pool_energy_shares(self):
        shares = benchmark_browser.pool_energy_shares(self.root, os.path.join("pool", "E50.h5"), 50.0)
        np.testing.assert_allclose(shares, np.linspace(0.2, 0.6, 3))
        self.assertIsNone(benchmark_browser.pool_energy_shares(self.root, "pool/missing.h5", 50.0))


if __name__ == "__main__":
    unittest.main()
