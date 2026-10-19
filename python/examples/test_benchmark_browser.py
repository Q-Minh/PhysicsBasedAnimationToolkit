import os
import tempfile
import unittest

import h5py
import numpy as np

import benchmark_browser


def write_problem_set(f, group, energy, U, Q, q, qdot, extra=None):
    """A minimal problem set: rest positions, samples and basis, as make_benchmark.py writes them."""
    n, m = U.shape
    N, S = n // 3, q.shape[0]
    X = np.arange(n, dtype=np.float32).reshape(N, 3)
    g = f.create_group(group)
    g.attrs.update(dict(target_energy=energy, pool=f"pools/pool/E{energy:g}", pool_indices=np.arange(S), rhat=1.01,
                        ess=100.0))
    g.create_dataset("X", data=X)
    u = q @ U.T + 0.5 * np.einsum("nij,si,sj->sn", Q, q, q)
    g.create_dataset("samples/x", data=(X.ravel() + u).reshape(S, N, 3).astype(np.float32))
    g.create_dataset("samples/energy", data=np.linspace(0.2, 0.6, S) * energy)
    g.create_dataset("samples/q", data=q)
    g.create_dataset("samples/qdot", data=qdot)
    g.create_dataset("basis/U", data=U)
    g.create_dataset("basis/Q", data=Q)
    for name, data in (extra or {}).items():
        g.create_dataset(f"samples/{name}", data=data)


class TestBrowserData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "dataset.h5")
        rng = np.random.default_rng(0)
        n, m, S = 12, 2, 3
        self.U, self.Q = rng.standard_normal((n, m)), rng.standard_normal((n, m, m))
        self.Q = 0.5 * (self.Q + self.Q.transpose(0, 2, 1))
        self.q, self.qdot = rng.standard_normal((S, m)), rng.standard_normal((S, m))
        self.v = rng.standard_normal((S, n // 3, 3))
        sets = []
        with h5py.File(self.path, "w") as f:
            for energy in [5000.0, 50.0]:
                write_problem_set(f, f"static/E{energy:g}", energy, self.U, self.Q, self.q, self.qdot)
                write_problem_set(f, f"pools/pool/E{energy:g}", energy, self.U, self.Q, self.q, self.qdot)
                sets.append(f"static/E{energy:g}")
            for h in ["0.01", "0.0001"]:
                write_problem_set(
                    f, f"dynamic_h{h}/E50", 50.0, self.U, self.Q, self.q, self.qdot,
                    dict(v=self.v, xtilde=np.zeros((S, n // 3, 3)), x_ref=np.ones((S, n // 3, 3)),
                         ref_newton_iterations=np.full(S, 4), ref_error_mnorm=np.full(S, 1e-6), ref_min_J=np.ones(S)),
                )
                sets.append(f"dynamic_h{h}/E50")
            f.attrs["sets"] = sets

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_orders_sets_and_energies(self):
        sets = benchmark_browser.scan_dataset(self.path)
        self.assertEqual(list(sets), ["static", "dynamic_h0.0001", "dynamic_h0.01"])
        self.assertEqual(sets["static"], [(50.0, "static/E50"), (5000.0, "static/E5000")])

    def test_static_velocities_from_the_shared_sample(self):
        data = benchmark_browser.load_problems(self.path, "static/E50")
        n, m = self.U.shape
        for s in range(self.q.shape[0]):
            J = self.U + (self.Q.reshape(n * m, m) @ self.q[s]).reshape(n, m)
            np.testing.assert_allclose(data["v"][s].ravel(), J @ self.qdot[s], rtol=1e-12)
        self.assertNotIn("xtilde", data)

    def test_dynamic_sets_use_their_stored_arrays(self):
        data = benchmark_browser.load_problems(self.path, "dynamic_h0.01/E50")
        np.testing.assert_allclose(data["v"], self.v)
        self.assertEqual(data["x_ref"].shape, data["x"].shape)
        np.testing.assert_array_equal(data["ref_newton_iterations"], 4)
        np.testing.assert_array_equal(data["ref_error_mnorm"], 1e-6)

    def test_pool_energy_shares(self):
        shares = benchmark_browser.pool_energy_shares(self.path, "pools/pool/E50", 50.0)
        np.testing.assert_allclose(shares, np.linspace(0.2, 0.6, 3))
        self.assertIsNone(benchmark_browser.pool_energy_shares(self.path, "pools/pool/missing", 50.0))


if __name__ == "__main__":
    unittest.main()
