import types
import unittest

import numpy as np

import make_benchmark


def scene(dirichlet_offset=0.0, dirichlet_speed=0.0, bdf_steps=1):
    """A stand-in for a FemElastoDynamics with 4 nodes, of which nodes 0 and 2 are Dirichlet constrained."""
    X = np.arange(12, dtype=float).reshape(3, 4)
    nodes = np.array([0, 2])
    return types.SimpleNamespace(
        X=X,
        dirichlet_nodes=nodes,
        dirichlet_coordinates=X[:, nodes] + dirichlet_offset,
        dirichlet_velocities=np.full((3, 2), dirichlet_speed),
        bdf=types.SimpleNamespace(s=bdf_steps, h=1e-2),
    )


class TestSceneChecks(unittest.TestCase):
    def test_accepts_dirichlet_at_rest(self):
        self.assertEqual(make_benchmark.check_scene(scene()), [])

    def test_rejects_displaced_or_moving_dirichlet(self):
        with self.assertRaises(ValueError):
            make_benchmark.check_scene(scene(dirichlet_offset=1e-3))
        with self.assertRaises(ValueError):
            make_benchmark.check_scene(scene(dirichlet_speed=0.1))

    def test_warns_about_multistep_bdf(self):
        warnings = make_benchmark.check_scene(scene(bdf_steps=2))
        self.assertEqual(len(warnings), 1)
        self.assertIn("BDF2", warnings[0])

    def test_rejects_mesh_input(self):
        for path in ["beam.msh", r"C:\meshes\beam.msh"]:
            with self.assertRaises(ValueError):
                make_benchmark.load_scene(path)


class TestSamplingPlan(unittest.TestCase):
    def test_default_separate_pools(self):
        plans = make_benchmark.sampling_plan(static_gravity=False, dynamic_gravity=True, has_dynamic=True)
        self.assertEqual(
            plans,
            [
                make_benchmark.SamplingPlan("pool", False, True, False),
                make_benchmark.SamplingPlan("pool_gravity", True, False, True),
            ],
        )

    def test_same_setting_shares_one_pool(self):
        for gravity, pool in [(True, "pool_gravity"), (False, "pool")]:
            plans = make_benchmark.sampling_plan(gravity, gravity, has_dynamic=True)
            self.assertEqual(plans, [make_benchmark.SamplingPlan(pool, gravity, True, True)])

    def test_static_only(self):
        plans = make_benchmark.sampling_plan(static_gravity=False, dynamic_gravity=True, has_dynamic=False)
        self.assertEqual(plans, [make_benchmark.SamplingPlan("pool", False, True, False)])


if __name__ == "__main__":
    unittest.main()
