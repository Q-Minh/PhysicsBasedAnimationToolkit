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


def constraint(name, kind="Fixed", begin=0.0, duration=1e12):
    """A stand-in for python/tools/sim/gpu/kinematics/constraints.KinematicConstraint."""
    return types.SimpleNamespace(name=name, kind=types.SimpleNamespace(name=kind), begin=begin, duration=duration)


class TestSceneInputs(unittest.TestCase):
    def test_input_kind(self):
        for path, kind in [
            ("scene.Dump.h5:fem", "fem"),
            (r"C:\data\scene.Dump.h5:fem", "fem"),
            (r"C:\data\trampoline.h5", "scene"),
            ("trampoline.HDF5", "scene"),
            (r"C:\meshes\beam.msh", "mesh"),
            ("beam.mesh", "mesh"),
        ]:
            self.assertEqual(make_benchmark.modalrom.input_kind(path), kind, path)

    def test_accepts_constraints_fixed_over_the_problems(self):
        check = make_benchmark.modalrom.check_scene_constraints
        # The editor stores "forever" as a duration of ~1e12 s; finite windows covering [0, t_end] are fine too
        check([constraint("rim"), constraint("rim for 1000 s", duration=1000.0)], None, "scene.h5", t_end=1e-2)
        # Constraints entirely outside [0, t_end] never act on the problems, whatever they do
        check([constraint("later", begin=5.0), constraint("spin later", kind="Rotate", begin=5.0)], None, "scene.h5", 1e-2)

    def test_rejects_what_problems_cant_represent(self):
        cases = [
            ([constraint("spin", kind="Rotate")], None, "moves"),
            ([constraint("starts mid-step", begin=5e-3)], None, "starts or ends"),
            ([constraint("released mid-step", duration=5e-3)], None, "starts or ends"),
            ([constraint("rim")], object(), "static colliders"),
        ]
        for constraints, colliders, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                make_benchmark.modalrom.check_scene_constraints(constraints, colliders, "scene.h5", t_end=1e-2)


class TestBodyCount(unittest.TestCase):
    def test_counts_separate_bodies(self):
        one = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])  # two tets sharing nodes 1, 2, 3
        self.assertEqual(make_benchmark.modalrom.count_bodies(one, 5), 1)
        two = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])  # two disjoint tets
        self.assertEqual(make_benchmark.modalrom.count_bodies(two, 8), 2)


class TestSamplingPlan(unittest.TestCase):
    def test_default_separate_pools(self):
        plans = make_benchmark.sampling_plan(static_gravity=False, dynamic_gravity=True, has_dynamic=True)
        self.assertEqual(
            plans,
            [
                make_benchmark.SamplingPlan("pools/pool", False, True, False),
                make_benchmark.SamplingPlan("pools/pool_gravity", True, False, True),
            ],
        )

    def test_same_setting_shares_one_pool(self):
        for gravity, pool in [(True, "pools/pool_gravity"), (False, "pools/pool")]:
            plans = make_benchmark.sampling_plan(gravity, gravity, has_dynamic=True)
            self.assertEqual(plans, [make_benchmark.SamplingPlan(pool, gravity, True, True)])

    def test_static_only(self):
        plans = make_benchmark.sampling_plan(static_gravity=False, dynamic_gravity=True, has_dynamic=False)
        self.assertEqual(plans, [make_benchmark.SamplingPlan("pools/pool", False, True, False)])


class TestParseScene(unittest.TestCase):
    def test_named_and_default_names(self):
        self.assertEqual(make_benchmark.parse_scene("Y1e6=C:/data/t.h5"), ("Y1e6", "C:/data/t.h5"))
        self.assertEqual(make_benchmark.parse_scene("C:/data/trampoline_circle_Y1e6.h5"), ("trampoline_circle_Y1e6", "C:/data/trampoline_circle_Y1e6.h5"))
        self.assertEqual(make_benchmark.parse_scene("C:/data/beam.Dump.h5:fem"), ("beam.Dump", "C:/data/beam.Dump.h5:fem"))

    def test_rejects_names_reserved_by_paths(self):
        for spec in ["a/b=t.h5", "a:b=t.h5", "=t.h5"]:
            with self.assertRaises(ValueError):
                make_benchmark.parse_scene(spec)


if __name__ == "__main__":
    unittest.main()
