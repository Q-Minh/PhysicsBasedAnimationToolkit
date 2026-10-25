"""
Merge the profiling results of one run made in legs (``simulate.py --resume``, each leg with its own
``--profile-out``) into one result, which every report and paper figure then takes as a single run.

Each leg is first annotated from its own output with ``profiling/trajectory.py`` (``--trajectories`` here does it,
without changing the legs' files), since a leg's output holds only its own frames. The merged result has the legs'
steps in step order and the first leg's metadata, plus ``legs``: each leg's file, first and last step, date, commit
and warm-up. The legs must be the same run: same result source, scene, solver, time step, sizes and parameters,
and no step in two legs. The warm-up steps at each seam (``--profile-warmup`` counts from the resumed step) are
missing from the merged result: tables leave them out, and the timeline shows them as gaps
(``step_stats.on_step_grid``).

Usage (from the repo root):

    python -m python.tools.sim.profiling.legs leg1.json leg2.json leg3.json -o merged.json \
        [--trajectories leg1.h5 leg2.h5 leg3.h5]
"""

import argparse

import numpy as np

from . import results

# Metadata the legs of one run share; the others (date, commit, warm-up) go to each leg's ``legs`` entry
SAME_METADATA = ("scene", "solver", "cd", "dt", "substeps", "bdf_order", "n_nodes", "n_elements", "execution",
                 "solver_params", "contact_params", "cd_params")
SPEED_KEYS = ("peak_speed", "rms_speed")  # profiling/trajectory.py's annotations


def merge_legs(legs: list[dict], sources: list[str]) -> dict:
    """One result from the legs of a run (any order), ``sources`` being their files (see the module docstring).

    Raises:
        ValueError: If a leg has no steps, the legs differ in source, :data:`SAME_METADATA` or annotations, or
            two legs share steps.
    """
    for leg, source in zip(legs, sources):
        if not leg["steps"]:
            raise ValueError(f"{source}: no steps")
    order = sorted(range(len(legs)), key=lambda i: min(s["t"] for s in legs[i]["steps"]))
    first = legs[order[0]]
    speeds = [{k for k in SPEED_KEYS if any(k in s for s in leg["steps"])} for leg in legs]
    previous_last = None
    for i in order:
        leg, source = legs[i], sources[i]
        differ = [k for k in SAME_METADATA if leg["metadata"].get(k) != first["metadata"].get(k)]
        if leg["source"] != first["source"]:
            differ.insert(0, "source")
        if differ:
            raise ValueError(f"{source} differs from {sources[order[0]]} in {', '.join(differ)}: not legs of one run")
        if speeds[i] != speeds[order[0]]:
            raise ValueError(f"{source} has {sorted(speeds[i]) or 'no speeds'}, {sources[order[0]]} "
                             f"{sorted(speeds[order[0]]) or 'none'}: annotate every leg from its own output")
        t = sorted(s["t"] for s in leg["steps"])
        if previous_last is not None and t[0] <= previous_last:
            raise ValueError(f"{source} starts at step {t[0]}, before the previous leg's last step {previous_last}")
        previous_last = t[-1]
    kernels: dict[tuple[str, str], list] = {}
    for leg in legs:
        for k in leg.get("kernels", []):
            acc = kernels.setdefault((k["path"], k["name"]), [0.0, 0])
            acc[0] += k["busy_ms"]
            acc[1] += k["count"]
    metadata = dict(first["metadata"])
    metadata["legs"] = [
        {"source": sources[i], "first_step": min(s["t"] for s in legs[i]["steps"]),
         "last_step": max(s["t"] for s in legs[i]["steps"]),
         **{k: legs[i]["metadata"].get(k) for k in ("date", "git_commit", "warmup_steps")}}
        for i in order
    ]
    return {
        "schema": results.SCHEMA,
        "source": first["source"],
        "metadata": metadata,
        "steps": sorted((s for leg in legs for s in leg["steps"]), key=lambda s: s["t"]),
        "kernels": sorted(({"path": p, "name": n, "busy_ms": ms, "count": c} for (p, n), (ms, c) in kernels.items()),
                          key=lambda k: -k["busy_ms"]),
    }


def main():
    """Command line: merge the legs' results, annotating each from its output first if given."""
    parser = argparse.ArgumentParser(description="Merge the profiling results of a run made in legs")
    parser.add_argument("legs", nargs="+", help="The legs' profiling results .json (simulate.py --profile-out).")
    parser.add_argument("-o", "--out", required=True, help="The merged result .json.")
    parser.add_argument("--trajectories", nargs="*", default=[],
                        help="Each leg's simulate.py output .h5, in the legs' order: annotates the legs "
                        "(profiling/trajectory.py) before merging, without changing their files.")
    args = parser.parse_args()
    legs = [results.load(p) for p in args.legs]
    if args.trajectories:
        from . import trajectory  # h5py, only when annotating

        if len(args.trajectories) != len(legs):
            raise SystemExit(f"--trajectories needs one output per leg ({len(legs)}), got {len(args.trajectories)}")
        for leg, path in zip(legs, args.trajectories):
            trajectory.annotate_from(leg, path)
    merged = merge_legs(legs, args.legs)
    results.save(merged, args.out)
    seams = [(a["last_step"], b["first_step"]) for a, b in zip(merged["metadata"]["legs"], merged["metadata"]["legs"][1:])]
    print(f"Merged {len(legs)} legs, {len(merged['steps'])} steps (seams, last -> first step: {seams}), "
          f"wrote {args.out}")


import unittest


class TestLegs(unittest.TestCase):

    @staticmethod
    def _run(n=30, source="production"):
        steps = [{"t": t, "busy_ms": {}, "launches": {}, "step_timeline_ms": 10.0 + t % 7, "rms_speed": 0.1 * t,
                  "status": {"vbd": [{"substep": 0, "outer": 1, "sweeps": 20 + t % 3, "reason": "CONVERGED",
                                      "gradient_ratio": 0.5, "feasibility_ratio": 0.1, "max_outer": 5,
                                      "max_sweeps": 100}]}}
                 for t in range(1, n)]
        return {"schema": results.SCHEMA, "source": source, "steps": steps, "kernels": [],
                "metadata": {"scene": "path/to/scene.h5", "solver": "AAAVBD", "dt": 0.01, "substeps": 1,
                             "solver_params": {"n_max_iters": 5}, "warmup_steps": 1, "date": "d0"}}

    @staticmethod
    def _split(run: dict, seams: tuple[int, ...]) -> list[dict]:
        """Legs resumed at the seams' steps, each missing its first (warm-up) step."""
        edges = [0, *seams, 1 << 30]
        legs = []
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            leg = {**run, "metadata": {**run["metadata"], "date": f"d{i}"}}
            leg["steps"] = [s for s in run["steps"] if lo + (i > 0) <= s["t"] < hi]
            legs.append(leg)
        return legs

    def test_merge_matches_the_run_without_the_seams(self):
        from . import paper_figures, step_stats

        run = self._run()
        legs = self._split(run, (10, 20))
        merged = merge_legs(legs[::-1], ["c.json", "b.json", "a.json"])  # any order
        seams = {10, 20}
        self.assertEqual([s["t"] for s in merged["steps"]], [t for t in range(1, 30) if t not in seams])
        self.assertEqual([(leg["source"], leg["first_step"], leg["last_step"], leg["date"])
                          for leg in merged["metadata"]["legs"]],
                         [("a.json", 1, 9, "d0"), ("b.json", 11, 19, "d1"), ("c.json", 21, 29, "d2")])
        without = {**run, "steps": [s for s in run["steps"] if s["t"] not in seams]}
        self.assertEqual(paper_figures.scene_cell(merged), paper_figures.scene_cell(without))
        # The timeline's series: the run's, NaN at the seams
        grid = step_stats.on_step_grid(step_stats.per_step(merged))
        full = step_stats.per_step(run)
        self.assertEqual(grid["t"].tolist(), full["t"].tolist())
        gap = np.isin(full["t"], list(seams))
        self.assertEqual((~grid["measured"]).tolist(), gap.tolist())
        for key in ("time_ms", "rms_speed", "gradient_ratio", "iterations"):
            np.testing.assert_array_equal(grid[key][~gap], full[key][~gap])
            self.assertTrue(np.isnan(grid[key][gap]).all())

    def test_refuses_mismatched_legs(self):
        a, b = self._split(self._run(), (10,))
        with self.assertRaisesRegex(ValueError, "before the previous leg"):
            merge_legs([a, {**b, "steps": [{**b["steps"][0], "t": 5}]}], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "solver_params"):
            merge_legs([a, {**b, "metadata": {**b["metadata"], "solver_params": {"n_max_iters": 4}}}], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "source"):
            merge_legs([a, {**b, "source": "kernels"}], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "annotate every leg"):
            merge_legs([a, {**b, "steps": [{k: v for k, v in s.items() if k != "rms_speed"} for s in b["steps"]]}],
                       ["a", "b"])


if __name__ == "__main__":
    main()
