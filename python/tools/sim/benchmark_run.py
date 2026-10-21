"""
Benchmark run files: one TOML file holding everything a benchmark run needs (``benchmark.py --run``).

A run file is a config (``config.py``: ``extends`` resolves relative to it) with two extra tables:

- ``[run]``: the run's options, in ``RUN_OPTIONS`` (``dataset``, ``sets``, ``solvers``, ``max_seconds``,
  ``budget_factor``, ``newton_stall_iterations``, ``newton_stall_reduction``), ``[run.levels]`` (levels
  per benchmark, e.g. ``static = [0.5, 0.9]``; ``all`` for every benchmark without its own) and
  ``[run.snapshots]`` (``SNAPSHOT_OPTIONS``). Relative paths (the dataset) are relative to the current
  directory, the repo root for the commands of ``benchmark.py``.
- ``[case."<KEY>"]``: solver parameters for some problems, keyed like ``--scene-params``: ``SCENE`` for
  every benchmark of a scene, ``BENCHMARK/SCENE`` for one benchmark's (e.g. ``[case."static/het".Solver.
  Newton.LinearSolver]``), merged in that order.

Every other table (``Solver``, ``Contact``, ``Integration``) holds the solver parameters shared by every case.

Precedence, lowest first: the code defaults, the run file, the command line's options and parameter files
(``--sim-params``, ``--scene-params``), then ``--set``. Unknown run options fail here; unknown solver
parameters fail in ``benchmark.py`` against the parameters the code serializes (:func:`unknown_parameters`).

``python -m python.tools.sim.benchmark_run`` runs this module's tests.
"""

import pathlib
from dataclasses import dataclass, field

from . import benchmark_results as results
from . import config

RUN = "run"
CASE = "case"
# [run] keys -> benchmark.py argument names
RUN_OPTIONS = {
    "dataset": "dataset",
    "sets": "sets",
    "solvers": "solvers",
    "max_seconds": "max_seconds",
    "budget_factor": "budget_factor",
    "newton_stall_iterations": "newton_stall_iterations",
    "newton_stall_reduction": "newton_stall_reduction",
}
# [run.snapshots] keys -> benchmark.py argument names
SNAPSHOT_OPTIONS = {
    "enabled": "snapshots",
    "fraction": "snapshot_fraction",
    "video_seconds": "snapshot_video_seconds",
    "fps": "snapshot_fps",
    "frames": "snapshot_frames",
    "memory_mb": "snapshot_memory_mb",
    "ring_ratio": "snapshot_ring_ratio",
}
ALL_BENCHMARKS = "all"  # [run.levels] key for every benchmark without its own


@dataclass
class RunFile:
    """A parsed run file."""

    path: str = ""
    text: str = ""  # the file as written
    run: dict = field(default_factory=dict)  # [run]
    shared: dict = field(default_factory=dict)  # solver parameters of every case (extends resolved)
    cases: dict = field(default_factory=dict)  # [case."KEY"] solver parameters

    @property
    def resolved(self) -> dict:
        """The whole file with its ``extends`` merged in, as ``config.read`` resolves it."""
        out = config.merge(self.shared, {CASE: self.cases} if self.cases else {})
        return config.merge(out, {RUN: self.run} if self.run else {})


def read_run_file(path: str) -> RunFile:
    """Read a run file and check its ``[run]`` options (ValueError on unknown keys)."""
    cfg = config.read(path)
    run, cases = cfg.pop(RUN, {}), cfg.pop(CASE, {})
    allowed = set(RUN_OPTIONS) | {"levels", "snapshots"}
    unknown = [f"run.{k}" for k in run if k not in allowed]
    unknown += [f"run.snapshots.{k}" for k in run.get("snapshots", {}) if k not in SNAPSHOT_OPTIONS]
    if unknown:
        raise ValueError(f"{path}: unknown run options: {', '.join(unknown)}")
    for key, value in cases.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: case.{key} must be a table of solver parameters")
    return RunFile(path=str(path), text=pathlib.Path(path).read_text(encoding="utf-8"), run=run, shared=cfg, cases=cases)


def apply_run_options(args, run: dict, defaults: dict) -> None:
    """Fill the command line's options that weren't given (None, or [] for --levels) from the run file's
    ``[run]``, then from ``defaults`` (argument name -> code default). ``--levels`` given on the command line
    override the run file's per benchmark."""
    values = {RUN_OPTIONS[k]: v for k, v in run.items() if k in RUN_OPTIONS}
    values.update({SNAPSHOT_OPTIONS[k]: v for k, v in run.get("snapshots", {}).items()})
    if isinstance(values.get("solvers"), list):
        values["solvers"] = ",".join(values["solvers"])
    if isinstance(values.get("sets"), str):
        values["sets"] = [values["sets"]]
    for name in set(values) | set(defaults):
        if getattr(args, name, None) is None:
            setattr(args, name, values.get(name, defaults.get(name)))
    levels = [
        f"{'' if benchmark == ALL_BENCHMARKS else benchmark + '='}{','.join(f'{v:g}' for v in vs)}"
        for benchmark, vs in run.get("levels", {}).items()
    ]
    given = {entry.rpartition("=")[0] or None for entry in args.levels}
    args.levels = [e for e in levels if (e.rpartition("=")[0] or None) not in given] + list(args.levels)


def case_parameters(run_file: RunFile | None, benchmark: str, scene: str) -> dict:
    """A case's solver parameters from the run file: the shared ones, then ``case."SCENE"``, then
    ``case."BENCHMARK/SCENE"``."""
    if run_file is None:
        return {}
    out = config.merge(run_file.shared, run_file.cases.get(scene, {}))
    return config.merge(out, run_file.cases.get(f"{benchmark}/{scene}", {}))


def unmatched_cases(run_file: RunFile | None, benchmark_scenes) -> list[str]:
    """The run file's case keys that match none of the run's (benchmark, scene) pairs."""
    return results.unmatched_scene_params(run_file.cases, benchmark_scenes) if run_file else []


def unknown_parameters(cfg: dict, reference: dict) -> list[str]:
    """Solver-parameter keys of ``cfg`` the code doesn't serialize (``reference``, e.g.
    ``config.params_dict(state)``), deprecated keys excepted (``config.apply`` warns about those)."""
    unknown, _ = config.check(cfg, reference)
    return [k for k in unknown if k not in config.DEPRECATED]


import unittest


class TestRunFile(unittest.TestCase):
    def _write(self, root: pathlib.Path, name: str, text: str) -> str:
        path = root / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_read_and_cases(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            root = pathlib.Path(root)
            self._write(root, "shared.toml", "[Solver.VBD]\nvlinsolve = 0\n[Solver.Newton.LinearSolver]\nrtol = 1e-6\n")
            path = self._write(root, "run.toml", """extends = ["shared.toml"]
[run]
dataset = "data/benchmarks/d.h5"
solvers = ["AAAVBD", "Newton"]
max_seconds = 60.0
[run.levels]
all = [0.5, 0.9]
static = [0.5, 0.9, 0.99]
[run.snapshots]
enabled = true
fraction = 0.25
[Solver.VBD]
gtol = 0.0
[case.het.Solver.Newton.LinearSolver]
n_max_iters = 1000
[case."static/het".Solver.Newton.LinearSolver]
n_max_iters = 4000
""")
            r = read_run_file(path)
            self.assertIn("extends", r.text)
            self.assertEqual(r.shared["Solver"]["VBD"], {"vlinsolve": 0, "gtol": 0.0})
            lin = lambda b, s: case_parameters(r, b, s)["Solver"]["Newton"]["LinearSolver"]  # noqa: E731
            self.assertEqual(lin("static", "het"), {"rtol": 1e-6, "n_max_iters": 4000})  # BENCHMARK/SCENE last
            self.assertEqual(lin("dynamic_h0.01", "het"), {"rtol": 1e-6, "n_max_iters": 1000})
            self.assertEqual(lin("static", "Y1e6"), {"rtol": 1e-6})
            pairs = [("static", "het"), ("dynamic_h0.01", "het")]
            self.assertEqual(unmatched_cases(r, pairs), [])
            self.assertEqual(unmatched_cases(r, [("static", "Y1e6")]), ["het", "static/het"])
            self.assertEqual(r.resolved["run"]["max_seconds"], 60.0)
            self.assertEqual(r.resolved["case"]["het"]["Solver"]["Newton"]["LinearSolver"]["n_max_iters"], 1000)
            # Unknown run options fail
            bad = self._write(root, "bad.toml", "[run]\nmax_second = 60.0\n[run.snapshots]\nfps2 = 24\n")
            with self.assertRaisesRegex(ValueError, "run.max_second, run.snapshots.fps2"):
                read_run_file(bad)

    def test_precedence(self):
        from types import SimpleNamespace

        run = {"solvers": ["AAAVBD", "Newton"], "max_seconds": 60.0, "budget_factor": 2.0,
               "levels": {"all": [0.5, 0.9], "static": [0.5, 0.99]}, "snapshots": {"enabled": True, "fraction": 0.25}}
        defaults = {"budget_factor": 1.0, "snapshots": False, "snapshot_fraction": 1.0, "snapshot_fps": 24.0}
        # The command line gave --budget-factor 3 and --levels static=...: they win; the rest from the run file
        args = SimpleNamespace(solvers=None, max_seconds=None, budget_factor=3.0, snapshots=None,
                               snapshot_fraction=None, snapshot_fps=None, levels=["static=0.5,0.999"])
        apply_run_options(args, run, defaults)
        self.assertEqual((args.solvers, args.max_seconds, args.budget_factor), ("AAAVBD,Newton", 60.0, 3.0))
        self.assertEqual((args.snapshots, args.snapshot_fraction, args.snapshot_fps), (True, 0.25, 24.0))
        table = results.parse_levels_options(args.levels)
        self.assertEqual(table["static"], (0.5, 0.999))
        self.assertEqual(table[None], (0.5, 0.9))
        # No run file: the code defaults
        args = SimpleNamespace(budget_factor=None, snapshots=None, levels=[])
        apply_run_options(args, {}, defaults)
        self.assertEqual((args.budget_factor, args.snapshots, args.levels), (1.0, False, []))

    def test_unknown_parameters(self):
        reference = {"Solver": {"Newton": {"LinearSolver": {"n_max_iters": 500, "rtol": 1e-6}}}}
        cfg = {"Solver": {"Newton": {"LinearSolver": {"n_max_iter": 4000}}, "Newtn": {}}}
        self.assertEqual(unknown_parameters(cfg, reference),
                         ["Solver.Newton.LinearSolver.n_max_iter", "Solver.Newtn"])
        self.assertEqual(unknown_parameters({"Solver": {"Newton": {"LinearSolver": {"rtol": 1e-2}}}}, reference), [])


if __name__ == "__main__":
    unittest.main()
