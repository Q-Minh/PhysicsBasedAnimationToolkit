# type: ignore
r"""
Solver convergence benchmark: solver time per accuracy level of error reduction.

The problems come from one benchmark dataset file (python/examples/make_benchmark.py; contract in
plans/MASTER.md), which holds the FEM they are posed on and its problem sets (e.g. ``static/E50``,
``dynamic_h0.01/E500``). Each problem of each selected set is solved once per solver. A set's attribute
``problem`` gives the kind:

- **static**: release the sample shape x with no loads. The solution is the rest shape X.
- **dynamic**: one implicit-Euler (BDF1) step of the set's size h from the sample (x, v), toward the
  set's inertial target xtilde (gravity included). The solution is the set's reference x_ref
  (full-order Newton); its accuracy ``ref_error_mnorm`` is recorded.

Benchmarks and cases: the sets of one kind and time step form a benchmark (``static``,
``dynamic_h0.0001``, ``dynamic_h0.01``), one figure each (``benchmark_plot.py``). Within a benchmark, each
scene is a case (an x-axis entry); a scene with several selected energy levels gives one case per level
(``<scene>_E<E>``). A set's scene is its ``scene`` attribute, or for datasets without one (version 1),
the stem of the dataset's ``input`` scene file.

The error e = |x - x*|_M (lumped-mass norm over free nodes, x* = X or x_ref) is logged after every
solver iteration (see ``static.py``). For accuracy levels l (error reductions 1 - e/e0), the benchmark
records per solve the time each level is first crossed and its incremental time: an iteration's time goes
to the highest level it crosses, and the levels it jumps over get none; levels crossed after the budget
aren't reached (``benchmark_results.py``). ``--levels [BENCHMARK=]L1,L2,...`` (repeatable) sets them per
benchmark, e.g. ``--levels 0.5,0.9,0.99 --levels static=0.5,0.9,0.99,0.999``; the defaults are decades,
50, 90, 99, 99.9, 99.99 and 99.999% for the static benchmark and up to 99.99% for the dynamic ones (their
float32 references resolve about that much at best). Only that error stops a solve, at ``--target-reduction`` (default: the benchmark's
highest level), at its time budget or iteration cap, or for Newton at its stall test; the solvers' own
convergence and feasibility tests are bypassed.

Time budgets, per sample (Newton normalization, the default): Newton solves each problem first, to the
target, until its stall test stops it (``--newton-stall-iterations`` K, ``--newton-stall-reduction`` delta:
the last K iterations improved the best error by less than delta per iteration on average,
min_{j<=c} e_j > (1 - delta)^K min_{j<=c-K} e_j), or at most ``--max-seconds`` (default 60 s, a practical
cap). Its time to the highest level it
reached, T_N, sets that problem's budget for every other solver: ``--budget-factor`` (default 1) x T_N.
Every solver's increments on that problem are divided by T_N (``normalized_increment``), so Newton's
reached increments sum to 1. A case needs Newton among ``--solvers``, unless
``--budget [BENCHMARK[/CASE]=]SECONDS`` (repeatable, the most specific entry first: e.g. ``--budget
static=5 --budget static/Y1e8=10``, or ``--budget 5`` for every case) gives it a fixed budget: then every
solver, Newton included, gets that budget, and the increments are divided by it.

Run files: ``--run RUN.toml`` (``benchmark_run.py``) holds a run's options ([run]: dataset, sets, solvers,
levels per benchmark, Newton's cap, budget factor, stall test, snapshots), the solver parameters every case
shares, and per-case ones ([case."SCENE"], [case."BENCHMARK/SCENE"]); the full run is
``configs/benchmark/trampolines.toml``. Command-line options and parameter files win over it, then --set.
The results store the run file's text (``run_toml``) and its resolved form, and each case its parameters
(``sim_params_toml``). ``--dry-run`` prints each case's sets, levels, budget and Newton CG settings without
solving, and fails on unknown solver parameters. To write a run file: ``--list`` prints the dataset's case keys
(BENCHMARK/SCENE, for ``[case."<benchmark>/<scene>".<Table.path>]``) with each set's kind, h, scene, energy,
problem count, mesh, materials, Dirichlet nodes and reference accuracy, without the GPU; ``python -m
python.tools.sim.config params [PREFIX]`` lists every solver parameter (default, type, enum names, description).

Usage:
    python -m python.tools.sim.benchmark [--run RUN.toml] [--dry-run | --list] \
        --dataset DATASET.h5 [--sets PATTERN ...] -o results.h5 \
        [--solvers VBD,AAAVBD,Chebyshev,AndersonSolver,Newton] [--levels [BENCHMARK=]0.5,0.6,...,0.99 ...] \
        [--sim-params A.toml [B.h5 ...]] [--scene-params [BENCHMARK/]SCENE=C.toml[,D.toml] ...] \
        [--set Table.key=value ...] \
        [--max-seconds 60] [--budget-factor 1] [--newton-stall-iterations K] [--newton-stall-reduction D] \
        [--budget [BENCHMARK[/CASE]=]SECONDS ...] \
        [--spec spec.h5] [--save-spec spec.h5] [--sample-begin 0] [--sample-end -1] \
        [--max-iterations 1000000] [--target-reduction 0.99] [--block 16] \
        [--snapshots [--snapshot-video-seconds 5] [--snapshot-fps 24] [--snapshot-frames F] \
         [--snapshot-fraction 1] [--snapshot-memory-mb 32] [--snapshot-ring-ratio 2]]

Representative commands (from the repo root; datasets live in the main checkout's data/benchmarks):

    # One static solve: one set, one solver, the first problem, a fixed 5 s budget
    python -m python.tools.sim.benchmark --dataset data/benchmarks/trampolines-test.h5 \
        --sets Y1e6/static/E10000 --solvers AAAVBD --sample-end 1 --budget 5 -o out/one.h5

    # The benchmark: per problem, Newton first (at most 60 s), then every other solver within Newton's time
    # T_N to its highest level, with the run file's levels, snapshots and per-case Newton CG settings; check
    # it first with --dry-run, and point --dataset at the main checkout's data from a worktree
    python -m python.tools.sim.benchmark --run python/tools/sim/configs/benchmark/trampolines.toml --dry-run -o out/results.h5
    python -m python.tools.sim.benchmark --run python/tools/sim/configs/benchmark/trampolines.toml -o out/results.h5

    # The same with one change, e.g. two scenes and another CG cap (the command line wins over the run file)
    python -m python.tools.sim.benchmark --run python/tools/sim/configs/benchmark/trampolines.toml \
        --sets "Y1e6/*" "het/*" --set Solver.Newton.LinearSolver.n_max_iters=3000 -o out/variant.h5

    # The figures
    python -m python.tools.sim.benchmark_plot out/results.h5 -o out/figures/benchmark

    # Parameter variations: layer configs and override single keys (see python/tools/sim/configs/README.md)
    python -m python.tools.sim.benchmark ... --sim-params base.toml static.toml \
        --set Solver.Newton.LinearSolver.n_max_iters=3000 --set Solver.AAAVBD.rhohat=0.9

    # Reproducible cases: save the effective StaticSolveSpec (dataset, sets, caps), then rerun from it
    python -m python.tools.sim.benchmark --dataset DATASET.h5 --sets static/E500 --max-seconds 10 --save-spec case.h5 -o a.h5
    python -m python.tools.sim.benchmark --spec case.h5 -o b.h5

Output: one HDF5 results file (layout in ``benchmark_results.py``'s docstring): the spec, levels per
benchmark, budget options, dataset, device, commit and parameter sources; per benchmark, case and solver,
each sample's budget and normalization time (T_N), crossing times, increments (also over T_N), reached and
skipped flags, e0, iterations, sweeps, solver time, host time (the solve's wall-clock time, including
loading and reading back), final relative error, termination and full per-iteration log; and per level,
the mean, std, min, max, median, quartiles and count of the increments, normalized increments and
crossing times over the samples that reached it. A solve's ``termination`` is ``target`` (reached
--target-reduction), ``iteration_cap``, ``time_cap`` (its budget), ``non_finite`` (NaN or infinite error),
``stalled`` (Newton's stall test), or for Newton ``newton_non_descent``, ``newton_line_search_failed`` or
``newton_zero_gradient``: Newton stops at its first rejected step, since x doesn't change after it.
Dynamic cases add the reference's accuracy (``reference_error_bound``, the set's ``samples/ref_error_mnorm``,
and ``reference_relative_error_bound``, over e0) and ``resolvable``, the levels that accuracy resolves.

Trajectories (opt-in, for videos): ``--snapshots`` stores the positions at F equal solver-time intervals over
the problem's budget (``--budget-factor`` x T_N, also for Newton, whose frames stop there; or the fixed
budget), plus each level crossing, in the same run (``benchmark_results.py`` documents the layout).
F = ``--snapshot-video-seconds`` x ``--snapshot-fps`` (default 5 s x 24 fps = 120 frames), unless
``--snapshot-frames`` sets it; each trajectory stores its ``playback_factor`` = video seconds / budget, the
video's slow-down relative to solver time (e.g. 200 for a 25 ms budget), per problem. ``--snapshot-fraction`` (default 1:
every sample) picks evenly spaced samples of each set. The run prints the estimated disk size first:
about F x N x 12 bytes per solve (~15 MB for 120 frames of the 10643-node trampolines).

During the solve, every log entry may copy x into a fixed device ring (``static.SnapshotRing``): between the
entry's timestamps, so outside the solver time, and without touching the solver's state. The ring holds
min(``--snapshot-ring-ratio`` x F, what ``--snapshot-memory-mb`` holds) frames (default 2 x F within 32 MB:
248 frames of the trampolines, i.e. 240 ring slots and 8 level slots), allocated once per solver and case;
when it fills, every other snapshot is dropped and the stride between snapshots doubles, so it spans any
solve length. A cap that holds fewer than 2 F slots prints a warning, and frames fall up to a ring stride from
their target times (each trajectory stores them). After the solve, only the chosen frames are downloaded.

Pitfalls:
    - The benchmark is contact-free: contact detection is deactivated whatever the parameters say.
    - The problems are posed on the dataset's FEM (its ``fem`` group, or the set's ``fem`` attribute): its
      Dirichlet nodes stay fixed at rest. Solver parameters come only from the run file (shared, then its
      cases), then --sim-params, then the scene's --scene-params (``SCENE=``, then ``BENCHMARK/SCENE=``,
      e.g. ``static/Y1e8=``), then --set (otherwise the code defaults, configs/defaults.toml). Unknown keys
      stop the run.
    - Dynamic problems take their time step from the set: ``Integration.dt``, ``substeps`` and the BDF
      order from the parameters are overridden (one BDF1 step of size h).
    - At small h (e.g. 1e-4) a step starts close to its solution (e0 ~ 1e-3), so float32 rounding
      floors the error around e/e0 ~ 1e-5 to 1e-4. Newton's line search accepts steps within float32
      roundoff of the energy, so at a floor Newton keeps stepping on noise: its stall test ends it.
    - The stall test measures the error, not the energy: Newton's damped steps on large deformations can
      reduce e by only a few percent per step (static het: 4-9% for ~60 steps), so a large delta stops it
      early and loses levels (plans/MASTER.md, the floor check).
    - Solves run to the target reduction or the caps, not to a solver tolerance: ``Solver.<solver>.gtol``,
      ``n_max_iters`` and ``n_subproblem_max_iters`` don't stop them (a static solve is one long
      subproblem). Newton's CG keeps its relative tolerance, but its absolute one (``atol``) is set
      to 0: in h^2-scaled units at small h, |g| drops below it after one step. The vertex-solve and acceleration parameters do matter.
    - Newton isn't graph-captured. Each Newton step synchronizes with the host, so its times include host
      work and vary more between runs, and so do the budgets T_N it sets. Its CG cap and tolerance
      (``Solver.Newton.LinearSolver.n_max_iters``, ``rtol``) trade exact steps for time: see the per-case
      table in ``configs/benchmark/trampolines.toml``.
    - Times are solver time from device timestamps, logging excluded; the budget counts the same time.
      Every iteration is timed on its own, also inside replayed graph blocks, and iterations a block runs
      past the stop are dropped, so --block doesn't change the crossings. Don't run other GPU work meanwhile.
    - Sets with ``importance_weights_required`` are refused; R-hat >= 1.05 or nonzero sampling gravity
      only print a warning.
    - Each solver's first solve on a set also compiles kernels and captures its graph. That shows in the host
      time, not in the reported solver time.
"""

import argparse
import fnmatch
import math
import os
import subprocess
import time
import warnings
from dataclasses import dataclass

import h5py
import numpy as np
import warp as wp
from tqdm import tqdm

from .state import (
    SimulationState,
    SolverType,
    CDType,
    load_params,
    load_problem,
)
from . import benchmark_results as results
from . import benchmark_run, config, static

# Code defaults of the options a run file can set (benchmark_run.RUN_OPTIONS, SNAPSHOT_OPTIONS): the command
# line's value wins, then the run file's, then these
DEFAULTS = {
    "solvers": ",".join(s.name for s in SolverType),
    "budget_factor": 1.0,
    "newton_stall_iterations": 2,
    "newton_stall_reduction": 0.02,
    "snapshots": False,
    "snapshot_video_seconds": 5.0,
    "snapshot_fps": 24.0,
    "snapshot_frames": None,
    "snapshot_fraction": 1.0,
    "snapshot_ring_ratio": 2.0,
    "snapshot_memory_mb": 32.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solver convergence benchmark (GPU) on the static and dynamic problem sets of a benchmark "
        "dataset: solver time per accuracy level of error reduction."
    )
    parser.add_argument("--run", default=None, metavar="RUN.toml",
                        help="Run file (benchmark_run.py): the run's options ([run]), shared and per-case solver "
                        "parameters ([case.\"SCENE\"], [case.\"BENCHMARK/SCENE\"]). Options given here win over it, "
                        "then --sim-params / --scene-params, then --set.")
    parser.add_argument("--list", action="store_true", dest="list_cases",
                        help="List the dataset's problem sets (from --dataset or the run file) as run-file case keys "
                        "(BENCHMARK/SCENE), with kind, h, scene, energy, problem count, mesh, materials, Dirichlet "
                        "nodes and the dynamic sets' reference accuracy, plus the levels, budget and case parameters "
                        "a --run file gives them; then exit (no GPU, no sample batches loaded).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print each case's sets, levels, budget and effective Newton CG settings, and exit "
                        "without solving (no GPU).")
    parser.add_argument("--dataset", default=None,
                        help="Benchmark dataset (.h5) from python/examples/make_benchmark.py: the FEM and the "
                        "problem sets. Overrides the spec's dataset; required unless --spec provides one.")
    parser.add_argument("--sets", nargs="+", default=None,
                        help="Problem sets to run, as fnmatch patterns over the dataset's set names (e.g. "
                        "static/E50 'dynamic_h0.01/*'). Overrides the spec's sets (spec default: all sets).")
    parser.add_argument("--sim-params", nargs="+", default=[], dest="sim_params",
                        help="Simulation parameters for every scene: .toml configs or .h5 files as exported by "
                        "the GUI (file.h5 or file.h5:group), merged left to right (see configs/README.md). "
                        "Without any, the code defaults apply (configs/defaults.toml).")
    parser.add_argument("--scene-params", action="append", default=[], metavar="[BENCHMARK/]SCENE=A.toml[,B.toml]",
                        dest="scene_params",
                        help="Parameters for one scene's problems, merged after --sim-params and before --set: "
                        "SCENE= for every benchmark of the scene, BENCHMARK/SCENE= (e.g. static/Y1e8=) for one "
                        "benchmark's, merged after the scene's. Repeatable, once per key.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="Override one parameter after the parameter files, as Table.Sub.key=TOML-literal, "
                        "e.g. Solver.Newton.LinearSolver.n_max_iters=3000. Repeatable.")
    parser.add_argument("--solvers", default=None,
                        help=f"Comma-separated solver names to run, in order (default: all, {DEFAULTS['solvers']}).")
    parser.add_argument("--levels", action="append", default=[], metavar="[BENCHMARK=]L1,L2,...",
                        help="Accuracy levels, as comma-separated error reductions 1 - e/e0 in (0, 1), for every "
                        "benchmark or one (e.g. static=0.5,0.9,0.99,0.999). Repeatable; a benchmark's own entry "
                        "wins (default: static " + ",".join(f"{v:g}" for v in results.STATIC_LEVELS) + ", dynamic "
                        + ",".join(f"{v:g}" for v in results.DYNAMIC_LEVELS) + ").")
    parser.add_argument("--budget-factor", type=float, default=None, dest="budget_factor",
                        help="Every other solver's budget on a problem: this multiple of Newton's time T_N to its "
                        f"highest level there (default: {DEFAULTS['budget_factor']:g}).")
    parser.add_argument("--newton-stall-iterations", type=int, default=None, dest="newton_stall_iterations",
                        metavar="K",
                        help="Newton's stall test: stop once the last K iterations improved the best error by less "
                        f"than (1 - DELTA)^K (default: {DEFAULTS['newton_stall_iterations']}; 0 = off, Newton then "
                        "runs to the target or --max-seconds).")
    parser.add_argument("--newton-stall-reduction", type=float, default=None, dest="newton_stall_reduction",
                        metavar="DELTA",
                        help="The stall test's average per-iteration reduction of the best error (default: "
                        f"{DEFAULTS['newton_stall_reduction']:g}). On the floor check, 0.04 was the largest that lost "
                        "no level (static het and Y1e5).")
    parser.add_argument("--budget", action="append", default=[], metavar="[BENCHMARK[/CASE]=]SECONDS",
                        help="A fixed time budget per solve instead of Newton's, in seconds of solver time, for every "
                        "case, a benchmark (e.g. static=5) or one case (static/Y1e8=10): every solver gets it and "
                        "the increments are divided by it. Repeatable; the most specific entry wins.")
    parser.add_argument("--spec", default=None,
                        help="StaticSolveSpec (.h5) to start from. The options below override its fields.")
    parser.add_argument("--save-spec", default=None, dest="save_spec",
                        help="Write the effective StaticSolveSpec (.h5), to rerun the same case with --spec.")
    parser.add_argument("--sample-begin", type=int, default=None, dest="sample_begin",
                        help="First sample index (spec default: 0).")
    parser.add_argument("--sample-end", type=int, default=None, dest="sample_end",
                        help="One past the last sample index; -1 means all samples (spec default: -1).")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                        help="Newton's time cap per solve, in seconds of solver time, for the cases without --budget "
                        "(spec default: 60).")
    parser.add_argument("--max-iterations", type=int, default=None, dest="max_iterations",
                        help="Iteration cap per solve, counting VBD sweeps, Anderson steps or Newton steps "
                        "(default here: 1,000,000, so the time budget decides).")
    parser.add_argument("--target-reduction", type=float, default=None, dest="target_reduction",
                        help="Stop a solve once e <= (1 - target) e0, as a fraction (default: the benchmark's "
                        "highest level).")
    parser.add_argument("--block", type=int, default=None,
                        help="Solver iterations per replayed CUDA-graph block, rounded up to a multiple of "
                        "Anderson's window m. Larger blocks mean fewer replays; iterations past the stop are "
                        "dropped (spec default: 16).")
    parser.add_argument("--snapshots", action="store_true", default=None,
                        help="Store trajectories for videos: F frames at equal solver-time intervals over the budget, "
                        "plus each level crossing. Snapshots are copied on the device between iterations, outside "
                        "the timed span, and leave the solves unchanged.")
    parser.add_argument("--snapshot-video-seconds", type=float, default=None, dest="snapshot_video_seconds",
                        help=f"Video duration a trajectory plays in (default: {DEFAULTS['snapshot_video_seconds']:g} s); "
                        "F = duration x fps.")
    parser.add_argument("--snapshot-fps", type=float, default=None, dest="snapshot_fps",
                        help=f"Video frame rate (default: {DEFAULTS['snapshot_fps']:g}).")
    parser.add_argument("--snapshot-frames", type=int, default=None, dest="snapshot_frames", metavar="F",
                        help="Frames per trajectory, instead of duration x fps.")
    parser.add_argument("--snapshot-fraction", type=float, default=None, dest="snapshot_fraction",
                        help="Fraction of each set's samples to store trajectories for, evenly spaced (default: "
                        f"{DEFAULTS['snapshot_fraction']:g}, every sample). A trajectory is ~F x N x 12 bytes on disk.")
    parser.add_argument("--snapshot-ring-ratio", type=float, default=None, dest="snapshot_ring_ratio",
                        help=f"Ring slots per requested frame (default: {DEFAULTS['snapshot_ring_ratio']:g}), within "
                        "--snapshot-memory-mb.")
    parser.add_argument("--snapshot-memory-mb", type=float, default=None, dest="snapshot_memory_mb",
                        help="Device memory cap for the snapshots, allocated once per solver and case (default: "
                        f"{DEFAULTS['snapshot_memory_mb']:g} MB). The ring holds min(ratio x F, what fits) frames; fewer than ratio x F, with a "
                        "warning, place frames up to a ring stride from their target times.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output results file (.h5); required unless --list or --dry-run.")
    args = parser.parse_args()
    if args.output is None and not (args.list_cases or args.dry_run):
        parser.error("the following arguments are required to solve: -o/--output")
    return args


def make_spec(args) -> static.StaticSolveSpec:
    spec = static.StaticSolveSpec()
    spec.max_iterations = 1_000_000
    if args.spec:
        with h5py.File(args.spec, "r") as f:
            spec.deserialize(f["StaticSolveSpec"] if "StaticSolveSpec" in f else f)
    for name, value in [
        ("dataset", args.dataset),
        ("sets", ",".join(args.sets) if args.sets else None),
        ("sample_begin", args.sample_begin),
        ("sample_end", args.sample_end),
        ("max_seconds", args.max_seconds),
        ("max_iterations", args.max_iterations),
        ("target_reduction", args.target_reduction),
        ("block", args.block),
        ("newton_stall_iterations", args.newton_stall_iterations),
        ("newton_stall_reduction", args.newton_stall_reduction),
    ]:
        if value is not None:
            setattr(spec, name, value)
    if not spec.dataset:
        raise SystemExit("No problems: pass --dataset or a --spec with a dataset")
    return spec


def copy_spec(spec: static.StaticSolveSpec, **changes) -> static.StaticSolveSpec:
    out = static.StaticSolveSpec()
    for name, value in {**spec.to_dict(), **changes}.items():
        setattr(out, name, value)
    return out


def select_sets(spec: static.StaticSolveSpec) -> list[str]:
    """The dataset's problem sets matching the spec's patterns (all when empty), in dataset order."""
    names = static.dataset_sets(spec.dataset)
    patterns = [p for p in spec.sets.split(",") if p]
    selected = [n for n in names if not patterns or any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not selected:
        raise SystemExit(f"No set of {spec.dataset} matches {spec.sets!r}; its sets are {', '.join(names)}")
    return selected


# --- Scenes and cases (the dataset-format-specific layer) ---


@dataclass
class Case:
    """One selected problem set, placed in its benchmark (figure) and case (x-axis entry)."""

    set_name: str  # the set's path in the dataset
    batch: static.SampleBatch
    scene: str
    label: str  # the scene's display name
    fem: str  # the FEM group's path in the dataset
    benchmark: str  # benchmark_results.benchmark_name
    case: str = ""  # filled by make_cases


def _scene_of(dataset: str, f: h5py.File, attrs: dict) -> tuple[str, str, str]:
    """(scene, label, FEM group) of a problem set, from its attributes: its ``scene``/``fem`` attributes, else
    (version-1 datasets) the dataset's ``input`` scene file stem and the root FEM."""
    scene = attrs.get("scene")
    if scene is None:
        source = f.attrs.get("input") or dataset
        scene = os.path.splitext(os.path.basename(str(source).replace("\\", "/")))[0]
        return str(scene), str(scene), "fem"
    scene = str(scene)
    label = f[scene].attrs.get("label", scene) if scene in f else scene
    return scene, str(label), str(attrs.get("fem", f"{scene}/fem"))


def make_cases(dataset: str, set_names: list[str]) -> list[Case]:
    cases = []
    with h5py.File(dataset, "r") as f:
        for name in set_names:
            batch = static.load_sample_batch(f"{dataset}:{name}")
            scene, label, fem = _scene_of(dataset, f, batch.attrs)
            cases.append(Case(name, batch, scene, label, fem, results.benchmark_name(batch.problem, batch.h)))
    for c in cases:
        energies = {o.batch.target_energy for o in cases if (o.benchmark, o.scene) == (c.benchmark, c.scene)}
        c.case = c.scene if len(energies) == 1 else f"{c.scene}_E{c.batch.target_energy:g}"
    return cases


def case_config(args, run_file, benchmark: str, scene: str, scene_params: dict) -> dict:
    """A case's solver parameters: the run file's (shared, case.SCENE, case.BENCHMARK/SCENE), then
    --sim-params and the case's --scene-params files, then --set."""
    sources = args.sim_params + results.scene_params_for(scene_params, benchmark, scene)
    return config.merge(benchmark_run.case_parameters(run_file, benchmark, scene), config.resolve(sources, args.overrides))


def _short(v: float) -> str:
    """3 significant digits, exponents without '+' and leading zeros: 1e6, 2.5e-5, 0.45."""
    return f"{v:.3g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _fem_summary(group: h5py.Group) -> dict:
    """Mesh size, materials and Dirichlet nodes of a serialized FemElastoDynamics (a dataset's ``fem`` group)."""
    found = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1] in ("X", "E", "lamegU", "egU", "dmask", "m"):
            found.setdefault(name.rsplit("/", 1)[-1], obj)

    group.visititems(visit)
    n_nodes, n_tets = int(found["X"].shape[1]), int(found["E"].shape[1])
    # One quadrature point per element (the first): Young's modulus and Poisson ratio from its Lame coefficients
    mu, lam = np.asarray(found["lamegU"][()], dtype=np.float64)
    _, first = np.unique(np.asarray(found["egU"][()]).ravel(), return_index=True)
    mu, lam = mu[first], lam[first]
    young = np.array([float(f"{v:.3g}") for v in mu * (3 * lam + 2 * mu) / (lam + mu)])
    poisson = sorted({float(f"{v:.3g}") for v in lam / (2 * (lam + mu))})
    values, counts = np.unique(young, return_counts=True)
    dmask = np.asarray(found["dmask"][()]).ravel() != 0
    return {
        "nodes": n_nodes,
        "tets": n_tets,
        "materials": ", ".join(f"E {_short(v)} ({c} tets)" for v, c in zip(values, counts))
        + f"; nu {', '.join(_short(v) for v in poisson)}",
        "dirichlet": int(dmask.sum()),
        "m": np.asarray(found["m"][()], dtype=np.float64).ravel(),
        "free": ~dmask,
    }


def list_sets(dataset: str, set_names: list[str]) -> list[dict]:
    """Per problem set, what a run file needs to target it: ``key`` (the run-file case key BENCHMARK/SCENE),
    ``set``, ``benchmark``, ``problem``, ``h``, ``scene``, ``label``, ``energy``, ``problems``, ``nodes``, ``tets``,
    ``materials`` (Young's moduli with their tet counts, Poisson ratios), ``dirichlet`` (fixed nodes), and for
    dynamic sets ``reference_accuracy``: the median over its problems of the reference's relative accuracy
    ref_error_mnorm / |x - x_ref|_M. Reads attributes, the FEM's mesh and materials and, for dynamic sets,
    the samples' x and x_ref; no GPU."""
    rows, fems = [], {}
    with h5py.File(dataset, "r") as f:
        for name in set_names:
            g = f[name]
            attrs = dict(g.attrs)
            problem = str(attrs.get("problem", "static"))
            h = float(attrs["h"]) if problem != "static" and "h" in attrs else None
            scene, label, fem = _scene_of(dataset, f, attrs)
            if fem not in fems:
                fems[fem] = _fem_summary(f[fem])
            info = fems[fem]
            benchmark = results.benchmark_name(problem, h)
            row = {
                "key": f"{benchmark}/{scene}", "set": name, "benchmark": benchmark, "problem": problem, "h": h,
                "scene": scene, "label": label, "energy": float(attrs.get("target_energy", math.nan)),
                "problems": int(g["samples/x"].shape[0]),
                **{k: info[k] for k in ("nodes", "tets", "materials", "dirichlet")},
            }
            if problem == "dynamic" and "samples/ref_error_mnorm" in g:
                d = np.asarray(g["samples/x"][()], dtype=np.float64) - np.asarray(g["samples/x_ref"][()], dtype=np.float64)
                e0 = np.sqrt(np.einsum("n,snd->s", info["m"] * info["free"], d * d))
                bound = np.asarray(g["samples/ref_error_mnorm"][()], dtype=np.float64)
                row["reference_accuracy"] = float(np.median(bound / np.maximum(e0, 1e-30)))
            rows.append(row)
    return rows


def print_sets(rows: list[dict], run_file, levels_table, fixed_budgets: dict, scene_params: dict, args) -> None:
    """Print :func:`list_sets`, with the levels, budget and case parameters the run would give each set."""
    for r in rows:
        kind = "static" if r["problem"] == "static" else f"dynamic, h = {r['h']:g} s"
        print(f"{r['key']}  [case.\"{r['key']}\"]")
        print(f"    set {r['set']}: {kind}, scene {r['scene']} ({r['label']}), E = {r['energy']:g} J, "
              f"{r['problems']} problems")
        print(f"    mesh {r['nodes']} nodes, {r['tets']} tets; {r['materials']}; {r['dirichlet']} Dirichlet nodes"
              + (f"; reference accuracy (median) {_short(r['reference_accuracy'])} e0" if "reference_accuracy" in r else ""))
        if run_file is not None or args.levels:
            levels = ",".join(f"{v:g}" for v in results.levels_for(levels_table, r["benchmark"]))
            seconds, _ = results.fixed_budget(fixed_budgets, r["benchmark"], r["key"].split("/", 1)[1])
            print(f"    levels {levels}; budget "
                  + (f"fixed {seconds:g} s" if seconds is not None else f"{args.budget_factor:g} x Newton's T_N"))
        # The run file's parameters for this case beyond its shared ones (case.SCENE, case.BENCHMARK/SCENE)
        shared = dict(config.flatten(run_file.shared)) if run_file is not None else {}
        case_only = [f"{k}={v}" for k, v in config.flatten(benchmark_run.case_parameters(run_file, r["benchmark"], r["scene"]))
                     if shared.get(k, object()) != v]
        if case_only:
            print(f"    case parameters: {', '.join(case_only)}")


def dataset_benchmark_scenes(dataset: str) -> list[tuple[str, str]]:
    """(benchmark, scene) of every problem set of the dataset, from the sets' attributes (no samples loaded)."""
    out = []
    with h5py.File(dataset, "r") as f:
        for name in static.dataset_sets(dataset):
            attrs = dict(f[name].attrs)
            problem = str(attrs.get("problem", "static"))
            h = float(attrs["h"]) if problem != "static" and "h" in attrs else None
            out.append((results.benchmark_name(problem, h), _scene_of(dataset, f, attrs)[0]))
    return out


def make_state(dataset: str, fem: str, h: float | None, cfg: dict, source: str = "") -> SimulationState:
    """Build the contact-free simulation on a FEM of the dataset with the solver parameters ``cfg``
    (:func:`case_config`; unknown keys exit); ``h`` set means one-step dynamic problems of that step."""
    # The dataset's FEM has no kinematics group: its Dirichlet nodes stay fixed at rest
    # (check_sample_batch checks the samples against that), and the benchmark is contact-free.
    fem_cpu, _, _ = load_problem(f"{dataset}:{fem}")
    params_cpu = load_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu)
    if cfg:
        unknown = benchmark_run.unknown_parameters(cfg, config.params_dict(state))
        if unknown:
            raise SystemExit(f"{source or 'solver parameters'}: unknown keys: {', '.join(unknown)}")
        with warnings.catch_warnings():
            # A run file or config sets only some keys; the others keep their code defaults
            warnings.filterwarnings("ignore", message=".*keys not set.*", category=config.ConfigWarning)
            config.apply(state, cfg, source=source or "solver parameters")
    for cd_params in state.cd_params.values():
        if hasattr(cd_params, "deactivate"):
            cd_params.deactivate = True
    state.cd_type = CDType.VertexSdf
    if h is None:
        state.reset()
    else:
        static.prepare_dynamic_problem(state, h)
    return state


def dry_run(args, spec, run_file, cases, levels_table, budgets, scene_params) -> None:
    """Print each case's sets, levels, budget and effective Newton CG settings (the case's parameters over
    configs/defaults.toml), checking its parameters against those defaults. No GPU."""
    defaults = config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "defaults.toml"))
    print(f"Solvers {args.solvers}; budget factor {args.budget_factor:g}; Newton cap {spec.max_seconds:g} s; "
          f"stall K = {args.newton_stall_iterations}, delta = {args.newton_stall_reduction:g}; snapshots "
          + (f"{args.snapshot_fraction:g} of the samples" if args.snapshots else "off"))
    failed = False
    for c in cases:
        cfg = case_config(args, run_file, c.benchmark, c.scene, scene_params)
        unknown = benchmark_run.unknown_parameters(cfg, defaults)
        failed = failed or bool(unknown)
        cg = config.merge(defaults, cfg)["Solver"]["Newton"]["LinearSolver"]
        seconds = budgets[(c.benchmark, c.case)][0]
        budget = f"fixed {seconds:g} s" if seconds is not None else "Newton's T_N"
        levels = ",".join(f"{v:g}" for v in results.levels_for(levels_table, c.benchmark))
        print(f"{c.benchmark + '/' + c.case:28s} {c.set_name:30s} {c.batch.n_samples:3d} samples, levels {levels}, "
              f"budget {budget}; Newton CG n_max_iters {cg['n_max_iters']}, rtol {cg['rtol']:g}"
              + (f"; UNKNOWN KEYS {', '.join(unknown)}" if unknown else ""))
    if failed:
        raise SystemExit("Unknown solver parameters (see above)")


def summarize(log: static.StaticSolveLog, levels, budget: float) -> dict:
    return {
        "e0": float(log.error[0]),
        "iterations": int(log.error.size - 1),
        "sweeps": int(log.sweeps[-1]),
        "solver_time": float(log.time[-1]),
        "final_relative_error": float(log.relative_error[-1]),
        "termination": log.termination,
        "crossings": results.level_crossings(log.relative_error, log.time, levels, budget),
    }


def newton_step_arrays(records) -> dict:
    """Newton's per-step records of one solve (``BenchmarkSolver.newton_records``) as arrays for
    ``logs/<sample>/newton/``."""
    return {
        "newton/status": np.array([r.status.name.lower() for r in records], dtype=str),
        "newton/cg_iterations": np.array([r.max_cg_iters for r in records], dtype=np.int64),
        "newton/cg_capped": np.array([r.cg_capped for r in records], dtype=bool),
        "newton/cg_residual": np.array([r.cg_residual for r in records], dtype=np.float64),
        "newton/cg_tolerance": np.array([r.cg_tolerance for r in records], dtype=np.float64),
        "newton/backtracks": np.array([r.backtracks for r in records], dtype=np.int64),
    }


def snapshot_frames(args) -> int:
    """Frames per trajectory: --snapshot-frames, else video seconds x fps (0 without --snapshots)."""
    if not args.snapshots:
        return 0
    if args.snapshot_frames is not None:
        return max(1, int(args.snapshot_frames))
    return max(1, int(round(args.snapshot_video_seconds * args.snapshot_fps)))


def snapshot_samples(fraction: float, indices: list[int]) -> list[int]:
    """Evenly spaced ``fraction`` of the sample indices (at least one; all for fraction >= 1)."""
    if not indices or fraction <= 0:
        return []
    k = min(len(indices), max(1, int(round(fraction * len(indices)))))
    picks = np.unique(np.round(np.linspace(0, len(indices) - 1, k)).astype(np.int64))
    return [indices[i] for i in picks]


def trajectory(
    ring: static.SnapshotRing,
    log: static.StaticSolveLog,
    crossings,
    n_frames: int,
    budget: float,
    video_seconds: float,
) -> dict:
    """The solve's video frames from its snapshots: ring snapshots nearest to evenly spaced solver times
    (benchmark_results.select_frames), and each reached level's crossing (its own snapshot when the device
    detected the same entry, else the nearest ring snapshot)."""
    n = log.error.size
    entries, slots = ring.stored()
    keep = entries < n  # not the iterations a block ran past the stop
    entries, slots = entries[keep], slots[keep]
    # Frames over the problem's budget, and not past it (Newton runs on past T_N to its stall test)
    picks, targets = results.select_frames(log.time, entries, n_frames, budget, return_targets=True, end=budget)
    frames = {}
    target_of = {}
    for i, target in zip(picks, targets):
        frames[int(entries[i])] = ("ring", int(slots[i]))
        target_of.setdefault(int(entries[i]), target)
    level_entry = ring.level_entries()
    L = len(crossings.reached)
    level_at = np.full(L, -1, dtype=np.int64)
    level_exact = np.zeros(L, dtype=bool)
    for l in range(L):
        if not crossings.reached[l] or crossings.time[l] > budget:
            continue
        e = int(crossings.iteration[l])
        if level_entry[l] == e:
            frames[e] = ("level", l)
            level_at[l], level_exact[l] = e, True
        elif entries.size:
            i = int(np.abs(entries - e).argmin())
            frames.setdefault(int(entries[i]), ("ring", int(slots[i])))
            level_at[l] = entries[i]
    order = sorted(frames)
    x = np.zeros((len(order), ring.x.shape[1], 3), dtype=np.float32)
    for kind, fetch in (("ring", ring.fetch), ("level", ring.fetch_levels)):
        rows = [j for j, e in enumerate(order) if frames[e][0] == kind]
        if rows:
            x[rows] = fetch([frames[order[j]][1] for j in rows])
    index = {e: j for j, e in enumerate(order)}
    return {
        "iteration": np.array(order, dtype=np.int64),
        "t": log.time[np.array(order, dtype=np.int64)],
        # The solver time each frame stands for (NaN for frames kept only as level crossings)
        "t_target": np.array([target_of.get(e, np.nan) for e in order], dtype=np.float64),
        "x": x,
        "level_frames": np.array([index[e] if e >= 0 else -1 for e in level_at], dtype=np.int64),
        "level_exact": level_exact,
        "requested_frames": int(n_frames),
        "ring_slots": int(ring.slots),
        "stride": int(ring.state.numpy()[1]),
        "device_bytes": int(ring.device_bytes),
        "video_seconds": float(video_seconds),
        # The video's slow-down relative to solver time: video seconds per second of solver time
        "playback_factor": float(video_seconds / budget) if budget > 0 and math.isfinite(budget) else math.nan,
    }


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    """Whether the checkout has uncommitted changes, i.e. the code that ran differs from git_commit."""
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
    except Exception:
        return False


def software_versions() -> dict:
    """Versions of what the timings depend on: Warp, the CUDA driver and toolkit versions Warp reports, and
    pbatoolkit ("" if its package metadata is missing)."""
    import importlib.metadata

    try:
        pbatoolkit = importlib.metadata.version("pbatoolkit")
    except importlib.metadata.PackageNotFoundError:
        pbatoolkit = ""
    return {
        "version_warp": wp.__version__,
        "version_cuda_driver": ".".join(str(v) for v in wp.get_cuda_driver_version()),
        "version_cuda_toolkit": ".".join(str(v) for v in wp.get_cuda_toolkit_version()),
        "version_pbatoolkit": pbatoolkit,
    }


def _attr(v):
    return v.item() if isinstance(v, np.generic) else v


def main():
    args = parse_args()
    try:
        run_file = benchmark_run.read_run_file(args.run) if args.run else None
    except (OSError, ValueError) as e:
        raise SystemExit(str(e))
    benchmark_run.apply_run_options(args, run_file.run if run_file else {}, DEFAULTS)
    try:
        levels_table = results.parse_levels_options(args.levels)
    except ValueError as e:
        raise SystemExit(str(e))
    spec = make_spec(args)
    if not os.path.isfile(spec.dataset):
        raise SystemExit(f"Dataset not found: {spec.dataset} (relative paths are relative to the current directory; "
                         "pass --dataset path/to/dataset.h5, e.g. the main checkout's data/benchmarks/ from a worktree)")
    if args.list_cases:
        try:
            fixed_budgets = results.parse_budget_options(args.budget, "--budget")
        except ValueError as e:
            raise SystemExit(str(e))
        print_sets(list_sets(spec.dataset, select_sets(spec)), run_file, levels_table, fixed_budgets,
                   results.parse_scene_params(args.scene_params), args)
        return
    if args.save_spec:
        with h5py.File(args.save_spec, "w") as f:
            spec.serialize(f.create_group("StaticSolveSpec"))
    solvers = [SolverType[name] for name in args.solvers.split(",") if name]
    cases = make_cases(spec.dataset, select_sets(spec))
    try:
        scene_params = results.parse_scene_params(args.scene_params)
    except ValueError as e:
        raise SystemExit(str(e))
    unknown = results.unmatched_scene_params(scene_params, [(c.benchmark, c.scene) for c in cases])
    if unknown:
        raise SystemExit(f"--scene-params for scenes not in the run: {', '.join(unknown)}")
    # Run file cases may name problems --sets leaves out, but each must exist in the dataset (typos fail)
    unknown = benchmark_run.unmatched_cases(run_file, dataset_benchmark_scenes(spec.dataset)) if run_file else []
    if unknown:
        raise SystemExit(f"{args.run}: cases for scenes not in {spec.dataset}: {', '.join(unknown)}")
    unknown = sorted({k for k in levels_table if k is not None} - {c.benchmark for c in cases})
    if unknown:
        raise SystemExit(f"--levels for benchmarks not in the run: {', '.join(unknown)}")
    # Budgets per (benchmark, case): a fixed --budget, else Newton's T_N per problem
    try:
        fixed = results.parse_budget_options(args.budget, "--budget")
    except ValueError as e:
        raise SystemExit(str(e))
    budgets = {(c.benchmark, c.case): results.fixed_budget(fixed, c.benchmark, c.case) for c in cases}
    newton_cases = [f"{b}/{c}" for (b, c), (seconds, _) in budgets.items() if seconds is None]
    if newton_cases and SolverType.Newton not in solvers:
        raise SystemExit(f"Budgets come from Newton's solves, but Newton isn't in --solvers: add it, or give "
                         f"{', '.join(newton_cases)} a fixed --budget")
    if args.dry_run:
        dry_run(args, spec, run_file, cases, levels_table, budgets, scene_params)
        return
    wp.init()
    with h5py.File(spec.dataset, "r") as f:
        dataset_attrs = {k: _attr(v) for k, v in f.attrs.items()}

    total = len(solvers) * sum(len(spec.sample_indices(c.batch.n_samples)) for c in cases)
    n_frames = snapshot_frames(args)
    if n_frames:
        # Video duration of the trajectories: the frames over the budget, at the video's frame rate
        video_seconds = (args.snapshot_video_seconds if args.snapshot_frames is None
                         else n_frames / args.snapshot_fps)
        n_levels = max(len(results.levels_for(levels_table, c.benchmark)) for c in cases)
        frames_max = n_frames + 2 + n_levels  # the targets, the end, the level crossings
        estimate = sum(
            len(snapshot_samples(args.snapshot_fraction, list(spec.sample_indices(c.batch.n_samples))))
            * len(solvers) * frames_max * 12 * c.batch.X.shape[0]
            for c in cases
        )
        print(f"Snapshots: {n_frames} frames per trajectory ({video_seconds:g} s of video), at most "
              f"~{estimate / 2**30:.2f} GiB on disk for this run")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, "w") as h5f, tqdm(total=total, desc="Solves", unit="solve") as pbar:
        h5f.attrs.update(
            format=results.FORMAT,
            version=results.VERSION,
            levels_options=" ".join(f"--levels {v}" for v in args.levels),
            budget_factor=args.budget_factor,
            budget_options=" ".join(f"--budget {b}" for b in args.budget),
            newton_stall_iterations=spec.newton_stall_iterations,
            newton_stall_reduction=spec.newton_stall_reduction,
            dataset=spec.dataset,
            device=wp.get_device().name,
            git_commit=_git_commit(),
            git_dirty=_git_dirty(),
            **software_versions(),
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            sim_params=" + ".join(([f"--run {args.run}"] if args.run else []) + args.sim_params
                                  + [f"--set {o}" for o in args.overrides]) or "(code defaults)",
            run_file=args.run or "",
            run_toml=run_file.text if run_file else "",
            run_toml_resolved=config.dumps(run_file.resolved) if run_file else "",
            solvers=[s.name for s in solvers],
        )
        spec.serialize(h5f.create_group("StaticSolveSpec"))
        h5f.create_group("dataset_attrs").attrs.update(
            {k: v for k, v in dataset_attrs.items() if isinstance(v, (int, float, str, bool))}
        )
        # Case groups in dataset order, so the figures' axes follow it
        groups = {}
        for c in cases:
            seconds, source = budgets[(c.benchmark, c.case)]
            if seconds is None:
                source = (f"{args.budget_factor:g} x Newton's time to its highest level (T_N) per problem; Newton "
                          f"at most {spec.max_seconds:g} s")
            groups[c.set_name] = results.require_case(h5f, c.benchmark, c.case, {
                "benchmark": c.benchmark,
                "problem": c.batch.problem,
                "h": c.batch.h if c.batch.h is not None else math.nan,
                "scene": c.scene,
                "label": c.label,
                "energy": c.batch.target_energy,
                "set": c.set_name,
                "budget": seconds,  # the fixed budget; per problem otherwise (each solver's "budget")
                "budget_source": source,
                "max_seconds": spec.max_seconds,
                "scene_params": ",".join(results.scene_params_for(scene_params, c.benchmark, c.scene)),
            }, levels=results.levels_for(levels_table, c.benchmark))
        # One simulation per (scene, time step) at a time, reused across its sets
        keys = list(dict.fromkeys((c.scene, c.fem, c.batch.h, c.benchmark) for c in cases))
        for scene, fem, h, benchmark in keys:
            state = make_state(
                spec.dataset, fem, h, case_config(args, run_file, benchmark, scene, scene_params),
                f"{benchmark}/{scene} parameters",
            )
            if h is None:
                static.prepare_static_problem(state, spec.fext_scale)
            cg = state.newton_linear_solver_params
            for c in (c for c in cases if (c.scene, c.fem, c.batch.h, c.benchmark) == (scene, fem, h, benchmark)):
                batch, group = c.batch, groups[c.set_name]
                # Newton's CG limits, per case: --scene-params may set them per scene and benchmark
                group.attrs.update(newton_cg_n_max_iters=cg.n_max_iters, newton_cg_rtol=cg.rtol, newton_cg_atol=cg.atol)
                # Every solver parameter of the case's simulation, as configured (before the per-solve
                # overrides the module docstring lists), so the run can be reproduced from the file alone
                group.attrs["sim_params_toml"] = config.dumps(config.params_dict(state))
                static.check_sample_batch(batch, state.fem)
                for w in batch.warnings():
                    print(f"Warning: {batch.path}: {w}")
                levels = results.levels_for(levels_table, c.benchmark)
                fixed_seconds = budgets[(c.benchmark, c.case)][0]
                target = args.target_reduction if args.target_reduction is not None else levels[-1]
                case_spec = copy_spec(spec, target_reduction=target,
                                      max_seconds=fixed_seconds if fixed_seconds is not None else spec.max_seconds)
                indices = list(case_spec.sample_indices(batch.n_samples))
                # Newton first: its times T_N set the other solvers' budgets, per problem
                order = solvers
                if fixed_seconds is None:
                    order = [SolverType.Newton] + [s for s in solvers if s != SolverType.Newton]
                newton_time = {}
                for solver in order:
                    ring = None
                    if n_frames:
                        n_nodes = state.fem.data.x.shape[0]
                        slots = static.SnapshotRing.slots_for(
                            n_frames, n_nodes, len(levels), int(args.snapshot_memory_mb * 2**20),
                            args.snapshot_ring_ratio,
                        )
                        ring = static.SnapshotRing(n_nodes, slots, levels)
                        if slots < args.snapshot_ring_ratio * n_frames:
                            print(f"Warning: {args.snapshot_memory_mb:g} MB holds {slots} ring slots of {n_nodes} "
                                  f"nodes, fewer than {args.snapshot_ring_ratio:g} x {n_frames} frames: frames fall "
                                  "up to a ring stride from their target times")
                    runner = static.BenchmarkSolver(state, solver, case_spec, snapshots=ring)
                    summaries, logs, trajectories = [], [], []
                    snapshot_set = set(snapshot_samples(args.snapshot_fraction, indices)) if n_frames else set()
                    for s in indices:
                        if fixed_seconds is not None:
                            budget = fixed_seconds
                        elif solver == SolverType.Newton:
                            budget = spec.max_seconds
                        else:
                            budget = args.budget_factor * newton_time[s]
                        t_host = time.perf_counter()
                        if batch.problem == "dynamic":
                            log = runner.solve(batch.x[s], batch.x_ref[s], batch.v[s], batch.xtilde[s], budget)
                        else:
                            log = runner.solve(batch.x[s], max_seconds=budget)
                        t_host = time.perf_counter() - t_host
                        summary = summarize(log, levels, budget)
                        if fixed_seconds is None and solver == SolverType.Newton:
                            tn, _ = results.normalization_time(summary["crossings"])
                            if not math.isfinite(tn):
                                tn = spec.max_seconds
                                print(f"Warning: {c.benchmark}/{c.case} sample {s}: Newton reached no level within "
                                      f"{spec.max_seconds:g} s; T_N = {tn:g} s")
                            newton_time[s] = tn
                        normalization = fixed_seconds if fixed_seconds is not None else newton_time[s]
                        summary["budget"] = budget
                        summary["normalization"] = normalization
                        summary["host_time"] = t_host
                        summary["sample"] = int(s)
                        summary["deformation_energy"] = float(batch.energy[s])
                        if batch.min_J is not None:
                            summary["min_J"] = float(batch.min_J[s])
                        if batch.problem == "dynamic":
                            bound = float(batch.ref_error_mnorm[s])
                            summary["reference_error_bound"] = bound
                            summary["reference_relative_error_bound"] = (
                                bound / summary["e0"] if summary["e0"] > 0 else math.inf
                            )
                        summaries.append(summary)
                        extra = newton_step_arrays(runner.newton_records) if solver == SolverType.Newton else {}
                        logs.append((log.error, log.time, log.sweeps, extra))
                        if ring is not None and s in snapshot_set:
                            span = normalization if fixed_seconds is not None else args.budget_factor * normalization
                            trajectories.append(trajectory(
                                ring, log, summary["crossings"], n_frames, span, video_seconds
                            ))
                        else:
                            trajectories.append(None)  # one entry per sample, so each is written under its own
                        pbar.set_postfix(case=f"{c.benchmark}/{c.case}", solver=solver.name,
                                         e=f"{summary['final_relative_error']:.2e}")
                        pbar.update(1)
                    results.write_solver(group, solver.name, levels, summaries, logs, trajectories)
                    h5f.flush()
                    del runner, ring
            del state
    print(f"Wrote {args.output}")


import unittest


class TestRecords(unittest.TestCase):
    def test_newton_step_arrays(self):
        from .gpu.newton.solver import NewtonSubproblemRecord, SubproblemStatus

        records = [
            NewtonSubproblemRecord(SubproblemStatus.STEPPED, 1, 500, True, 2, 1e-3, 1e-4),
            NewtonSubproblemRecord(SubproblemStatus.LINE_SEARCH_FAILED, 1, 120, False, 20),
        ]
        a = newton_step_arrays(records)
        self.assertEqual(list(a["newton/status"]), ["stepped", "line_search_failed"])
        np.testing.assert_array_equal(a["newton/cg_iterations"], [500, 120])
        np.testing.assert_array_equal(a["newton/cg_capped"], [True, False])
        np.testing.assert_array_equal(a["newton/backtracks"], [2, 20])
        self.assertEqual(a["newton/cg_residual"][0], 1e-3)
        self.assertTrue(np.isnan(a["newton/cg_tolerance"][1]))  # the record's default
        self.assertEqual(newton_step_arrays([])["newton/cg_iterations"].size, 0)

    def test_versions(self):
        wp.init()
        v = software_versions()
        self.assertEqual(set(v), {"version_warp", "version_cuda_driver", "version_cuda_toolkit", "version_pbatoolkit"})
        self.assertEqual(v["version_warp"], wp.__version__)
        self.assertRegex(v["version_cuda_driver"], r"^\d+\.\d+$")
        self.assertTrue(all(isinstance(x, str) for x in v.values()))


class TestListSets(unittest.TestCase):
    def test_synthetic(self):
        import tempfile

        rng = np.random.default_rng(0)
        N, T, S = 6, 4, 3
        mu = np.array([1.0, 1.0, 1.0, 100.0])  # two materials, nu = 0.25 (lambda = mu)
        x = rng.standard_normal((S, N, 3))
        x_ref = x + 0.1 * rng.standard_normal((S, N, 3))
        m = np.arange(1.0, N + 1.0)
        dmask = np.array([1, 0, 0, 0, 0, 1])
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "d.h5")
            with h5py.File(path, "w") as f:
                f.attrs.update(dataset_format=static.DATASET_FORMAT, sets=["S/static/E5", "S/dynamic_h0.01/E5"])
                f.create_group("S").attrs["label"] = "Scene S"
                fem = f.create_group("S/fem/pbat.sim.dynamics.FemElastoDynamics")
                fem.create_dataset("pbat.fem.Mesh/X", data=np.zeros((3, N)))
                fem.create_dataset("pbat.fem.Mesh/E", data=np.zeros((4, T), dtype=np.int64))
                fem.create_dataset("lamegU", data=np.stack([mu, mu]))
                fem.create_dataset("egU", data=np.arange(T)[:, None])
                fem.create_dataset("dmask", data=dmask[:, None])
                fem.create_dataset("m", data=m[:, None])
                for name, problem in [("S/static/E5", "static"), ("S/dynamic_h0.01/E5", "dynamic")]:
                    g = f.create_group(name)
                    g.attrs.update(problem=problem, h=0.01, scene="S", fem="S/fem", target_energy=5.0)
                    g.create_dataset("samples/x", data=x)
                    if problem == "dynamic":
                        g.create_dataset("samples/x_ref", data=x_ref)
                        g.create_dataset("samples/ref_error_mnorm", data=np.full(S, 1e-4))
            rows = list_sets(path, ["S/static/E5", "S/dynamic_h0.01/E5"])
        self.assertEqual([r["key"] for r in rows], ["static/S", "dynamic_h0.01/S"])
        r = rows[1]
        self.assertEqual((r["problem"], r["h"], r["scene"], r["label"], r["energy"], r["problems"]),
                         ("dynamic", 0.01, "S", "Scene S", 5.0, S))
        self.assertEqual((r["nodes"], r["tets"], r["dirichlet"]), (N, T, 2))
        self.assertEqual(r["materials"], "E 2.5 (3 tets), E 250 (1 tets); nu 0.25")  # E = 2 mu (1 + nu)
        free = dmask == 0
        e0 = np.sqrt(np.sum(m[None, free, None] * (x - x_ref)[:, free] ** 2, axis=(1, 2)))
        self.assertAlmostEqual(r["reference_accuracy"], float(np.median(1e-4 / e0)))
        self.assertNotIn("reference_accuracy", rows[0])

    def test_pilot_dataset(self):
        # The repo's git-ignored data/ (the main checkout), or PBAT_BENCHMARK_PILOT (e.g. from a worktree)
        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        path = os.environ.get("PBAT_BENCHMARK_PILOT", os.path.join(repo, "data", "benchmarks", "trampolines-pilot.h5"))
        if not os.path.isfile(path):
            self.skipTest("trampolines-pilot.h5 not found (set PBAT_BENCHMARK_PILOT to path/to/trampolines-pilot.h5)")
        (row,) = list_sets(path, ["het/static/E10000"])
        self.assertEqual(row["key"], "static/het")
        self.assertEqual(row["materials"], "E 1e6 (31539 tets), E 1e8 (10049 tets); nu 0.45")


if __name__ == "__main__":
    main()
