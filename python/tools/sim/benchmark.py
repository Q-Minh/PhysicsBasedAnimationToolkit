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
solver iteration (see ``static.py``). For accuracy levels l (error reductions 1 - e/e0, ``--levels``,
default 50, 60, 70, 80, 85, 90, 95, 99%), the benchmark records per solve the time each level is first
crossed and its incremental time: an iteration's time goes to the highest level it crosses, and the
levels it jumps over get none; levels crossed after the budget aren't reached (``benchmark_results.py``).
Only that error stops a solve, at
``--target-reduction`` (default: the highest level) or at its time budget or iteration cap; the solvers'
own convergence and feasibility tests are bypassed.

Time budgets, per case, in order of precedence (each case records its ``budget`` and ``budget_source``):

1. ``--budget [BENCHMARK[/CASE]=]SECONDS`` (repeatable), the most specific entry first: e.g.
   ``--budget static=5 --budget static/Y1e8=10``, or ``--budget 5`` for every case.
2. With ``--budget-from pilot.h5``: ``--budget-factor`` (default 1) times the ``--budget-statistic`` (default
   median) of Newton's time to the pilot's highest level in that case, at most ``--budget-cap
   [BENCHMARK[/CASE]=]SECONDS`` (repeatable, same matching). The pilot is an earlier results file with
   Newton, e.g. a Newton-only run with a generous ``--max-seconds``. The median is the default since
   Newton's times vary between runs (host synchronization every step) and a single slow sample would
   move the mean.
3. ``--max-seconds``: cases with neither, e.g. those where no pilot Newton solve reached the level (with a
   warning).

Usage:
    python -m python.tools.sim.benchmark --dataset DATASET.h5 [--sets PATTERN ...] -o results.h5 \
        [--solvers VBD,AAAVBD,Chebyshev,AndersonSolver,Newton] [--levels 0.5,0.6,...,0.99] \
        [--sim-params A.toml [B.h5 ...]] [--scene-params SCENE=C.toml[,D.toml] ...] [--set Table.key=value ...] \
        [--budget [BENCHMARK[/CASE]=]SECONDS ...] [--budget-from pilot.h5 [--budget-factor 1] \
        [--budget-statistic median] [--budget-cap [BENCHMARK[/CASE]=]SECONDS ...]] [--max-seconds 60] \
        [--spec spec.h5] [--save-spec spec.h5] [--sample-begin 0] [--sample-end -1] \
        [--max-iterations 1000000] [--target-reduction 0.99] [--block 16]

Representative commands (from the repo root; datasets live in the main checkout's data/benchmarks):

    # One static solve: one set, one solver, the first problem, a 5 s budget
    python -m python.tools.sim.benchmark --dataset data/benchmarks/trampoline-test.h5 \
        --sets static/E50 --solvers AAAVBD --sample-end 1 --max-seconds 5 -o out/one.h5

    # Pilot: Newton alone, generous budget; then the benchmark with budgets of Newton's median time to 99%,
    # except a fixed 5 s for the static benchmark, and at most 3 s elsewhere
    python -m python.tools.sim.benchmark --dataset DATASET.h5 --solvers Newton --max-seconds 60 \
        --sim-params python/tools/sim/configs/benchmark/shared.toml -o out/pilot.h5
    python -m python.tools.sim.benchmark --dataset DATASET.h5 --budget-from out/pilot.h5 --budget static=5 \
        --budget-cap 3 --sim-params python/tools/sim/configs/benchmark/shared.toml --scene-params Y1e8=y1e8.toml \
        -o out/results.h5

    # The figures
    python -m python.tools.sim.benchmark_plot out/results.h5 -o out/figures/benchmark

    # Parameter variations: layer configs and override single keys (see python/tools/sim/configs/README.md)
    python -m python.tools.sim.benchmark ... --sim-params base.toml static.toml \
        --set Solver.Newton.LinearSolver.n_max_iters=3000 --set Solver.AAAVBD.rhohat=0.9

    # Reproducible cases: save the effective StaticSolveSpec (dataset, sets, caps), then rerun from it
    python -m python.tools.sim.benchmark --dataset DATASET.h5 --sets static/E500 --max-seconds 10 --save-spec case.h5 -o a.h5
    python -m python.tools.sim.benchmark --spec case.h5 -o b.h5

Output: one HDF5 results file (layout in ``benchmark_results.py``'s docstring): the spec, levels, budgets,
dataset, device, commit and parameter sources; per benchmark, case and solver, each sample's crossing
times, increments, reached and skipped flags, e0, iterations, sweeps, solver time, host time (the solve's
wall-clock time, including loading and reading back), final relative error,
termination and full per-iteration log; and per level, the mean, std, min, max, median, quartiles and
count of the increments and crossing times over the samples that reached it. A solve's ``termination`` is
``target`` (reached --target-reduction), ``iteration_cap``, ``time_cap`` (its budget), ``non_finite`` (NaN or
infinite error), or for Newton ``newton_non_descent``, ``newton_line_search_failed`` or
``newton_zero_gradient``: Newton stops at its first rejected step, since x doesn't change after it.
Dynamic cases add the reference's accuracy (``reference_error_bound``, the set's ``samples/ref_error_mnorm``,
and ``reference_relative_error_bound``, over e0) and ``resolvable``, the levels that accuracy resolves.

Pitfalls:
    - The benchmark is contact-free: contact detection is deactivated whatever the parameters say.
    - The problems are posed on the dataset's FEM (its ``fem`` group, or the set's ``fem`` attribute): its
      Dirichlet nodes stay fixed at rest. Solver parameters come only from --sim-params, then the case's
      --scene-params, then --set (otherwise the code defaults, configs/defaults.toml).
    - Dynamic problems take their time step from the set: ``Integration.dt``, ``substeps`` and the BDF
      order from the parameters are overridden (one BDF1 step of size h).
    - At small h (e.g. 1e-4) a step starts close to its solution (e0 ~ 1e-3), so float32 rounding
      floors the error around e/e0 ~ 1e-5 to 1e-4. Newton usually ends there with
      ``newton_line_search_failed``: the energy decrease is below float32 resolution.
    - Solves run to the target reduction or the caps, not to a solver tolerance: ``Solver.<solver>.gtol``,
      ``n_max_iters`` and ``n_subproblem_max_iters`` don't stop them (a static solve is one long
      subproblem). Newton's CG keeps its relative tolerance, but its absolute one (``atol``) is set
      to 0: in h^2-scaled units at small h, |g| drops below it after one step. The vertex-solve and acceleration parameters do matter.
    - Newton isn't graph-captured. Each Newton step synchronizes with the host, so its times include host
      work and vary more between runs. Raise ``Solver.Newton.LinearSolver.n_max_iters`` for static problems:
      the default 500 leaves Newton steps inexact, and 1500-3000 is typical.
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
from . import config, static


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solver convergence benchmark (GPU) on the static and dynamic problem sets of a benchmark "
        "dataset: solver time per accuracy level of error reduction."
    )
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
    parser.add_argument("--scene-params", action="append", default=[], metavar="SCENE=A.toml[,B.toml]",
                        dest="scene_params",
                        help="Parameters for one scene's problems, merged after --sim-params and before --set. "
                        "Repeatable, once per scene.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="Override one parameter after the parameter files, as Table.Sub.key=TOML-literal, "
                        "e.g. Solver.Newton.LinearSolver.n_max_iters=3000. Repeatable.")
    parser.add_argument("--solvers", default=",".join(s.name for s in SolverType),
                        help="Comma-separated solver names to run, in order (default: all, "
                        + ",".join(s.name for s in SolverType) + ").")
    parser.add_argument("--levels", default=",".join(f"{v:g}" for v in results.DEFAULT_LEVELS),
                        help="Accuracy levels, as comma-separated error reductions 1 - e/e0 in (0, 1) "
                        "(default: %(default)s).")
    parser.add_argument("--budget", action="append", default=[], metavar="[BENCHMARK[/CASE]=]SECONDS",
                        help="Time budget per solve, in seconds of solver time, for every case, a benchmark (e.g. "
                        "static=5) or one case (static/Y1e8=10). Repeatable; the most specific entry wins. "
                        "Takes precedence over --budget-from, which takes precedence over --max-seconds.")
    parser.add_argument("--budget-from", default=None, dest="budget_from",
                        help="Pilot results file (.h5) with Newton's solves: the budget of a case without --budget "
                        "is --budget-factor times Newton's --budget-statistic time to the pilot's highest level, "
                        "at most --budget-cap. Cases the pilot lacks get --max-seconds.")
    parser.add_argument("--budget-factor", type=float, default=1.0, dest="budget_factor",
                        help="Multiple of the pilot's Newton time (default: %(default)s).")
    parser.add_argument("--budget-cap", action="append", default=[], metavar="[BENCHMARK[/CASE]=]SECONDS",
                        dest="budget_cap",
                        help="Upper limit on the pilot-derived budgets, matched like --budget. Repeatable.")
    parser.add_argument("--budget-statistic", choices=results.BUDGET_STATISTICS, default="median",
                        dest="budget_statistic",
                        help="Statistic of the pilot's Newton times over its samples that reached the level "
                        "(default: %(default)s).")
    parser.add_argument("--spec", default=None,
                        help="StaticSolveSpec (.h5) to start from. The options below override its fields.")
    parser.add_argument("--save-spec", default=None, dest="save_spec",
                        help="Write the effective StaticSolveSpec (.h5), to rerun the same case with --spec.")
    parser.add_argument("--sample-begin", type=int, default=None, dest="sample_begin",
                        help="First sample index (spec default: 0).")
    parser.add_argument("--sample-end", type=int, default=None, dest="sample_end",
                        help="One past the last sample index; -1 means all samples (spec default: -1).")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                        help="Time budget per solve of the cases without --budget or a pilot time, in seconds of "
                        "solver time (spec default: 60).")
    parser.add_argument("--max-iterations", type=int, default=None, dest="max_iterations",
                        help="Iteration cap per solve, counting VBD sweeps, Anderson steps or Newton steps "
                        "(default here: 1,000,000, so the time budget decides).")
    parser.add_argument("--target-reduction", type=float, default=None, dest="target_reduction",
                        help="Stop a solve once e <= (1 - target) e0, as a fraction (default: the highest level).")
    parser.add_argument("--block", type=int, default=None,
                        help="Solver iterations per replayed CUDA-graph block, rounded up to a multiple of "
                        "Anderson's window m. Larger blocks mean fewer replays; iterations past the stop are "
                        "dropped (spec default: 16).")
    parser.add_argument("-o", "--output", required=True, help="Output results file (.h5).")
    return parser.parse_args()


def make_spec(args, levels) -> static.StaticSolveSpec:
    spec = static.StaticSolveSpec()
    spec.max_iterations = 1_000_000
    spec.target_reduction = levels[-1]
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


def parse_scene_params(entries: list[str]) -> dict[str, list[str]]:
    """``SCENE=A.toml,B.toml`` entries -> {scene: [A.toml, B.toml]}."""
    out = {}
    for entry in entries:
        scene, sep, files = entry.partition("=")
        if not sep or not scene or not files:
            raise SystemExit(f"--scene-params expects SCENE=A.toml[,B.toml], got {entry!r}")
        if scene in out:
            raise SystemExit(f"--scene-params given twice for scene {scene!r}")
        out[scene] = [p for p in files.split(",") if p]
    return out


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


def _scene_of(dataset: str, f: h5py.File, batch: static.SampleBatch) -> tuple[str, str, str]:
    """(scene, label, FEM group) of a batch: its ``scene``/``fem`` attributes, else (version-1 datasets)
    the dataset's ``input`` scene file stem and the root FEM."""
    scene = batch.attrs.get("scene")
    if scene is None:
        source = f.attrs.get("input") or dataset
        scene = os.path.splitext(os.path.basename(str(source).replace("\\", "/")))[0]
        return str(scene), str(scene), "fem"
    scene = str(scene)
    label = f[scene].attrs.get("label", scene) if scene in f else scene
    return scene, str(label), str(batch.attrs.get("fem", f"{scene}/fem"))


def make_cases(dataset: str, set_names: list[str]) -> list[Case]:
    cases = []
    with h5py.File(dataset, "r") as f:
        for name in set_names:
            batch = static.load_sample_batch(f"{dataset}:{name}")
            scene, label, fem = _scene_of(dataset, f, batch)
            cases.append(Case(name, batch, scene, label, fem, results.benchmark_name(batch.problem, batch.h)))
    for c in cases:
        energies = {o.batch.target_energy for o in cases if (o.benchmark, o.scene) == (c.benchmark, c.scene)}
        c.case = c.scene if len(energies) == 1 else f"{c.scene}_E{c.batch.target_energy:g}"
    return cases


def make_state(args, dataset: str, fem: str, h: float | None, sim_params: list[str]) -> SimulationState:
    """Build the contact-free simulation on a FEM of the dataset; ``h`` set means one-step dynamic problems
    of that step."""
    # The dataset's FEM has no kinematics group: its Dirichlet nodes stay fixed at rest
    # (check_sample_batch checks the samples against that), and the benchmark is contact-free.
    fem_cpu, _, _ = load_problem(f"{dataset}:{fem}")
    params_cpu = load_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu)
    if sim_params or args.overrides:
        config.load_sim_params(state, sim_params, args.overrides)
    for cd_params in state.cd_params.values():
        if hasattr(cd_params, "deactivate"):
            cd_params.deactivate = True
    state.cd_type = CDType.VertexSdf
    if h is None:
        state.reset()
    else:
        static.prepare_dynamic_problem(state, h)
    return state


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


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _attr(v):
    return v.item() if isinstance(v, np.generic) else v


def main():
    args = parse_args()
    levels = results.parse_levels(args.levels)
    wp.init()
    spec = make_spec(args, levels)
    if args.save_spec:
        with h5py.File(args.save_spec, "w") as f:
            spec.serialize(f.create_group("StaticSolveSpec"))
    solvers = [SolverType[name] for name in args.solvers.split(",") if name]
    cases = make_cases(spec.dataset, select_sets(spec))
    scene_params = parse_scene_params(args.scene_params)
    unknown = set(scene_params) - {c.scene for c in cases}
    if unknown:
        raise SystemExit(f"--scene-params for scenes not in the run: {', '.join(sorted(unknown))}")
    # Budgets per (benchmark, case): --budget, else the pilot's (capped), else --max-seconds
    try:
        direct = results.parse_budget_options(args.budget, "--budget")
        caps = results.parse_budget_options(args.budget_cap, "--budget-cap")
    except ValueError as e:
        raise SystemExit(str(e))
    pilot = {}
    pilot_label = ""
    if args.budget_from:
        pilot = results.budgets_from_pilot(args.budget_from, args.budget_factor, args.budget_statistic)
        pilot_label = f"{args.budget_factor:g} x {args.budget_statistic} Newton time in {args.budget_from}"
    budgets = {}
    for c in cases:
        budgets[(c.benchmark, c.case)] = results.resolve_budget(
            c.benchmark, c.case, direct, caps, pilot, spec.max_seconds, pilot_label
        )
    missing = [f"{b}/{c}" for (b, c), (_, source) in budgets.items() if args.budget_from and source == "--max-seconds"]
    if missing:
        print(f"Warning: {args.budget_from} has no Newton solve reaching its highest level for "
              f"{', '.join(missing)}; their budget is --max-seconds ({spec.max_seconds:g} s)")
    with h5py.File(spec.dataset, "r") as f:
        dataset_attrs = {k: _attr(v) for k, v in f.attrs.items()}

    total = len(solvers) * sum(len(spec.sample_indices(c.batch.n_samples)) for c in cases)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, "w") as h5f, tqdm(total=total, desc="Solves", unit="solve") as pbar:
        h5f.attrs.update(
            format=results.FORMAT,
            version=results.VERSION,
            levels=np.array(levels),
            budget_factor=args.budget_factor if args.budget_from else math.nan,
            budget_statistic=args.budget_statistic if args.budget_from else "",
            budget_pilot=args.budget_from or "",
            budget_options=" ".join([f"--budget {b}" for b in args.budget] + [f"--budget-cap {b}" for b in args.budget_cap]),
            dataset=spec.dataset,
            device=wp.get_device().name,
            git_commit=_git_commit(),
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            sim_params=" + ".join(args.sim_params + [f"--set {o}" for o in args.overrides]) or "(code defaults)",
            solvers=[s.name for s in solvers],
        )
        spec.serialize(h5f.create_group("StaticSolveSpec"))
        h5f.create_group("dataset_attrs").attrs.update(
            {k: v for k, v in dataset_attrs.items() if isinstance(v, (int, float, str, bool))}
        )
        # Case groups in dataset order, so the figures' axes follow it
        groups = {}
        for c in cases:
            budget = pilot.get((c.benchmark, c.case))
            seconds, source = budgets[(c.benchmark, c.case)]
            groups[c.set_name] = results.require_case(h5f, c.benchmark, c.case, {
                "benchmark": c.benchmark,
                "problem": c.batch.problem,
                "h": c.batch.h if c.batch.h is not None else math.nan,
                "scene": c.scene,
                "label": c.label,
                "energy": c.batch.target_energy,
                "set": c.set_name,
                "budget": seconds,
                "budget_source": source,
                "budget_reference_time": budget["reference_time"] if budget else None,
                "budget_reference_reached": f"{budget['reached']}/{budget['n_samples']}" if budget else None,
                "scene_params": ",".join(scene_params.get(c.scene, [])),
            })
        # One simulation per (scene, time step) at a time, reused across its sets
        keys = list(dict.fromkeys((c.scene, c.fem, c.batch.h) for c in cases))
        for scene, fem, h in keys:
            state = make_state(args, spec.dataset, fem, h, args.sim_params + scene_params.get(scene, []))
            if h is None:
                static.prepare_static_problem(state, spec.fext_scale)
            if "newton_cg_n_max_iters" not in h5f.attrs:
                cg = state.newton_linear_solver_params
                h5f.attrs.update(newton_cg_n_max_iters=cg.n_max_iters, newton_cg_rtol=cg.rtol, newton_cg_atol=cg.atol)
            for c in (c for c in cases if (c.scene, c.fem, c.batch.h) == (scene, fem, h)):
                batch, group = c.batch, groups[c.set_name]
                static.check_sample_batch(batch, state.fem)
                for w in batch.warnings():
                    print(f"Warning: {batch.path}: {w}")
                case_spec = copy_spec(spec, max_seconds=float(group.attrs["budget"]))
                indices = list(case_spec.sample_indices(batch.n_samples))
                for solver in solvers:
                    runner = static.BenchmarkSolver(state, solver, case_spec)
                    summaries, logs = [], []
                    for s in indices:
                        t_host = time.perf_counter()
                        if batch.problem == "dynamic":
                            log = runner.solve(batch.x[s], batch.x_ref[s], batch.v[s], batch.xtilde[s])
                        else:
                            log = runner.solve(batch.x[s])
                        t_host = time.perf_counter() - t_host
                        summary = summarize(log, levels, case_spec.max_seconds)
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
                        logs.append((log.error, log.time, log.sweeps))
                        pbar.set_postfix(case=f"{c.benchmark}/{c.case}", solver=solver.name,
                                         e=f"{summary['final_relative_error']:.2e}")
                        pbar.update(1)
                    results.write_solver(group, solver.name, levels, summaries, logs)
                    h5f.flush()
            del state
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
