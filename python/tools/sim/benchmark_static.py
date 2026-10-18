# type: ignore
r"""
Static convergence benchmark: wall-clock time per percentile of error reduction.

Each MCMC sample of a batch (python/examples/rom_mcmc.py; contract in plans/MASTER.md) is released into
a static solve with no loads, whose equilibrium is the rest shape X, once per solver. The error
e = |x - X|_M (lumped-mass norm over free nodes) is logged after every solver iteration (see
``static.py``). The benchmark reports, per solver and percentile p = 5%, 10%, ..., 95%, the solver time
until e <= (1 - p) e0. A solve stops at ``--target-reduction`` or at its time/iteration budget, so
unreached percentiles are reported as missing (``null``).

Usage:
    python -m python.tools.sim.benchmark_static --fem-elasto-dynamics DUMP.h5:fem --samples BATCH.h5 \
        -o results.json [--solvers VBD,AAAVBD,Chebyshev,AndersonSolver,Newton] \
        [--sim-params A.toml [B.h5 ...]] [--set Table.key=value ...] [--spec spec.h5] [--save-spec spec.h5] \
        [--sample-begin 0] [--sample-end -1] [--max-seconds 60] [--max-iterations 1000000] \
        [--target-reduction 0.95] [--block 16]

Representative commands (from the repo root; scenes and samples live in the main checkout's data/):

    # One static solve: one solver, the first sample, a 5 s cap
    python -m python.tools.sim.benchmark_static \
        --fem-elasto-dynamics C:/git/PhysicsBasedAnimationToolkit/data/scenarios/trampoline.Dump.h5:fem \
        --samples C:/git/PhysicsBasedAnimationToolkit/data/benchmarks/trampoline-test/static/E50.h5 \
        --solvers AAAVBD --sample-end 1 --max-seconds 5 -o out/one.json

    # Every solver on every sample of a batch, with the scene's paper parameters
    python -m python.tools.sim.benchmark_static --fem-elasto-dynamics DUMP.h5:fem --samples BATCH.h5 \
        --sim-params python/tools/sim/configs/paper/beam1e6.GroundTruth.toml --max-seconds 10 -o out/all.json

    # Parameter variations: layer configs and override single keys (see python/tools/sim/configs/README.md)
    python -m python.tools.sim.benchmark_static ... --sim-params base.toml static.toml \
        --set Solver.Newton.LinearSolver.n_max_iters=3000 --set Solver.AAAVBD.rhohat=0.9

    # Reproducible cases: save the effective StaticSolveSpec, then rerun from it
    python -m python.tools.sim.benchmark_static ... --samples BATCH.h5 --max-seconds 10 --save-spec case.h5 -o a.json
    python -m python.tools.sim.benchmark_static --fem-elasto-dynamics DUMP.h5:fem --spec case.h5 -o b.json

Outputs:
    ``results.json``: the effective spec; metadata (scene, parameter sources, sample-file attributes,
    Newton CG limits, device, git commit, date); and per solver, one summary per sample (e0, iterations,
    VBD sweeps, solver time, final relative error, time and iterations to each percentile, starting
    deformation energy, min det F, and a log-spaced curve of relative error vs time and sweeps), plus
    aggregates per percentile (median and quartiles of the time over the samples that reached it, and
    how many did).
    ``results.h5`` (same name, next to it): the spec and the full per-iteration logs,
    ``<solver>/<sample>/{error, time, sweeps}``.

Pitfalls:
    - The benchmark is contact-free: contact detection is deactivated whatever the parameters say.
    - Solves run to the target reduction or the caps, not to a solver tolerance: ``Solver.<solver>.gtol``,
      ``n_max_iters`` and ``n_subproblem_max_iters`` don't stop them (a static solve is one long
      subproblem). The vertex-solve and acceleration parameters do matter.
    - Newton isn't graph-captured. Each Newton step synchronizes with the host, so its times include host
      work and vary more between runs. Raise ``Solver.Newton.LinearSolver.n_max_iters`` for static problems:
      the default 300 leaves Newton steps inexact, and 1500-3000 is typical.
    - Times are GPU wall-clock from device timestamps, with the logging excluded. Don't run other GPU work
      meanwhile.
    - Batches with ``importance_weights_required`` are refused; R-hat >= 1.05 or nonzero sampling gravity
      only print a warning.
    - Each solver's first solve also compiles kernels and captures its graph. That shows in the host time,
      not in the reported solver time.
"""

import argparse
import json
import math
import os
import subprocess
import time

import h5py
import numpy as np
import warp as wp
from tqdm import tqdm

from .state import (
    SimulationState,
    SolverType,
    CDType,
    load_fem_dynamics,
    load_vbd_params,
    load_chebyshev_params,
    load_anderson_params,
    load_newton_params,
)
from . import config, static

PERCENTILES = [p / 100 for p in range(5, 100, 5)]
_CURVE_POINTS = 400


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static convergence benchmark (GPU): solver time per 5% level of error reduction."
    )
    parser.add_argument("--fem-elasto-dynamics", required=True, dest="fem_elasto_dynamics",
                        help="Scene: file.h5:group/path of the serialized FemElastoDynamics, e.g. an editor "
                        "Dump's 'fem' group. Must be the mesh the samples were drawn on.")
    parser.add_argument("--samples", default=None,
                        help="MCMC sample batch (.h5) with the initial shapes. Overrides the spec's "
                        "sample_file; required unless --spec provides one.")
    parser.add_argument("--sim-params", nargs="+", default=[], dest="sim_params",
                        help="Simulation parameters: .toml configs or .h5 files as exported by the GUI "
                        "(file.h5 or file.h5:group), merged left to right (see configs/README.md). "
                        "Without any, the code defaults apply (configs/defaults.toml).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="Override one parameter after --sim-params, as Table.Sub.key=TOML-literal, "
                        "e.g. Solver.Newton.LinearSolver.n_max_iters=3000. Repeatable.")
    parser.add_argument("--solvers", default=",".join(s.name for s in SolverType),
                        help="Comma-separated solver names to run, in order (default: all, "
                        + ",".join(s.name for s in SolverType) + ").")
    parser.add_argument("--spec", default=None,
                        help="StaticSolveSpec (.h5) to start from. The options below override its fields.")
    parser.add_argument("--save-spec", default=None, dest="save_spec",
                        help="Write the effective StaticSolveSpec (.h5), to rerun the same case with --spec.")
    parser.add_argument("--sample-begin", type=int, default=None, dest="sample_begin",
                        help="First sample index (spec default: 0).")
    parser.add_argument("--sample-end", type=int, default=None, dest="sample_end",
                        help="One past the last sample index; -1 means all samples (spec default: -1).")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds",
                        help="Time cap per solve, in seconds of wall-clock time (spec default: 60).")
    parser.add_argument("--max-iterations", type=int, default=None, dest="max_iterations",
                        help="Iteration cap per solve, counting VBD sweeps, Anderson steps or Newton steps "
                        "(default here: 1,000,000, so the time cap decides).")
    parser.add_argument("--target-reduction", type=float, default=None, dest="target_reduction",
                        help="Stop a solve once e <= (1 - target) e0, as a fraction (spec default: 0.95).")
    parser.add_argument("--block", type=int, default=None,
                        help="Solver iterations per replayed CUDA-graph block, rounded up to a multiple of "
                        "Anderson's window m. Larger blocks mean fewer replays but can overshoot the stop "
                        "by up to one block (spec default: 16).")
    parser.add_argument("-o", "--output", required=True,
                        help="Output results.json; results.h5 with the full logs is written next to it.")
    return parser.parse_args()


def make_spec(args) -> static.StaticSolveSpec:
    spec = static.StaticSolveSpec()
    spec.max_iterations = 1_000_000
    if args.spec:
        with h5py.File(args.spec, "r") as f:
            spec.deserialize(f["StaticSolveSpec"] if "StaticSolveSpec" in f else f)
    for name, value in [
        ("sample_file", args.samples),
        ("sample_begin", args.sample_begin),
        ("sample_end", args.sample_end),
        ("max_seconds", args.max_seconds),
        ("max_iterations", args.max_iterations),
        ("target_reduction", args.target_reduction),
        ("block", args.block),
    ]:
        if value is not None:
            setattr(spec, name, value)
    if not spec.sample_file:
        raise SystemExit("No samples: pass --samples or a --spec with sample_file")
    return spec


def make_state(args) -> SimulationState:
    fem_cpu = load_fem_dynamics(args.fem_elasto_dynamics)
    params_cpu = {s: load_vbd_params(fem_cpu) for s in [SolverType.VBD, SolverType.AAAVBD]}
    params_cpu[SolverType.Chebyshev] = load_chebyshev_params(fem_cpu)
    params_cpu[SolverType.AndersonSolver] = load_anderson_params(fem_cpu)
    params_cpu[SolverType.Newton] = load_newton_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu)
    if args.sim_params or args.overrides:
        config.load_sim_params(state, args.sim_params, args.overrides)
    # The static benchmark is contact-free.
    for cd_params in state.cd_params.values():
        if hasattr(cd_params, "deactivate"):
            cd_params.deactivate = True
    state.cd_type = CDType.VertexSdf
    state.reset()
    return state


def _log_spaced(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.unique(np.concatenate([[0], np.geomspace(1, n - 1, k - 1).astype(np.int64)]))


def _nan_to_none(v: float):
    return None if (v is None or not math.isfinite(v)) else float(v)


def summarize(log: static.StaticSolveLog) -> dict:
    idx = _log_spaced(log.error.size, _CURVE_POINTS)
    return {
        "e0": float(log.error[0]),
        "iterations": int(log.error.size - 1),
        "sweeps": int(log.sweeps[-1]),
        "time": float(log.time[-1]),
        "final_relative_error": float(log.relative_error[-1]),
        "time_to_reduction": {f"{p:.2f}": _nan_to_none(log.time_to_reduction(p)) for p in PERCENTILES},
        "iterations_to_reduction": {
            f"{p:.2f}": _nan_to_none(log.iterations_to_reduction(p)) for p in PERCENTILES
        },
        "curve": {
            "iteration": idx.tolist(),
            "time": log.time[idx].tolist(),
            "sweeps": log.sweeps[idx].tolist(),
            "relative_error": log.relative_error[idx].tolist(),
        },
    }


def aggregate(samples: list[dict]) -> dict:
    out = {}
    for p in PERCENTILES:
        key = f"{p:.2f}"
        t = np.array([s["time_to_reduction"][key] for s in samples if s["time_to_reduction"][key] is not None])
        out[key] = {
            "reached": int(t.size),
            "fraction_reached": float(t.size / len(samples)) if samples else 0.0,
            "median": _nan_to_none(float(np.median(t)) if t.size else math.nan),
            "q25": _nan_to_none(float(np.quantile(t, 0.25)) if t.size else math.nan),
            "q75": _nan_to_none(float(np.quantile(t, 0.75)) if t.size else math.nan),
        }
    return out


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main():
    args = parse_args()
    wp.init()
    spec = make_spec(args)
    if args.save_spec:
        with h5py.File(args.save_spec, "w") as f:
            spec.serialize(f.create_group("StaticSolveSpec"))
    solvers = [SolverType[name] for name in args.solvers.split(",") if name]
    state = make_state(args)
    batch = static.load_sample_batch(spec.sample_file)
    static.check_sample_batch(batch, state.fem)
    for w in batch.warnings():
        print(f"Warning: {spec.sample_file}: {w}")
    static.prepare_static_problem(state, spec.fext_scale)
    indices = list(spec.sample_indices(batch.n_samples))
    results = {
        "spec": spec.to_dict(),
        "percentiles": PERCENTILES,
        "metadata": {
            "fem_elasto_dynamics": args.fem_elasto_dynamics,
            "sim_params": " + ".join(args.sim_params + [f"--set {o}" for o in args.overrides])
            or "(Dump parameters)",
            "sample_attrs": {k: v for k, v in batch.attrs.items() if isinstance(v, (int, float, str, bool))},
            "device": wp.get_device().name,
            "git_commit": _git_commit(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "newton_linear_solver": {
                "n_max_iters": state.newton_linear_solver_params.n_max_iters,
                "rtol": state.newton_linear_solver_params.rtol,
                "atol": state.newton_linear_solver_params.atol,
            },
        },
        "solvers": {},
    }
    h5_path = os.path.splitext(args.output)[0] + ".h5"
    with h5py.File(h5_path, "w") as h5f:
        spec.serialize(h5f.create_group("StaticSolveSpec"))
        with tqdm(total=len(solvers) * len(indices), desc="Static solves", unit="solve") as pbar:
            for solver in solvers:
                runner = static.StaticSolver(state, solver, spec)
                samples = []
                for s in indices:
                    log = runner.solve(batch.x[s])
                    summary = summarize(log)
                    summary["sample"] = int(s)
                    summary["deformation_energy"] = float(batch.energy[s])
                    if batch.min_J is not None:
                        summary["min_J"] = float(batch.min_J[s])
                    samples.append(summary)
                    g = h5f.create_group(f"{solver.name}/{s:06d}")
                    g.create_dataset("error", data=log.error)
                    g.create_dataset("time", data=log.time)
                    g.create_dataset("sweeps", data=log.sweeps)
                    pbar.set_postfix(solver=solver.name, e=f"{summary['final_relative_error']:.2e}")
                    pbar.update(1)
                results["solvers"][solver.name] = {"samples": samples, "aggregate": aggregate(samples)}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {args.output} and {h5_path}")


if __name__ == "__main__":
    main()
