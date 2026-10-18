# type: ignore
"""
Static convergence benchmark: wall-clock time per percentile of error reduction.

Each MCMC sample of a batch (python/examples/rom_mcmc.py) is released into a static solve with no
loads, whose equilibrium is the rest shape X, once per solver. The error e = |x - X|_M (free nodes) is
logged after every solver iteration (see ``static.py``), and the benchmark reports, per solver and
percentile p in 5%, 10%, ..., 95%, the solver time until e <= (1 - p) e0. Solves stop at the target
reduction or at the time/iteration budget, so unreached percentiles are reported as missing.

Usage:
    python -m python.tools.sim.benchmark_static --fem-elasto-dynamics dump.h5:fem --samples batch.h5 \
        -o results.json [--sim-params static.h5] [--solvers VBD,AAAVBD,Chebyshev,AndersonSolver,Newton] \
        [--sample-begin 0] [--sample-end -1] [--max-seconds 60] [--max-iterations 1000000] \
        [--target-reduction 0.95] [--spec spec.h5] [--save-spec spec.h5]

Writes ``results.json`` (spec, metadata, per-sample percentile times and log-spaced curves, and
per-solver aggregates) and ``results.h5`` (the full per-iteration logs).
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
    deserialize_params,
)
from . import static

PERCENTILES = [p / 100 for p in range(5, 100, 5)]
_CURVE_POINTS = 400


def parse_args():
    parser = argparse.ArgumentParser(description="Static convergence benchmark (GPU)")
    parser.add_argument("--fem-elasto-dynamics", required=True, dest="fem_elasto_dynamics",
                        help="file.h5:group/path of the serialized FemElastoDynamics (Dump 'fem' group).")
    parser.add_argument("--samples", default=None, help="MCMC sample batch (.h5). Overrides the spec's sample_file.")
    parser.add_argument("--sim-params", default=None, dest="sim_params",
                        help="Simulation parameters (.h5), as exported by the GUI.")
    parser.add_argument("--solvers", default=",".join(s.name for s in SolverType),
                        help="Comma-separated solver names.")
    parser.add_argument("--spec", default=None, help="StaticSolveSpec (.h5) to start from.")
    parser.add_argument("--save-spec", default=None, dest="save_spec", help="Write the effective spec (.h5).")
    parser.add_argument("--sample-begin", type=int, default=None, dest="sample_begin")
    parser.add_argument("--sample-end", type=int, default=None, dest="sample_end")
    parser.add_argument("--max-seconds", type=float, default=None, dest="max_seconds")
    parser.add_argument("--max-iterations", type=int, default=None, dest="max_iterations")
    parser.add_argument("--target-reduction", type=float, default=None, dest="target_reduction")
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("-o", "--output", required=True, help="Output results.json (results.h5 is written next to it).")
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
    if args.sim_params:
        with h5py.File(args.sim_params, "r") as f:
            deserialize_params(state, f)
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
    static.prepare_static_problem(state, spec.fext_scale)
    indices = list(spec.sample_indices(batch.n_samples))
    results = {
        "spec": spec.to_dict(),
        "percentiles": PERCENTILES,
        "metadata": {
            "fem_elasto_dynamics": args.fem_elasto_dynamics,
            "sim_params": args.sim_params or "(Dump parameters)",
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
