# type: ignore
"""
Offline (headless) GPU simulation, the batch counterpart of ``ui.py``.

Runs the same ``SimulationState`` as the GUI, configured from the command line and an optional
parameter file exported from the GUI ("I/O > Export Parameters"), and writes the trajectory in
the same HDF5 layout as the GUI's "Serialization" feature.

Usage:
    python -m python.tools.sim.simulate --fem-elasto-dynamics dump.h5:fem -o out.h5 \
        [--sim-params params.h5] [--solver AAAVBD] [--cd VertexSdf] [--duration 10] [--fps 60]
"""

import argparse
import math
import time
import h5py
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
    init_h5_serialization,
    serialize_params,
    deserialize_params,
    should_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline GPU Simulation")
    parser.add_argument(
        "--fem-elasto-dynamics",
        type=str,
        required=True,
        help="file.h5:group/path to a serialized FemElastoDynamics object (e.g. an editor Dump file's 'fem' group).",
        dest="fem_elasto_dynamics",
    )
    parser.add_argument(
        "--sim-params",
        type=str,
        default=None,
        help="file.h5 of simulation parameters, as exported by the GUI (optional).",
        dest="sim_params",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        choices=[s.name for s in SolverType],
        default=SolverType.AAAVBD.name,
        help="Solver to use.",
        dest="solver",
    )
    parser.add_argument(
        "--cd",
        type=str,
        choices=[c.name for c in CDType],
        default=CDType.VertexSdf.name,
        help="Contact detection algorithm to use.",
        dest="cd",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=10.0,
        help="Simulated duration in seconds (s).",
        dest="duration",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Frame rate at which positions are written to the output.",
        dest="fps",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file.h5 for the simulated trajectory (overwritten).",
        dest="output",
    )
    return parser.parse_args()


def _write_frame(h5f: h5py.File, state: SimulationState, frame: int):
    """Same layout as ui._serialize_simulation."""
    h5f.create_dataset(f"sim/{frame:08d}/x", data=state.fem.data.x.numpy())


def main():
    args = parse_args()
    wp.init()
    fem_cpu = load_fem_dynamics(args.fem_elasto_dynamics)
    params_cpu = {
        s: load_vbd_params(fem_cpu) for s in [SolverType.VBD, SolverType.AAAVBD]
    }
    params_cpu[SolverType.Chebyshev] = load_chebyshev_params(fem_cpu)
    params_cpu[SolverType.AndersonSolver] = load_anderson_params(fem_cpu)
    params_cpu[SolverType.Newton] = load_newton_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu)

    # Configure the simulation, then reset it, as the GUI does whenever the solver, contact
    # detection algorithm or parameters change.
    state.solver = SolverType[args.solver]
    state.cd_type = CDType[args.cd]
    if args.sim_params:
        with h5py.File(args.sim_params, "r") as f:
            deserialize_params(state, f)
    state.reset()

    n_steps = math.ceil(args.duration / state.dt)
    h5f = init_h5_serialization(state.fem_cpu, args.output, args.fps)
    h5f["params"].attrs["solver"] = state.solver.name
    h5f["params"].attrs["cd"] = state.cd_type.name
    serialize_params(state, h5f["params"])

    frame = 0
    t_last_flush = time.perf_counter()
    t_start = t_last_flush
    pbar = None
    try:
        _write_frame(h5f, state, frame)
        frame += 1
        while state.t < n_steps:
            state.step()
            if should_save(state.t, state.dt, args.fps):
                _write_frame(h5f, state, frame)
                frame += 1
            if time.perf_counter() - t_last_flush >= 1.0:
                h5f.flush()
                t_last_flush = time.perf_counter()
            # The first step compiles/loads Warp modules and captures the CUDA graph, which prints
            # messages. Start the progress bar after it, so those messages don't break the bar up.
            if pbar is None:
                pbar = tqdm(total=n_steps, initial=state.t, desc="Simulating", unit="step")
            else:
                pbar.update(1)
    except KeyboardInterrupt:
        print("Interrupted, closing output file.")
    finally:
        if pbar is not None:
            pbar.close()
        h5f.close()
    elapsed = time.perf_counter() - t_start
    print(
        f"Simulated {state.t} steps ({state.t * state.dt:.4f}s) in {elapsed:.2f}s wall time, "
        f"wrote {frame} frames to {args.output}"
    )


if __name__ == "__main__":
    main()
