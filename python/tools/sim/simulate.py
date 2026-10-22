# type: ignore
"""
Offline (headless) GPU simulation, the batch counterpart of ``ui.py``.

Runs the same ``SimulationState`` as the GUI, configured from the command line and optional
parameter files (TOML configs, or HDF5 exported from the GUI's "I/O > Export Parameters"; see
``configs/README.md``), and writes the trajectory in the same HDF5 layout as the GUI's
"Serialization" feature.

Usage:
    python -m python.tools.sim.simulate --fem-elasto-dynamics dump.h5:fem -o out.h5 \
        [--sim-params params.toml ...] [--set Solver.VBD.gtol=1e-4] [--solver AAAVBD] [--cd VertexSdf] [--duration 10] [--fps 60] \
        [--profile kernels --profile-out profile.json --profile-warmup 10]

With ``--profile``, the steps after ``--profile-warmup`` run eagerly (no CUDA graph). ``kernels``
attributes the GPU busy time of every Warp launch to the profiling scope that issued it and prints a
per-phase summary. ``nvtx`` emits NVTX ranges for Nsight Systems (see
``python/tools/sim/profiling/nsys.py``).
"""

import argparse
import math
import time
import h5py
import warp as wp
from tqdm import tqdm

from . import config
from .gpu.common import profiling
from .profiling import results as profiling_results
from .state import (
    SimulationState,
    SolverType,
    CDType,
    load_problem,
    load_params,
    init_h5_serialization,
    serialize_params,
    should_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline GPU Simulation")
    problem = parser.add_mutually_exclusive_group(required=True)
    problem.add_argument(
        "--fem-elasto-dynamics",
        type=str,
        help="file.h5:group/path to a serialized FemElastoDynamics object (e.g. an editor Dump file's 'fem' group).",
        dest="fem_elasto_dynamics",
    )
    problem.add_argument(
        "--scene",
        type=str,
        help="CPU editor scene file (.h5), as read by python.tools.vbd.simulate --scene. Static mesh "
        "colliders are ignored.",
        dest="scene",
    )
    parser.add_argument(
        "--sim-params",
        type=str,
        nargs="+",
        default=[],
        help="Simulation parameters: .toml configs or HDF5 files as exported by the GUI, merged left "
        "to right (see python/tools/sim/configs/README.md).",
        dest="sim_params",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one parameter after --sim-params, e.g. --set Solver.VBD.gtol=1e-4 (repeatable).",
        dest="overrides",
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
    parser.add_argument(
        "--profile",
        type=str,
        choices=[b.value for b in profiling.Backend],
        default=profiling.Backend.OFF.value,
        help="Profiling backend. Anything but 'off' disables CUDA graph capture.",
        dest="profile",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Output file.json for the per-step, per-phase GPU timings (kernels backend), the run "
        "metadata that the Nsight Systems importer completes (nvtx backend), or each step's GPU time "
        "in production mode, i.e. graph replay for capturable solvers (off).",
        dest="profile_out",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=10,
        help="Number of initial steps excluded from profiling (module loading, warm-up). They run "
        "unprofiled, with graph capture when the solver supports it.",
        dest="profile_warmup",
    )
    return parser.parse_args()


def _write_frame(h5f: h5py.File, state: SimulationState, frame: int, status_from: int) -> int:
    """Same layout as ui._serialize_simulation, plus ``sim/<frame>/solve_status``: the statuses of the
    time steps since the previous frame, ``[status_from, state.t)`` (see status.StatusLog), and the
    attribute ``sim/<frame>.attrs["t"]``, the time step count at the frame.

    Returns:
        int: The first time step of the next frame's statuses.
    """
    h5f.create_dataset(f"sim/{frame:08d}/x", data=state.fem.data.x.numpy())
    # The time step count at the frame, so frames can be matched to steps (profiling/trajectory.py)
    h5f[f"sim/{frame:08d}"].attrs["t"] = state.t
    state.status_log.write_h5(
        h5f.require_group(f"sim/{frame:08d}/solve_status"), steps=(status_from, state.t)
    )
    return state.t


def main():
    args = parse_args()
    wp.init()
    fem_cpu, kinematics, contact_geometry = load_problem(args.fem_elasto_dynamics, args.scene)
    params_cpu = load_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu, kinematics, contact_geometry)

    # Configure the simulation, then reset it, as the GUI does whenever the solver, contact
    # detection algorithm or parameters change.
    state.solver = SolverType[args.solver]
    state.cd_type = CDType[args.cd]
    if args.sim_params or args.overrides:
        config.load_sim_params(state, args.sim_params, args.overrides)
    state.reset()
    backend = profiling.Backend(args.profile)
    recorder = None
    if backend != profiling.Backend.OFF or args.profile_out:
        # Without a profiling backend, --profile-out records each step's GPU time in production mode
        # (graph replay for capturable solvers), from CUDA events around the step.
        production = backend == profiling.Backend.OFF
        recorder = profiling_results.ProfileRecorder(
            "production" if production else args.profile,
            profiling_results.collect_metadata(
                state, args.fem_elasto_dynamics or args.scene, args.profile_warmup
            ),
        )
        graph = production and state.solvers[state.solver].supports_graph_capture
        recorder.metadata["execution"] = "graph" if graph else "eager"
        step_events = (wp.Event(enable_timing=True), wp.Event(enable_timing=True))

    n_steps = math.ceil(args.duration / state.dt)
    h5f = init_h5_serialization(state.fem_cpu, args.output, args.fps)
    h5f["params"].attrs["solver"] = state.solver.name
    h5f["params"].attrs["cd"] = state.cd_type.name
    serialize_params(state, h5f["params"])

    frame = 0
    t_last_flush = time.perf_counter()
    t_start = t_last_flush
    pbar = None
    measuring = False
    status_from = 0
    try:
        status_from = _write_frame(h5f, state, frame, status_from)
        frame += 1
        while state.t < n_steps:
            profile_step = recorder is not None and state.t >= args.profile_warmup
            if profile_step and not measuring:
                # Warm-up steps run unprofiled (with graph capture, when supported). Profiling, and an
                # attached profiler's capture range (nsys --capture-range=cudaProfilerApi), start here.
                # Production runs open the capture range too, for graph-mode nsys traces.
                profiling.set_backend(backend)
                if backend != profiling.Backend.KERNELS:
                    wp.cuda_profiler_start()
                measuring = True
            if profile_step:
                profiling.begin_step()
                wp.get_stream().record_event(step_events[0])
            state.step()
            if profile_step:
                wp.get_stream().record_event(step_events[1])
                step_profile = profiling.end_step()
                # The step's solve statuses, already on the host (SimulationState.step reads them)
                status = state.status_log.step_entry(state.t - 1)
                if step_profile is not None:
                    recorder.add_step(state.t - 1, step_profile, status)
                elif backend == profiling.Backend.OFF:
                    recorder.add_step_timeline(
                        state.t - 1, wp.get_event_elapsed_time(*step_events), status
                    )
            if should_save(state.t, state.dt, args.fps):
                status_from = _write_frame(h5f, state, frame, status_from)
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
        if measuring and backend != profiling.Backend.KERNELS:
            wp.cuda_profiler_stop()
        if pbar is not None:
            pbar.close()
        # The statuses of the steps after the last written frame (from status_from on) are not stored:
        # no use case needs the final partial interval, and every frame keeps one layout without a
        # trailing table. The summary printed below still covers every step.
        h5f.close()
    elapsed = time.perf_counter() - t_start
    print(
        f"Simulated {state.t} steps ({state.t * state.dt:.4f}s) in {elapsed:.2f}s wall time, "
        f"wrote {frame} frames to {args.output}"
    )
    print("Solve status:")
    for line in state.status_log.format_summary():
        print("  " + line)
    if recorder is not None:
        result = recorder.to_dict()
        if recorder.steps:
            print(profiling_results.format_summary(result))
        if args.profile_out:
            profiling_results.save(result, args.profile_out)
            print(f"Wrote profile to {args.profile_out}")


if __name__ == "__main__":
    main()
