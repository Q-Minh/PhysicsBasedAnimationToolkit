# type: ignore
"""
Error of simulated trajectories against a reference run of the same scene (e.g. a GPU Newton run converged at
every step), for the paper's argument that a solver's residual doesn't tell how accurate its time steps are.

Errors are mass-weighted RMS distances over the bodies' nodes (element-less collider nodes excluded),
``|x - y|_M / sqrt(sum m) = sqrt(sum_i m_i |x_i - y_i|^2 / sum_i m_i)`` (``trajectory_viewer.mass_rms``), overall
and per body group: bodies with the same node count, i.e. copies of one mesh (``--group COUNT=NAME`` names them).

- ``global``: for each run's frames (``simulate.py`` outputs), the distance to the reference's frame of the same time
  step (frames carry their step count), over time. The reference must store the runs' frame steps, e.g. every step
  (``simulate.py --fps`` above the step rate; ``--fps 100`` at dt = 1e-2 skips step 1, so use 200).
- ``local``: the one-step error, independent of the chaotic divergence of whole trajectories. From reference states
  (the reference's checkpoints, ``simulate.py --checkpoint-every``), each run configuration takes one time step, and
  is compared to the reference's next state, relative to the reference step's displacement:
  ``e = |x(t+h) - x_ref(t+h)|_M / max(|x_ref(t+h) - x_ref(t)|_M, 1e-9 m)``, as the benchmark's e/e0. Each step's
  solve status (residual and penetration ratios to their tolerances, early exit) and GPU time are recorded with it.
  One process per configuration restores the checkpoints in place (``checkpoint.restore``), so its CUDA graph is
  captured once. The checkpoints come from the reference's solver: a VBD-family run's own cross-step buffers
  (AAAVBD's betaG, the Chebyshev and Anderson histories, ...) start fresh at every reference state, as after
  ``reset()``, so each step is independent of the order of the others.
- ``figures``: the paper's PDFs from ``global`` and ``local`` results.

Usage (from the repo root):

    python -m python.tools.sim.reference_error global ref.h5 a.h5 b.h5 ... -o global.h5 [--labels A B ...] \\
        [--group 1104=capsules --group 4150=bowl]
    python -m python.tools.sim.reference_error local --scene scene.h5 --sim-params p.toml [--set KEY=VALUE ...] \\
        --reference ref.h5 [--checkpoints ref.checkpoint.h5] --run AAAVBD=AAAVBD \\
        --run AAAVBD.5x10=AAAVBD,Solver.AAAVBD.n_max_iters=5,Solver.AAAVBD.n_subproblem_max_iters=10 ... -o local.h5
    python -m python.tools.sim.reference_error figures --global global.h5 --local local.h5 -o prefix [--width 7]

``--run LABEL=SOLVER[,KEY=VALUE,...]`` adds a configuration: a solver and its overrides, applied after
``--sim-params`` and ``--set``.

Tests: ``python -m unittest python.tools.sim.reference_error`` (CPU only).
"""

import argparse
import contextlib
import io
import os
import time
import unittest

import h5py
import numpy as np

from .profiling.trajectory import frame_steps
from .trajectory_viewer import body_ids, mass_rms

DISPLACEMENT_FLOOR = 1e-9  # m, floor of the reference step's displacement in the relative one-step error


def frame_index(h5f: h5py.File) -> dict[int, str]:
    """A ``simulate.py`` output's frames by time step count."""
    frames = sorted(k for k in h5f["sim"].keys() if k.isdigit())
    return dict(zip(frame_steps(h5f).tolist(), frames))


class Bodies:
    """The scene's bodies (connected components of the tetrahedra) and their groups (bodies with the same node
    count), from a ``simulate.py`` output's FEM."""

    def __init__(self, h5f: h5py.File, names: dict[int, str] | None = None):
        n_nodes = h5f["FemElastoDynamics/X"].shape[0]
        E = np.asarray(h5f["FemElastoDynamics/E"], dtype=np.int64)  # (elements, 4)
        self.m = np.asarray(h5f["FemElastoDynamics/m"], dtype=np.float64).ravel()
        self.nodes = np.unique(E)  # the bodies' nodes; the others are colliders
        _, self.body = np.unique(body_ids(n_nodes, E)[self.nodes], return_inverse=True)  # per body node, its body
        self.body_counts = np.bincount(self.body)
        sizes = sorted(set(self.body_counts.tolist()))
        names = names or {}
        self.group_sizes = np.array(sizes)
        self.group_names = [names.get(s, f"{s} nodes") for s in sizes]
        self.body_group = np.searchsorted(self.group_sizes, self.body_counts)  # per body, its group

    def rms(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """The mass-weighted RMS distance between positions x and y overall, per group and per body."""
        d2 = np.einsum("ij,ij->i", x[self.nodes] - y[self.nodes], x[self.nodes] - y[self.nodes])
        m = self.m[self.nodes]
        md2_body = np.bincount(self.body, weights=m * d2)
        m_body = np.bincount(self.body, weights=m)
        md2_group = np.bincount(self.body_group, weights=md2_body, minlength=len(self.group_sizes))
        m_group = np.bincount(self.body_group, weights=m_body, minlength=len(self.group_sizes))
        overall = float(np.sqrt(md2_body.sum() / max(m_body.sum(), 1e-30)))
        return overall, np.sqrt(md2_group / np.maximum(m_group, 1e-30)), np.sqrt(md2_body / np.maximum(m_body, 1e-30))


def _read_x(h5f: h5py.File, frame: str) -> np.ndarray:
    return np.asarray(h5f[f"sim/{frame}/x"], dtype=np.float64)


def _check_scene(ref: h5py.File, other: h5py.File, path: str):
    X0, X1 = np.asarray(ref["FemElastoDynamics/X"]), np.asarray(other["FemElastoDynamics/X"])
    if X0.shape != X1.shape or not np.allclose(X0, X1, atol=1e-6 * max(float(np.ptp(X0)), 1e-6)):
        raise ValueError(f"{path} isn't the reference's scene (different rest positions)")


def global_error(reference: str, runs: list[str], labels: list[str], output: str, names: dict[int, str]):
    """Writes each run's per-frame distance to the reference (see the module docstring)."""
    with h5py.File(reference, "r") as ref, h5py.File(output, "w") as out:
        bodies = Bodies(ref, names)
        ref_frames = frame_index(ref)
        dt = float(ref["params/Integration"].attrs["dt"])
        out.attrs["reference"] = reference
        out.attrs["dt"] = dt
        out["groups/names"] = np.array(bodies.group_names, dtype=h5py.string_dtype())
        out["groups/node_counts"] = bodies.group_sizes
        out["bodies/group"] = bodies.body_group
        for path, label in zip(runs, labels):
            with h5py.File(path, "r") as run:
                _check_scene(ref, run, path)
                frames = frame_index(run)
                steps = [t for t in frames if t in ref_frames]
                missing = len(frames) - len(steps)
                if missing:
                    print(f"{label}: {missing} of {len(frames)} frames have no reference frame of their step, skipped")
                total, group, body = [], [], []
                for t in steps:
                    r = bodies.rms(_read_x(run, frames[t]), _read_x(ref, ref_frames[t]))
                    total.append(r[0])
                    group.append(r[1])
                    body.append(r[2])
                g = out.create_group(f"runs/{label}")
                g.attrs["path"] = path
                g.attrs["solver"] = str(run["params"].attrs.get("solver", ""))
                g["step"] = np.array(steps, dtype=np.int64)
                g["rms"] = np.array(total)
                g["rms_group"] = np.array(group).reshape(len(steps), -1)
                g["rms_body"] = np.array(body).reshape(len(steps), -1)
                if steps:
                    print(f"{label}: {len(steps)} frames, RMS distance at the end {total[-1]:.4g} m, max {max(total):.4g} m")


def parse_run(spec: str) -> tuple[str, str, list[str]]:
    """``LABEL=SOLVER[,KEY=VALUE,...]`` -> (label, solver, overrides)."""
    label, sep, rest = spec.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"--run {spec}: expected LABEL=SOLVER[,KEY=VALUE,...]")
    solver, *overrides = rest.split(",")
    return label, solver, overrides


def _snapshot(arrays: dict) -> dict:
    from .checkpoint import _to_numpy

    return {k: _to_numpy(a).copy() for k, a in arrays.items()}


def _assign(arrays: dict, values: dict):
    import cupy as cp

    for k, dst in arrays.items():
        if isinstance(dst, cp.ndarray):
            dst[...] = cp.asarray(values[k])
        else:
            dst.assign(values[k])


def _status(state, step: int) -> dict:
    """One step's solve status: residual and penetration ratios, early exit, iterations."""
    entry = state.status_log.step_entry(step)
    if "vbd" in entry:
        rows = entry["vbd"]
        return dict(
            gradient_ratio=max((r["gradient_ratio"] or np.nan) for r in rows),
            feasibility_ratio=max((r["feasibility_ratio"] or np.nan) for r in rows),
            converged=all(r["reason"] == "CONVERGED" for r in rows),
            outer=sum(r["outer"] for r in rows),
            iterations=sum(r["sweeps"] for r in rows),
        )
    rows = entry.get("newton", [])
    return dict(
        gradient_ratio=max(((r["gradient_ratio"] or np.nan) for r in rows), default=np.nan),
        feasibility_ratio=max(((r["feasibility_ratio"] or np.nan) for r in rows), default=np.nan),
        converged=bool(rows) and all(r["converged"] for r in rows),
        outer=sum(r["outer"] for r in rows),
        iterations=sum(r["newton_steps"] for r in rows),
    )


STATUS_FIELDS = ("gradient_ratio", "feasibility_ratio", "converged", "outer", "iterations")


def local_error(args):
    """One time step from each reference state with each run configuration (see the module docstring)."""
    import warp as wp

    from . import checkpoint, config
    from .state import CDType, SimulationState, SolverType, load_params, load_problem

    ckpt_path = args.checkpoints or os.path.splitext(args.reference)[0] + ".checkpoint.h5"
    available = checkpoint.steps(ckpt_path)
    ref = h5py.File(args.reference, "r")
    ref_frames = frame_index(ref)
    names = dict(parse_group(g) for g in args.group)
    bodies = Bodies(ref, names)
    steps = [t for t in available if t in ref_frames and t + 1 in ref_frames]
    if args.steps:
        wanted = set(args.steps)
        steps = [t for t in steps if t in wanted]
    if args.max_states and len(steps) > args.max_states:
        steps = [steps[i] for i in np.linspace(0, len(steps) - 1, args.max_states).round().astype(int)]
    if not steps:
        raise ValueError(f"No checkpoint of {ckpt_path} has the reference's frames of its step and the next")
    print(f"{len(steps)} reference states, steps {steps[0]} to {steps[-1]}")
    wp.init()
    out = h5py.File(args.output, "a")
    out.attrs["reference"] = args.reference
    out.attrs["checkpoints"] = ckpt_path
    if "groups" not in out:
        out["groups/names"] = np.array(bodies.group_names, dtype=h5py.string_dtype())
        out["groups/node_counts"] = bodies.group_sizes
    try:
        # The reference's own steps: their displacement, the scale of the relative error
        d_ref = np.array([bodies.rms(_read_x(ref, ref_frames[t + 1]), _read_x(ref, ref_frames[t]))[0] for t in steps])
        for spec in args.run:
            label, solver, overrides = parse_run(spec)
            fem_cpu, kinematics, geometry = load_problem(args.fem_elasto_dynamics, args.scene)
            state = SimulationState(fem_cpu, load_params(fem_cpu), kinematics, geometry)
            state.solver = SolverType[solver]
            state.cd_type = CDType[args.cd]
            config.load_sim_params(state, args.sim_params, list(args.overrides) + overrides)
            state.reset()
            fresh_arrays = checkpoint._solver_arrays(state)
            fresh = _snapshot(fresh_arrays)
            events = (wp.Event(enable_timing=True), wp.Event(enable_timing=True))
            rec = {k: [] for k in ("e_abs", "e_rel", "e_group", "step_ms", *STATUS_FIELDS)}
            t_start = time.perf_counter()
            for i, t in enumerate(steps):
                with contextlib.redirect_stdout(io.StringIO()):  # "Resuming with <solver> from a <ref> checkpoint"
                    checkpoint.restore(state, ckpt_path, t)
                _assign(fresh_arrays, fresh)
                state.status_log.clear()
                wp.get_stream().record_event(events[0])
                state.step()
                wp.get_stream().record_event(events[1])
                wp.synchronize()
                x = state.fem.data.x.numpy().astype(np.float64)
                e, e_group, _ = bodies.rms(x, _read_x(ref, ref_frames[t + 1]))
                rec["e_abs"].append(e)
                rec["e_rel"].append(e / max(d_ref[i], DISPLACEMENT_FLOOR))
                rec["e_group"].append(e_group)
                # The first step captures the CUDA graph (capturable solvers): not a step time
                rec["step_ms"].append(wp.get_event_elapsed_time(*events) if i > 0 else np.nan)
                for k, v in _status(state, t).items():
                    rec[k].append(v)
                if args.verbose:
                    print(f"  {label} step {t}: e {rec['e_rel'][-1]:.3g} (abs {e:.3g} m), gradient ratio "
                          f"{rec['gradient_ratio'][-1]:.3g}, feasibility {rec['feasibility_ratio'][-1]:.3g}")
            if label in out.get("runs", {}):
                del out[f"runs/{label}"]
            g = out.create_group(f"runs/{label}")
            g.attrs["solver"] = solver
            g.attrs["overrides"] = ",".join(list(args.overrides) + overrides)
            g["step"] = np.array(steps, dtype=np.int64)
            g["d_ref"] = d_ref
            for k, v in rec.items():
                g[k] = np.array(v)
            out.flush()
            e_rel = np.array(rec["e_rel"])
            print(
                f"{label}: {len(steps)} steps in {time.perf_counter() - t_start:.1f} s, one-step error e median "
                f"{np.median(e_rel):.3g}, p90 {np.percentile(e_rel, 90):.3g}, max {e_rel.max():.3g}; converged "
                f"{int(np.sum(rec['converged']))}/{len(steps)}, step {np.nanmean(rec['step_ms']):.1f} ms"
            )
    finally:
        out.close()
        ref.close()


def parse_group(spec: str) -> tuple[int, str]:
    count, sep, name = spec.partition("=")
    if not sep or not count.isdigit() or not name:
        raise ValueError(f"--group {spec}: expected NODE_COUNT=NAME")
    return int(count), name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Error of simulated trajectories against a reference run")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("global", help="Per-frame distance of whole runs to the reference")
    g.add_argument("reference", help="Reference simulate.py output (holds every frame step of the runs)")
    g.add_argument("runs", nargs="+", help="simulate.py outputs of the same scene")
    g.add_argument("--labels", nargs="+", default=None, help="One per run (default: the file names)")
    g.add_argument("--group", action="append", default=[], metavar="NODE_COUNT=NAME",
                   help="Name the group of bodies with this node count (repeatable)")
    g.add_argument("-o", "--output", required=True, help="Output .h5 (overwritten)")
    loc = sub.add_parser("local", help="One-step error from the reference's checkpointed states")
    problem = loc.add_mutually_exclusive_group(required=True)
    problem.add_argument("--fem-elasto-dynamics", dest="fem_elasto_dynamics", help="file.h5:group of a Dump's FEM")
    problem.add_argument("--scene", help="CPU editor scene file (.h5)")
    loc.add_argument("--sim-params", nargs="+", default=[], dest="sim_params", help="Parameter files, as simulate.py")
    loc.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                     help="Override for every run, after --sim-params (repeatable)")
    loc.add_argument("--cd", default="VertexSdf", help="Contact detection (simulate.py --cd)")
    loc.add_argument("--reference", required=True, help="Reference simulate.py output with every step's frame")
    loc.add_argument("--checkpoints", default=None, help="Reference checkpoints (default: <reference>.checkpoint.h5)")
    loc.add_argument("--run", action="append", required=True, metavar="LABEL=SOLVER[,KEY=VALUE,...]",
                     help="A run configuration (repeatable)")
    loc.add_argument("--steps", type=int, nargs="+", default=None, help="Only these checkpointed steps")
    loc.add_argument("--max-states", type=int, default=0, dest="max_states",
                     help="At most this many reference states, evenly spread (0: all)")
    loc.add_argument("--group", action="append", default=[], metavar="NODE_COUNT=NAME", help="As for global")
    loc.add_argument("-v", "--verbose", action="store_true", help="Print every step")
    loc.add_argument("-o", "--output", required=True,
                     help="Output .h5; runs are added to it, replacing runs of the same label")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "global":
        labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.runs]
        if len(labels) != len(args.runs):
            raise ValueError("--labels needs one label per run")
        global_error(args.reference, args.runs, labels, args.output, dict(parse_group(g) for g in args.group))
    elif args.command == "local":
        local_error(args)


class TestReferenceError(unittest.TestCase):
    def _output(self, path: str, X: np.ndarray, E: np.ndarray, m: np.ndarray, frames: dict[int, np.ndarray]):
        with h5py.File(path, "w") as f:
            f["FemElastoDynamics/X"] = X
            f["FemElastoDynamics/E"] = E
            f["FemElastoDynamics/m"] = m
            f["params/Integration"] = h5py.Empty("f")
            f["params/Integration"].attrs["dt"] = 0.01
            f["params"].attrs["solver"] = "VBD"
            for k, (t, x) in enumerate(sorted(frames.items())):
                f[f"sim/{k:08d}/x"] = x
                f[f"sim/{k:08d}"].attrs["t"] = t

    def test_global_error(self):
        import tempfile

        # Two one-tet bodies of 4 nodes, plus one 8-node body (two tets), plus a collider node
        tet = np.array([[0, 1, 2, 3]])
        E = np.vstack([tet, tet + 4, np.array([[8, 9, 10, 11], [9, 10, 11, 12], [12, 13, 14, 15]])])
        n = 17
        X = np.random.default_rng(0).random((n, 3))
        m = np.ones(n)
        m[8:16] = 2.0
        ref = {t: X + t for t in range(4)}
        run = {t: X + t for t in (0, 2, 3)}
        run[2] = run[2].copy()
        run[2][0:4] += [0.3, 0.0, 0.4]  # first body off by 0.5
        run[3] = run[3].copy()
        run[3][16] += 100.0  # the collider node doesn't count
        with tempfile.TemporaryDirectory() as d:
            self._output(os.path.join(d, "ref.h5"), X, E, m, ref)
            self._output(os.path.join(d, "run.h5"), X, E, m, run)
            main(["global", os.path.join(d, "ref.h5"), os.path.join(d, "run.h5"), "--labels", "A", "--group",
                  "4=small", "-o", os.path.join(d, "out.h5")])
            with h5py.File(os.path.join(d, "out.h5"), "r") as f:
                self.assertEqual([s.decode() for s in f["groups/names"][()]], ["small", "8 nodes"])
                g = f["runs/A"]
                np.testing.assert_array_equal(g["step"][()], [0, 2, 3])
                # Overall at step 2: 4 nodes of mass 1 off by 0.5, out of mass 4 + 4 + 16 = 24
                np.testing.assert_allclose(g["rms"][()], [0.0, np.sqrt(4 * 0.25 / 24), 0.0], atol=1e-12)
                np.testing.assert_allclose(g["rms_group"][1], [np.sqrt(4 * 0.25 / 8), 0.0], atol=1e-12)
                np.testing.assert_allclose(g["rms_body"][1], [0.5, 0.0, 0.0], atol=1e-12)

    def test_parse_run(self):
        self.assertEqual(parse_run("A.5x10=AAAVBD,Solver.AAAVBD.n_max_iters=5"),
                         ("A.5x10", "AAAVBD", ["Solver.AAAVBD.n_max_iters=5"]))
        self.assertEqual(parse_run("N=Newton"), ("N", "Newton", []))
        with self.assertRaises(ValueError):
            parse_run("AAAVBD")


if __name__ == "__main__":
    main()
