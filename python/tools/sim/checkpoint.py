# type: ignore
"""Checkpoints of a ``SimulationState``: everything a time step reads from the previous ones, so that a run
resumed from a checkpoint continues as the uninterrupted run would (``simulate.py --checkpoint-every`` and
``--resume``).

A checkpoint file holds one group per checkpointed time step, ``<t:08d>`` (``state.t``, outer time steps of
``dt``), with:

- ``fem/{x,v,xtilde,dmask}`` and ``bdf/{xt,xtilde}``: positions, velocities and the BDF history.
- ``kinematics/{step,anchor_x,anchor_step,enabled}``: the kinematic constraints' device step counter (in
  substeps), the anchors of their open runs and which constraints are enabled.
- ``contact/<vv|ve|vf|ee>/{u,v,s,gamma,lambda_n,lambda_f,friction_limit}``: each contact type's current AL
  state and keys (u, v), from which the next step's first constraint-set update warm starts. The detected
  pairs aren't stored: every step detects them again from ``xt`` before reading them.
- ``contact/{sigma_n,sigma_f}``: the penalties (adaptive ones are recomputed each step or subproblem).
- ``detection/sds``: the vertex SDF's signed distances, if VertexSdf detects.
- ``solver/<SolverType name>/...``: the active solver's buffers that carry across time steps. The
  VBD family's vertex state (``xb``, ``gk``, ``xk``, ``Hnk``, AAAVBD's ``betaG``, ``Qnk``, ``Qfk``), plus
  Chebyshev's and Anderson's histories. GPU Newton has none: its CG starts from zero and it reassembles its
  Hessian every step.

Restoring copies into the state's existing arrays, so registered handles and a captured graph stay valid.
It needs the same problem (mesh, kinematic constraints, contact capacities), ``dt``, ``substeps`` and
``bdf_scheme``. Other parameters may differ: resuming with another solver starts that solver's buffers fresh.
OGC's detector state isn't stored.
"""

import cupy as cp
import h5py
import numpy as np
import warp as wp

FORMAT_VERSION = 1

_CONSTRAINT_FIELDS = ("u", "v", "s", "gamma", "lambda_n", "lambda_f", "friction_limit")
_VBD_FIELDS = ("xb", "gk", "xk", "Hnk", "betaG", "Qnk", "Qfk")
_SOLVER_FIELDS = {
    "Chebyshev": ("omega", "xkm1", "xkm2"),
    "AndersonSolver": ("xkm1", "fkm1", "XkT", "FfT", "Gp", "G", "gammak"),
}


def _state_arrays(state) -> dict:
    """The arrays of ``state`` that a checkpoint stores, by their path in a checkpoint group."""
    fem = state.fem
    arrays = {
        "fem/x": fem.data.x,
        "fem/v": fem.data.v,
        "fem/xtilde": fem.data.xtilde,
        "fem/dmask": fem.data.dmask,
        "bdf/xt": fem.bdf.xt,
        "bdf/xtilde": fem.bdf.xtilde,
        "contact/sigma_n": state.contact.data.sigma_n,
        "contact/sigma_f": state.contact.data.sigma_f,
    }
    k = state.kinematics.data
    arrays.update(
        {
            "kinematics/step": k.step,
            "kinematics/anchor_x": k.anchor_x,
            "kinematics/anchor_step": k.anchor_step,
            "kinematics/enabled": k.enabled,
        }
    )
    for name, cset in zip(
        ("vv", "ve", "vf", "ee"),
        (state.contact.cvv, state.contact.cve, state.contact.cvf, state.contact.cee),
    ):
        for field in _CONSTRAINT_FIELDS:
            arrays[f"contact/{name}/{field}"] = getattr(cset, field).current
    sds = getattr(state.detector, "_sds", None)
    if sds is not None:
        arrays["detection/sds"] = sds
    arrays.update(_solver_arrays(state))
    return arrays


def _solver_arrays(state) -> dict:
    """The active solver's arrays that carry across time steps (none for Newton)."""
    name = state.solver.name
    if name == "Newton":
        return {}
    params = state.params[state.solver]
    arrays = {f"solver/{name}/{field}": getattr(params.data, field) for field in _VBD_FIELDS}
    for field in _SOLVER_FIELDS.get(name, ()):
        arrays[f"solver/{name}/{field}"] = getattr(params, field)
    return arrays


def _to_numpy(a) -> np.ndarray:
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a.numpy()


def write(state, path: str):
    """Append the state at time step ``state.t`` to the checkpoint file ``path`` (created if missing,
    replacing an earlier checkpoint of the same step), and close it, so that a run killed later leaves a
    readable file."""
    wp.synchronize()
    with h5py.File(path, "a") as f:
        name = f"{state.t:08d}"
        if name in f:
            del f[name]
        grp = f.create_group(name)
        grp.attrs["version"] = FORMAT_VERSION
        grp.attrs["t"] = state.t
        grp.attrs["solver"] = state.solver.name
        grp.attrs["cd"] = state.cd_type.name
        grp.attrs["dt"] = state.dt
        grp.attrs["substeps"] = state.substeps
        grp.attrs["bdf_scheme"] = state.bdf_scheme
        for key, a in _state_arrays(state).items():
            grp.create_dataset(key, data=_to_numpy(a))


def parse_spec(spec: str) -> tuple[str, int | None]:
    """Split ``ckpt.h5[:step]`` into the file and the time step (None: the file's latest). Only a trailing
    ``:<digits>`` is a step, so Windows drive letters stay in the path."""
    path, sep, step = spec.rpartition(":")
    if sep and step.isdigit():
        return path, int(step)
    return spec, None


def steps(path: str) -> list[int]:
    """The time steps checkpointed in ``path``, in increasing order."""
    with h5py.File(path, "r") as f:
        return sorted(int(name) for name in f.keys() if name.isdigit())


def restore(state, path: str, step: int | None = None) -> int:
    """Restore ``state`` from the checkpoint of time step ``step`` in ``path`` (default: the latest).

    Call it after configuring and resetting the state for the same problem. The arrays are copied in place.

    Returns:
        int: The restored time step, now ``state.t``.

    Raises:
        ValueError: If the checkpoint doesn't exist, or doesn't match the state's problem or time integration.
    """
    available = steps(path)
    if not available:
        raise ValueError(f"{path} holds no checkpoints")
    if step is None:
        step = available[-1]
    if step not in available:
        raise ValueError(f"{path} has no checkpoint of step {step}; it has steps {available}")
    wp.synchronize()
    with h5py.File(path, "r") as f:
        grp = f[f"{step:08d}"]
        if int(grp.attrs["version"]) != FORMAT_VERSION:
            raise ValueError(f"Checkpoint format {grp.attrs['version']}, expected {FORMAT_VERSION}")
        for key, value in (("dt", state.dt), ("substeps", state.substeps), ("bdf_scheme", state.bdf_scheme)):
            if grp.attrs[key] != value:
                raise ValueError(
                    f"The checkpoint was taken with {key} = {grp.attrs[key]}, this run has {value}: resume "
                    f"with the checkpointed run's time integration"
                )
        solver = str(grp.attrs["solver"])
        if solver != state.solver.name:
            print(
                f"Resuming with {state.solver.name} from a {solver} checkpoint: {state.solver.name}'s own "
                f"buffers start fresh"
            )
        for key, dst in _state_arrays(state).items():
            if key not in grp:
                if key.startswith("solver/") or key == "detection/sds":
                    continue
                raise ValueError(f"The checkpoint has no {key}")
            src = grp[key][()]
            current = _to_numpy(dst)
            if src.shape != current.shape or src.dtype != current.dtype:
                raise ValueError(
                    f"Checkpoint {key} is {src.dtype}{list(src.shape)}, the state's is "
                    f"{current.dtype}{list(current.shape)}: resume the same problem with the same contact "
                    f"capacities"
                )
            if isinstance(dst, cp.ndarray):
                dst[...] = cp.asarray(src)
            else:
                dst.assign(src)
    # The host-side copies of the restored device state
    for k, constraint in enumerate(state.kinematics.constraints):
        constraint.enabled = bool(state.kinematics.data.enabled.numpy()[k])
    state.t = step
    wp.synchronize()
    return step


import os
import tempfile
import unittest


class TestCheckpoint(unittest.TestCase):
    """2N steps straight vs. N steps, a checkpoint, a fresh state resumed from it and N more steps, on two
    stacked cubes in contact: the bottom one fixed, the top one falling 2 mm onto it while an open
    Translate run pushes its top face down."""

    n = 5

    def _state(self, solver):
        from pbatoolkit import pbat, pypbat
        from .gpu.kinematics.constraints import (
            KinematicConstraint,
            KinematicConstraintKind,
            fixed_constraints_from_dmask,
        )
        from .state import SimulationState, load_params

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        X = np.vstack([V, V + [0.21, 0.13, 1.002]]).astype(np.float64)  # offset: no vertex above a vertex
        E = np.vstack([C, C + 8])
        fem = pbat.sim.dynamics.FemElastoDynamics(X.T, E.T)
        fem.set_mass_matrix(1e3)
        mu, lam = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem.set_elastic_energy(mu, lam)
        fem.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem.set_time_integration_scheme(dt=1e-2, s=1)
        fem.constrain(np.repeat([1, 0], 8))
        push = KinematicConstraint(
            "push", KinematicConstraintKind.Translate, 0.02, 1.0, np.arange(12, 16),
            direction=np.array([0.0, 0.0, -1.0]), rate=0.05,
        )
        kinematics = (fixed_constraints_from_dmask(fem.dmask) + [push], 0.0, None)
        state = SimulationState(fem, load_params(fem), kinematics)
        state.solver = solver
        state.reset()
        return state

    def test_resume_matches_uninterrupted(self):
        from .state import SolverType

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.h5")
            for solver in SolverType:
                with self.subTest(solver=solver.name):
                    straight = self._state(solver)
                    for _ in range(2 * self.n):
                        straight.step()
                    first = self._state(solver)
                    for _ in range(self.n):
                        first.step()
                    write(first, path)
                    self.assertGreater(float(first.contact.cvf.lambda_n.current.numpy().sum()), 0.0)
                    resumed = self._state(solver)
                    self.assertEqual(restore(resumed, path), self.n)
                    self.assertEqual(resumed.kinematics.step, self.n)
                    for _ in range(self.n):
                        resumed.step()
                    x, x_resumed = straight.fem.data.x.numpy(), resumed.fem.data.x.numpy()
                    if solver == SolverType.Newton:
                        np.testing.assert_allclose(x_resumed, x, rtol=0, atol=1e-6)
                    else:
                        np.testing.assert_array_equal(x_resumed, x)
                    np.testing.assert_array_equal(resumed.fem.data.v.numpy(), straight.fem.data.v.numpy())

    def test_parse_spec(self):
        self.assertEqual(parse_spec("C:/runs/a.h5:170"), ("C:/runs/a.h5", 170))
        self.assertEqual(parse_spec("C:/runs/a.h5"), ("C:/runs/a.h5", None))
        self.assertEqual(parse_spec("a.h5"), ("a.h5", None))


if __name__ == "__main__":
    wp.init()
    unittest.main()
