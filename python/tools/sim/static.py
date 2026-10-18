# type: ignore
"""
Static equilibrium solves of the GPU simulator, and their convergence logs.

A static solve releases an initial shape into min_x Psi(x) - s fext^T x (``fem.static``, with load
scale s), holding the Dirichlet nodes, and records the error to the rest shape after every solver
iteration. The convergence benchmark (``benchmark_static.py``) uses MCMC shape samples
(``python/examples/rom_mcmc.py``; contract in plans/MASTER.md) as initial shapes and no loads, so the
equilibrium is exactly the rest shape X.

Error: e = sqrt((x - X)^T M (x - X)) over free nodes (lumped mass M).

Timing: VBD-family solves run as one captured CUDA graph. The distinct first iterations are
recorded once, then a block of identical iterations is replayed inside ``wp.capture_while`` until
the error falls below (1 - target_reduction) e0, or a budget runs out. After every iteration, an
``ErrorLog`` entry reads the device clock before and after the error reduction, so the logging
itself is excluded from the solver time. Newton isn't capturable and runs eagerly, one Newton step
per entry, so its time includes host work.
"""

from dataclasses import dataclass, field
import math

import cuda.compute
import cupy as cp
import h5py
import numpy as np
import warp as wp

from .common.fields import DocField, SerializableMixin
from .gpu.common.reduce import Reduce
from .gpu.common.timer import globaltimer
from .gpu.elasticity.fem import FemElastoDynamics, FemElastoDynamicsData, is_dirichlet_node
from . import gpu
from .state import SimulationState, SolverType


# --- MCMC sample batches ---


@dataclass
class SampleBatch:
    """Initial shapes from one MCMC sample file (one target energy)."""

    path: str
    X: np.ndarray  # (N, 3) float32 rest positions, Dump node order
    x: np.ndarray  # (S, N, 3) float32 sample positions X + u(q)
    energy: np.ndarray  # (S,) achieved deformation energies (J)
    attrs: dict = field(default_factory=dict)
    min_J: np.ndarray | None = None  # (S,) smallest element volume ratio det(F), when the file has it

    @property
    def target_energy(self) -> float:
        return float(self.attrs["target_energy"])

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    def warnings(self) -> list[str]:
        """Reasons the batch may not suit the benchmark's statistics (empty if none)."""
        out = []
        rhat = self.attrs.get("rhat")
        if rhat is not None and not float(rhat) < 1.05:
            out.append(f"split R-hat {float(rhat):.3f} >= 1.05: the chains haven't mixed, sample more or burn in longer")
        gravity = self.attrs.get("gravity")
        if gravity is not None and np.any(np.asarray(gravity, dtype=np.float64) != 0):
            out.append(f"sampled with gravity {gravity}: the benchmark's static problem has no loads")
        return out


def load_sample_batch(path: str) -> SampleBatch:
    """Load an MCMC sample file.

    Raises:
        ValueError: if the batch needs importance weights (``importance_weights_required``, e.g. the
            ray sampler), since the benchmark averages uniformly over samples, or if the layout
            doesn't match the contract.
    """
    with h5py.File(path, "r") as f:
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in f.attrs.items()}
        if bool(attrs.get("importance_weights_required", False)):
            raise ValueError(
                f"{path}: samples need importance weights (sampler={attrs.get('sampler')}), "
                "but static benchmarks average uniformly. Use an MCMC (polar/crw) batch."
            )
        X = np.asarray(f["X"], dtype=np.float32)
        x = np.asarray(f["samples/x"], dtype=np.float32)
        energy = np.asarray(f["samples/energy"], dtype=np.float64)
        min_J = np.asarray(f["samples/min_J"], dtype=np.float64) if "samples/min_J" in f else None
    if X.ndim != 2 or X.shape[1] != 3 or x.ndim != 3 or x.shape[1:] != X.shape:
        raise ValueError(f"{path}: expected X (N, 3) and samples/x (S, N, 3), got {X.shape}, {x.shape}")
    return SampleBatch(path=path, X=X, x=x, energy=energy, attrs=attrs, min_J=min_J)


def check_sample_batch(batch: SampleBatch, fem: FemElastoDynamics, atol: float = 1e-5) -> None:
    """Check that the batch was sampled on this FEM mesh and keeps its Dirichlet nodes at rest."""
    X = fem.data.X.numpy()
    if X.shape != batch.X.shape or not np.allclose(X, batch.X, atol=atol, rtol=0):
        raise ValueError(f"{batch.path}: rest positions don't match the FEM's X")
    fixed = fem.data.dmask.numpy() != 0
    if fixed.any():
        drift = np.abs(batch.x[:, fixed] - X[None, fixed]).max()
        if drift > atol:
            raise ValueError(f"{batch.path}: Dirichlet nodes move by up to {drift} in the samples")


# --- Solve specification ---


class StaticSolveSpec(SerializableMixin):
    """One static benchmark case: initial shapes, loads and stopping criteria."""

    sample_file = DocField("", "MCMC sample file (.h5) providing the initial shapes")
    sample_begin = DocField(0, "First sample index")
    sample_end = DocField(-1, "One past the last sample index (-1 = all samples)")
    fext_scale = DocField(0.0, "Scale of the external load (0 = no loads, equilibrium at X)")
    target_reduction = DocField(
        0.95, "Stop once the error drops below (1 - target_reduction) of the initial error"
    )
    max_iterations = DocField(20000, "Maximum solver iterations (sweeps, or Anderson/Newton steps)")
    max_seconds = DocField(60.0, "Maximum wall-clock time per solve (s)")
    block = DocField(16, "Solver iterations per replayed graph block (rounded to Anderson's m)")

    def sample_indices(self, n_samples: int) -> range:
        end = n_samples if self.sample_end < 0 else min(self.sample_end, n_samples)
        return range(max(0, self.sample_begin), end)

    def to_dict(self) -> dict:
        return {
            name: getattr(self, name)
            for name in dir(type(self))
            if isinstance(getattr(type(self), name, None), DocField)
        }


# --- Device-side error log ---


@wp.kernel
def _reset_log_kernel(count: wp.array[wp.int32], keep: wp.array[wp.int32]):
    count[0] = 0
    keep[0] = 1


@wp.kernel
def _stamp_kernel(t_a: wp.array[wp.int64], count: wp.array[wp.int32]):
    c = count[0]
    if c < t_a.shape[0]:
        t_a[c] = globaltimer()


@wp.kernel
def _error_terms_kernel(fem: FemElastoDynamicsData, w: wp.array[wp.float32]):
    i = wp.tid()
    if is_dirichlet_node(fem.dmask, i):
        w[i] = wp.float32(0)
        return
    d = fem.x[i] - fem.X[i]
    w[i] = fem.m[i] * wp.dot(d, d)


@wp.kernel
def _commit_kernel(
    e2_new: wp.array[wp.float32],
    n_sweeps: wp.int32,
    count: wp.array[wp.int32],
    e2: wp.array[wp.float32],
    sweeps: wp.array[wp.int32],
    t_a: wp.array[wp.int64],
    t_b: wp.array[wp.int64],
    target_ratio2: wp.float32,
    max_entries: wp.int32,
    max_ns: wp.int64,
    keep: wp.array[wp.int32],
):
    c = count[0]
    capacity = e2.shape[0]
    if c < capacity:
        e2[c] = e2_new[0]
        s = n_sweeps
        if c > 0:
            s += sweeps[c - 1]
        sweeps[c] = s
    count[0] = c + 1
    elapsed = t_a[wp.min(c, capacity - 1)] - t_a[0]
    k = wp.int32(0)
    if c + 1 < max_entries and c + 1 < capacity and e2_new[0] > target_ratio2 * e2[0] and elapsed < max_ns:
        k = wp.int32(1)
    keep[0] = k
    if c < capacity:
        t_b[c] = globaltimer()


class ErrorLog:
    """Per-iteration log of the squared error, cumulative sweeps and device timestamps.

    Entry 0 is the initial state. ``record`` appends an entry after a solver iteration and updates
    ``keep`` (1 while the error is above the target and the budgets aren't exhausted). It doesn't
    synchronize, so it can be captured, including in ``wp.capture_while`` bodies.
    """

    def __init__(self, fem: FemElastoDynamics, capacity: int):
        n_nodes = fem.data.x.shape[0]
        self._fem = fem
        self.capacity = int(capacity)
        self.e2 = wp.zeros(self.capacity, dtype=wp.float32)
        self.sweeps = wp.zeros(self.capacity, dtype=wp.int32)
        self.t_a = wp.zeros(self.capacity, dtype=wp.int64)
        self.t_b = wp.zeros(self.capacity, dtype=wp.int64)
        self.count = wp.zeros(1, dtype=wp.int32)
        self.keep = wp.zeros(1, dtype=wp.int32)
        self._w = wp.zeros(n_nodes, dtype=wp.float32)
        self._e2_new = wp.zeros(1, dtype=wp.float32)
        self._reduce = Reduce(
            d_in=cp.asarray(self._w),
            d_out=cp.asarray(self._e2_new),
            num_items=n_nodes,
            op=cuda.compute.OpKind.PLUS,
        )
        self.configure(target_reduction=0.95, max_entries=self.capacity, max_seconds=math.inf)

    def configure(self, target_reduction: float, max_entries: int, max_seconds: float) -> None:
        """Set the stopping criteria. Captured graphs bake them in, so recapture after changes."""
        self._target_ratio2 = float((1.0 - target_reduction) ** 2)
        self._max_entries = int(min(max_entries, self.capacity))
        self._max_ns = int(min(max_seconds * 1e9, 2**62)) if math.isfinite(max_seconds) else 2**62

    def reset(self) -> None:
        wp.launch(_reset_log_kernel, dim=1, inputs=[self.count, self.keep])

    def record(self, n_sweeps: int) -> None:
        fem = self._fem
        wp.launch(_stamp_kernel, dim=1, inputs=[self.t_a, self.count])
        wp.launch(_error_terms_kernel, dim=self._w.shape[0], inputs=[fem.data, self._w])
        self._reduce(wp.get_stream())
        wp.launch(
            _commit_kernel,
            dim=1,
            inputs=[
                self._e2_new,
                int(n_sweeps),
                self.count,
                self.e2,
                self.sweeps,
                self.t_a,
                self.t_b,
                self._target_ratio2,
                self._max_entries,
                self._max_ns,
                self.keep,
            ],
        )

    def read(self) -> "StaticSolveLog":
        """Copy the log to the host (synchronizes)."""
        n = min(int(self.count.numpy()[0]), self.capacity)
        e2 = self.e2.numpy()[:n].astype(np.float64)
        t_a = self.t_a.numpy()[:n]
        t_b = self.t_b.numpy()[:n]
        # Solver time of iteration j spans from the end of entry j-1's logging to the start of entry j's.
        dt = np.zeros(n, dtype=np.float64)
        dt[1:] = (t_a[1:] - t_b[:-1]) * 1e-9
        return StaticSolveLog(
            error=np.sqrt(np.maximum(e2, 0.0)),
            time=np.cumsum(dt),
            sweeps=self.sweeps.numpy()[:n].astype(np.int64),
        )


@dataclass
class StaticSolveLog:
    """Host copy of an :class:`ErrorLog`: entry j is the state after j solver iterations."""

    error: np.ndarray  # (n,) mass-weighted error to the rest shape
    time: np.ndarray  # (n,) cumulative solver wall-clock time (s), logging excluded
    sweeps: np.ndarray  # (n,) cumulative VBD sweeps (0 for Newton)

    @property
    def relative_error(self) -> np.ndarray:
        e0 = self.error[0] if self.error.size else 1.0
        return self.error / e0 if e0 > 0 else np.zeros_like(self.error)

    def time_to_reduction(self, p: float) -> float:
        """Solver time until the error first drops to (1 - p) e0, or NaN if it never does."""
        hit = np.nonzero(self.relative_error <= 1.0 - p)[0]
        return float(self.time[hit[0]]) if hit.size else math.nan

    def iterations_to_reduction(self, p: float) -> float:
        hit = np.nonzero(self.relative_error <= 1.0 - p)[0]
        return float(hit[0]) if hit.size else math.nan


# --- Solves ---


@dataclass
class _IterationSchedule:
    """How a VBD-family solver's subproblem iterations replay.

    Iterations kp < prefix are recorded individually, after which iterations repeat with the given
    period, so a block of a multiple of ``period`` iterations can be replayed.
    """

    iteration: object  # Callable[[int], None]
    prefix: int
    period: int
    sweeps_per_iteration: int


def _iteration_schedule(state: SimulationState, solver: SolverType) -> _IterationSchedule:
    fem, contact, params = state.fem, state.contact, state.params[solver]
    if solver == SolverType.VBD:
        return _IterationSchedule(
            lambda kp: gpu.vbd.solver.subproblem_iteration(fem, contact, params, kp), 0, 1, 1
        )
    if solver == SolverType.AAAVBD:
        # One static subproblem (k = 0): only kp == 0 differs.
        return _IterationSchedule(
            lambda kp: gpu.vbd.aaasolver.subproblem_iteration(0, fem, contact, params, kp), 1, 1, 1
        )
    if solver == SolverType.Chebyshev:
        return _IterationSchedule(
            lambda kp: gpu.vbd.chebsolver.subproblem_iteration(fem, contact, params, kp), 2, 1, 1
        )
    if solver == SolverType.AndersonSolver:
        m = int(params.m)
        return _IterationSchedule(
            lambda kp: gpu.vbd.aasolver.subproblem_iteration(fem, contact, params, kp),
            m,
            m,
            int(params.vbd_iters_per_anderson_iter),
        )
    raise ValueError(f"{solver.name} has no replayable iteration schedule")


def prepare_static_problem(state: SimulationState, fext_scale: float) -> np.ndarray:
    """Switch ``state.fem`` to static mode with the external load scaled by ``fext_scale``.

    Returns the original external load, (N, 3), so callers can restore it.
    """
    fem = state.fem
    fext0 = fem.data.fext.numpy().copy()
    fem.static = True
    fem.data.fext.assign((fext0 * np.float32(fext_scale)).astype(np.float32))
    return fext0


class StaticSolver:
    """Runs logged static solves of one solver on one ``SimulationState``, from given initial shapes.

    For VBD-family solvers, the whole solve is captured once (on the first :meth:`solve`) and
    replayed for every initial shape; the state must not be reset in between.
    """

    def __init__(self, state: SimulationState, solver: SolverType, spec: StaticSolveSpec):
        self.state = state
        self.solver = solver
        self.spec = spec
        schedule = None if solver == SolverType.Newton else _iteration_schedule(state, solver)
        self._schedule = schedule
        if schedule is not None:
            self._block = max(1, math.ceil(spec.block / schedule.period)) * schedule.period
            capacity = int(spec.max_iterations) + schedule.prefix + self._block + 1
        else:
            self._block = 1
            capacity = int(spec.max_iterations) + 1
        self.log = ErrorLog(state.fem, capacity)
        self.log.configure(spec.target_reduction, int(spec.max_iterations) + 1, spec.max_seconds)
        self._graph = None

    def _begin(self):
        """Solver setup before the timed iterations (contact-free here, so cheap).

        The static problem's setup prescribes the Dirichlet nodes, evaluating kinematic constraints
        at their frozen time.
        """
        state, params = self.state, self.state.params[self.solver]
        state.fem.setup_time_integration_optimization()
        gpu.vbd.solver.initialize_solve(state.fem, state.contact, state.detector, params)
        gpu.vbd.solver.prepare_subproblem(state.fem, state.contact, state.detector, params)
        if self.solver == SolverType.AAAVBD:
            params.data.betaG.fill_(float(params.data.betaG0[0]))

    def _record_vbd_family(self):
        schedule, log = self._schedule, self.log
        log.reset()
        log.record(0)
        for kp in range(schedule.prefix):
            schedule.iteration(kp)
            log.record(schedule.sweeps_per_iteration)

        def block():
            for kp in range(schedule.prefix, schedule.prefix + self._block):
                schedule.iteration(kp)
                log.record(schedule.sweeps_per_iteration)

        wp.capture_while(log.keep, while_body=block)

    def _solve_newton(self):
        state, params = self.state, self.state.params[SolverType.Newton]
        newton = gpu.newton.solver
        state.fem.setup_time_integration_optimization()
        newton.initialize_solve(state.fem, state.contact, state.detector, params)
        n_inner = params.data.n_subproblem_max_iters
        params.data.n_subproblem_max_iters = 1
        try:
            log = self.log
            log.reset()
            log.record(0)
            while int(log.keep.numpy()[0]) != 0:
                newton.solve_subproblem(state.fem, state.contact, params)
                log.record(0)
        finally:
            params.data.n_subproblem_max_iters = n_inner

    def solve(self, x0: np.ndarray) -> StaticSolveLog:
        """Release ``x0`` ((N, 3) positions) into the static problem and log until a stop criterion."""
        fem = self.state.fem
        fem.data.x.assign(np.asarray(x0, dtype=np.float32))
        fem.data.v.zero_()
        if self._schedule is None:
            self._solve_newton()
        else:
            if self._graph is None:
                self._begin()
                with wp.ScopedCapture() as capture:
                    self._begin()
                    self._record_vbd_family()
                self._graph = capture.graph
                fem.data.x.assign(np.asarray(x0, dtype=np.float32))
            wp.capture_launch(self._graph)
        return self.log.read()


import unittest


@wp.kernel
def _halve_displacement_kernel(fem: FemElastoDynamicsData):
    i = wp.tid()
    if not is_dirichlet_node(fem.dmask, i):
        fem.x[i] = fem.X[i] + wp.float32(0.5) * (fem.x[i] - fem.X[i])


class TestErrorLog(unittest.TestCase):
    def _make_fem(self) -> FemElastoDynamics:
        from pbatoolkit import pbat, pypbat

        V = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        )
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(V.T, C.T)
        fem_cpu.set_mass_matrix(1e3)
        mu, llambda = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, llambda)
        fem_cpu.constrain(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
        return FemElastoDynamics(fem_cpu)

    def test_captured_log(self):
        fem = self._make_fem()
        X = fem.data.X.numpy()
        m = fem.data.m.numpy()
        u = np.random.default_rng(0).standard_normal(X.shape).astype(np.float32)
        u[0] = 0
        log = ErrorLog(fem, capacity=64)
        log.configure(target_reduction=0.95, max_entries=64, max_seconds=math.inf)

        def iteration():
            wp.launch(_halve_displacement_kernel, dim=X.shape[0], inputs=[fem.data])
            log.record(1)

        fem.data.x.assign(X + u)
        with wp.ScopedCapture() as capture:
            log.reset()
            log.record(0)
            wp.capture_while(log.keep, while_body=iteration)
        wp.capture_launch(capture.graph)
        result = log.read()
        e0 = np.sqrt(np.sum(m[:, None] * u.astype(np.float64) ** 2))
        # 0.5^4 > 0.05 >= 0.5^5: the target is reached after exactly 5 halvings
        self.assertEqual(result.error.size, 6)
        self.assertAlmostEqual(result.error[0] / e0, 1.0, places=5)
        self.assertTrue(np.allclose(result.relative_error, 0.5 ** np.arange(6), rtol=1e-5))
        self.assertTrue(np.array_equal(result.sweeps, np.arange(6)))
        self.assertEqual(result.time[0], 0.0)
        self.assertTrue(np.all(np.diff(result.time) >= 0.0))
        self.assertEqual(result.iterations_to_reduction(0.95), 5.0)
        self.assertEqual(result.time_to_reduction(0.95), result.time[5])
        # Replaying the graph from a new shape logs it from scratch
        fem.data.x.assign(X + 2.0 * u)
        wp.capture_launch(capture.graph)
        replay = log.read()
        self.assertEqual(replay.error.size, 6)
        self.assertAlmostEqual(replay.error[0] / e0, 2.0, places=5)


if __name__ == "__main__":
    wp.init()
    unittest.main()
