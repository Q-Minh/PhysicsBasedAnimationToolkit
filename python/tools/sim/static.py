# type: ignore
"""
Static equilibrium solves of the GPU simulator, and their convergence logs.

A static solve releases an initial shape into min_x Psi(x) - s fext^T x (``fem.static``, with load
scale s), holding the Dirichlet nodes, and records the error to the rest shape after every solver
iteration. The convergence benchmark (``benchmark.py``) uses MCMC shape samples
(``python/examples/make_benchmark.py``; contract in plans/MASTER.md) as initial shapes and no loads,
so the equilibrium is exactly the rest shape X. The same machinery runs dynamic benchmark problems:
one BDF1 step (:func:`prepare_dynamic_problem`) measured against the problem's reference solution.

Error: e = sqrt((x - x*)^T M (x - x*)) over free nodes (lumped mass M), with x* = X for static
problems and the reference x_ref for dynamic ones.

Timing: VBD-family solves run as one captured CUDA graph. The distinct first iterations are
recorded once, then a block of identical iterations is replayed inside ``wp.capture_while`` until
the error falls below (1 - target_reduction) e0, or a budget runs out. After every iteration, an
``ErrorLog`` entry reads the device clock before and after the error reduction, so the logging
itself is excluded from the solver time. Newton isn't capturable and runs eagerly, one Newton step
per entry, so its time includes host work.

This module has no command line of its own: ``benchmark.py`` is the CLI (see its docstring for
commands, outputs and pitfalls), and the GUI's "Static Equilibrium" panel (``panels/static.py``) runs
static solves interactively. ``python -m python.tools.sim.static`` runs this module's tests.

Python usage:

    state = SimulationState(fem_cpu, params_cpu)   # e.g. as benchmark.make_state builds it
    state.cd_params[CDType.VertexSdf].deactivate = True
    state.reset()
    batch = load_sample_batch("batch.h5")
    check_sample_batch(batch, state.fem)
    prepare_static_problem(state, fext_scale=0.0)  # no loads: the equilibrium is X
    spec = StaticSolveSpec()
    spec.max_seconds = 10.0
    solver = BenchmarkSolver(state, SolverType.AAAVBD, spec)
    log = solver.solve(batch.x[0], batch.reference(0))  # StaticSolveLog
    log.time_to_reduction(0.5), log.relative_error[-1]
"""

from dataclasses import dataclass, field
from enum import IntEnum
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
    """Benchmark problems from one MCMC sample file (one target energy).

    The file's root attribute ``problem`` gives the kind (``"static"`` when absent):

    - **static**: release ``x`` with no loads; the solution is the rest shape ``X``.
    - **dynamic**: one implicit-Euler (BDF1) step of size ``h`` from ``x`` with velocity ``v``, i.e.
      minimize 1/(2 h^2) |y - xtilde|_M^2 + Psi(y) with the Dirichlet DOFs at rest. ``x_ref`` is a
      reference solution (full-order Newton). Its accuracy is ``ref_error_mnorm``, the M-norm (free
      DOFs, as the benchmark's error) of its last Newton correction, in files that have it, and
      otherwise roughly ``ref_newton_correction``, that correction's largest entry (m).

    ``attrs`` holds the file's root attributes as written by the sampler (e.g. ``target_energy``,
    ``sampler``, ``ensemble``, ``rhat``, ``gravity``, ``h``; see the sample-file contract in
    plans/MASTER.md).
    """

    path: str
    X: np.ndarray  # (N, 3) float32 rest positions, Dump node order
    x: np.ndarray  # (S, N, 3) float32 sample positions X + u(q)
    energy: np.ndarray  # (S,) achieved deformation energies (J)
    attrs: dict = field(default_factory=dict)
    min_J: np.ndarray | None = None  # (S,) smallest element volume ratio det(F), when the file has it
    problem: str = "static"  # "static" or "dynamic"
    v: np.ndarray | None = None  # (S, N, 3) initial velocities (dynamic)
    xtilde: np.ndarray | None = None  # (S, N, 3) inertial targets x + h v + h^2 aext (dynamic)
    x_ref: np.ndarray | None = None  # (S, N, 3) reference solutions (dynamic)
    ref_newton_correction: np.ndarray | None = None  # (S,) reference accuracy estimate (m) (dynamic)
    ref_error_mnorm: np.ndarray | None = None  # (S,) reference accuracy in the error's M-norm (dynamic)

    @property
    def h(self) -> float | None:
        """Time step of a dynamic batch (s), from the ``h`` attribute; None for static batches."""
        return float(self.attrs["h"]) if self.problem == "dynamic" else None

    def reference(self, s: int) -> np.ndarray:
        """(N, 3) solution sample ``s``'s error is measured against: ``X`` (static) or ``x_ref[s]``."""
        return self.X if self.problem == "static" else self.x_ref[s]

    @property
    def target_energy(self) -> float:
        """The energy level E* the batch was sampled at (J), from the ``target_energy`` attribute."""
        return float(self.attrs["target_energy"])

    @property
    def n_samples(self) -> int:
        """Number of samples S."""
        return self.x.shape[0]

    def warnings(self) -> list[str]:
        """Reasons the batch may not suit the benchmark's statistics.

        Returns:
            list[str]: One message per problem (split R-hat >= 1.05, nonzero sampling gravity), or an
            empty list.
        """
        out = []
        rhat = self.attrs.get("rhat")
        if rhat is not None and not float(rhat) < 1.05:
            out.append(f"split R-hat {float(rhat):.3f} >= 1.05: the chains haven't mixed, sample more or burn in longer")
        gravity = self.attrs.get("gravity")
        if self.problem == "static" and gravity is not None and np.any(np.asarray(gravity, dtype=np.float64) != 0):
            out.append(f"sampled with gravity {gravity}: the benchmark's static problem has no loads")
        return out


def load_sample_batch(path: str) -> SampleBatch:
    """Load an MCMC sample file.

    Args:
        path (str): The sample file (.h5).

    Returns:
        SampleBatch: Rest positions, sample positions, achieved energies, ``min_J`` when present, and
        the file's attributes.

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
        problem = str(attrs.get("problem", "static"))
        dynamic = {}
        if problem == "dynamic":
            for name in ["v", "xtilde", "x_ref"]:
                dynamic[name] = np.asarray(f[f"samples/{name}"], dtype=np.float32)
            dynamic["ref_newton_correction"] = np.asarray(
                f["samples/ref_newton_correction"], dtype=np.float64
            )
            if "samples/ref_error_mnorm" in f:
                dynamic["ref_error_mnorm"] = np.asarray(f["samples/ref_error_mnorm"], dtype=np.float64)
            if "h" not in attrs:
                raise ValueError(f"{path}: dynamic problems need the time step attribute h")
        elif problem != "static":
            raise ValueError(f"{path}: unknown problem kind {problem!r} (expected static or dynamic)")
    if X.ndim != 2 or X.shape[1] != 3 or x.ndim != 3 or x.shape[1:] != X.shape:
        raise ValueError(f"{path}: expected X (N, 3) and samples/x (S, N, 3), got {X.shape}, {x.shape}")
    for name, a in dynamic.items():
        if name not in ("ref_newton_correction", "ref_error_mnorm") and a.shape != x.shape:
            raise ValueError(f"{path}: samples/{name} has shape {a.shape}, expected {x.shape}")
    return SampleBatch(
        path=path, X=X, x=x, energy=energy, attrs=attrs, min_J=min_J, problem=problem, **dynamic
    )


def check_sample_batch(batch: SampleBatch, fem: FemElastoDynamics, atol: float = 1e-5) -> None:
    """Check that the batch was sampled on this FEM mesh and keeps its Dirichlet nodes at rest
    (in the initial shapes and, for dynamic batches, in the reference solutions).

    Args:
        batch (SampleBatch): The loaded batch.
        fem (FemElastoDynamics): The GPU FEM the samples will be solved on.
        atol (float): Absolute tolerance on positions (m).

    Raises:
        ValueError: if the rest positions differ from ``fem``'s, or a Dirichlet node moves in a sample.
    """
    X = fem.data.X.numpy()
    if X.shape != batch.X.shape or not np.allclose(X, batch.X, atol=atol, rtol=0):
        raise ValueError(f"{batch.path}: rest positions don't match the FEM's X")
    fixed = fem.data.dmask.numpy() != 0
    if fixed.any():
        for name in ["x", "x_ref"]:
            a = getattr(batch, name)
            if a is None:
                continue
            drift = np.abs(a[:, fixed] - X[None, fixed]).max()
            if drift > atol:
                raise ValueError(f"{batch.path}: Dirichlet nodes move by up to {drift} in samples/{name}")


# --- Solve specification ---


class StaticSolveSpec(SerializableMixin):
    """One static benchmark case: initial shapes, loads and stopping criteria.

    Serialized with :meth:`serialize`/:meth:`deserialize` (h5 attributes), which ``benchmark.py``
    exposes as ``--save-spec``/``--spec``. It isn't part of the simulation parameters
    (``state.serialize_params``) and isn't read from ``--sim-params`` configs.
    """

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
        """The sample indices ``[sample_begin, sample_end)`` clipped to a batch of ``n_samples``."""
        end = n_samples if self.sample_end < 0 else min(self.sample_end, n_samples)
        return range(max(0, self.sample_begin), end)

    def to_dict(self) -> dict:
        """The spec's fields as a plain dict, e.g. for JSON."""
        return {
            name: getattr(self, name)
            for name in dir(type(self))
            if isinstance(getattr(type(self), name, None), DocField)
        }


# --- Device-side error log ---


@wp.kernel
def _reset_log_kernel(count: wp.array[wp.int32], keep: wp.array[wp.int32], stop: wp.array[wp.int32]):
    count[0] = 0
    keep[0] = 1
    stop[0] = 0


@wp.kernel
def _stamp_kernel(t_a: wp.array[wp.int64], count: wp.array[wp.int32]):
    c = count[0]
    if c < t_a.shape[0]:
        t_a[c] = globaltimer()


@wp.kernel
def _error_terms_kernel(
    fem: FemElastoDynamicsData, reference: wp.array[wp.vec3f], w: wp.array[wp.float32]
):
    i = wp.tid()
    if is_dirichlet_node(fem.dmask, i):
        w[i] = wp.float32(0)
        return
    d = fem.x[i] - reference[i]
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
    stop: wp.array[wp.int32],
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
    # Stop reason (StopReason), in priority order; 0 keeps going
    e2c = e2_new[0]
    reason = wp.int32(0)
    if not wp.isfinite(e2c):
        reason = wp.int32(4)
    elif e2c <= target_ratio2 * e2[0]:
        reason = wp.int32(1)
    elif c + 1 >= max_entries or c + 1 >= capacity:
        reason = wp.int32(2)
    elif elapsed >= max_ns:
        reason = wp.int32(3)
    stop[0] = reason
    keep[0] = wp.where(reason == 0, wp.int32(1), wp.int32(0))
    if c < capacity:
        t_b[c] = globaltimer()


class StopReason(IntEnum):
    """Why a benchmark solve stopped (``StaticSolveLog.termination`` holds the lowercase name)."""

    RUNNING = 0  # not stopped (the log ended without a stop criterion)
    TARGET = 1  # the error reached the target reduction
    ITERATION_CAP = 2  # max_iterations entries
    TIME_CAP = 3  # max_seconds of device time
    NON_FINITE = 4  # the error is NaN or infinite
    NEWTON_NON_DESCENT = 5  # Newton's direction wasn't a descent direction
    NEWTON_LINE_SEARCH_FAILED = 6  # Newton's line search rejected every step length
    NEWTON_ZERO_GRADIENT = 7  # Newton's gradient is exactly zero (no step possible)


class ErrorLog:
    """Per-iteration log of the squared error to :attr:`reference`, cumulative sweeps and device timestamps.

    Entry 0 is the initial state. ``record`` appends an entry after a solver iteration and updates
    ``keep`` (1 while the error is above the target and the budgets aren't exhausted) and ``stop``
    (the :class:`StopReason`, ``RUNNING`` while ``keep`` is 1). It doesn't
    synchronize, so it can be captured, including in ``wp.capture_while`` bodies.
    """

    def __init__(self, fem: FemElastoDynamics, capacity: int):
        """
        Args:
            fem (FemElastoDynamics): The problem whose ``x`` is logged against :attr:`reference`, which
                starts as its rest positions ``X``.
            capacity (int): Number of entries the device arrays hold (initial state included).
        """
        n_nodes = fem.data.x.shape[0]
        self._fem = fem
        self.reference = wp.clone(fem.data.X)  # (N,) solution the error is measured against
        self.capacity = int(capacity)
        self.e2 = wp.zeros(self.capacity, dtype=wp.float32)
        self.sweeps = wp.zeros(self.capacity, dtype=wp.int32)
        self.t_a = wp.zeros(self.capacity, dtype=wp.int64)
        self.t_b = wp.zeros(self.capacity, dtype=wp.int64)
        self.count = wp.zeros(1, dtype=wp.int32)
        self.keep = wp.zeros(1, dtype=wp.int32)
        self.stop = wp.zeros(1, dtype=wp.int32)
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
        """Set the stopping criteria. Captured graphs bake them in, so recapture after changes.

        Args:
            target_reduction (float): ``keep`` drops to 0 once e <= (1 - target_reduction) e0.
            max_entries (int): ``keep`` drops to 0 at this many entries (clipped to the capacity).
            max_seconds (float): ``keep`` drops to 0 once this much device time has passed since entry
                0 (logging included); ``math.inf`` for no limit.
        """
        self._target_ratio2 = float((1.0 - target_reduction) ** 2)
        self._max_entries = int(min(max_entries, self.capacity))
        self._max_ns = int(min(max_seconds * 1e9, 2**62)) if math.isfinite(max_seconds) else 2**62

    def reset(self) -> None:
        """Empty the log, set ``keep`` to 1 and ``stop`` to ``RUNNING``. Capturable."""
        wp.launch(_reset_log_kernel, dim=1, inputs=[self.count, self.keep, self.stop])

    def record(self, n_sweeps: int) -> None:
        """Append an entry for the current positions and update ``keep``. Capturable.

        Args:
            n_sweeps (int): VBD sweeps done since the previous entry (0 for entry 0 and for Newton).
        """
        fem = self._fem
        wp.launch(_stamp_kernel, dim=1, inputs=[self.t_a, self.count])
        wp.launch(_error_terms_kernel, dim=self._w.shape[0], inputs=[fem.data, self.reference, self._w])
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
                self.stop,
            ],
        )

    def read(self) -> "StaticSolveLog":
        """Copy the log to the host (synchronizes).

        Returns:
            StaticSolveLog: Errors, cumulative solver times and cumulative sweeps per entry, and the
            stop reason.
        """
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
            termination=StopReason(int(self.stop.numpy()[0])).name.lower(),
        )


@dataclass
class StaticSolveLog:
    """Host copy of an :class:`ErrorLog`: entry j is the state after j solver iterations."""

    error: np.ndarray  # (n,) mass-weighted error to the reference (the rest shape for static problems)
    time: np.ndarray  # (n,) cumulative solver wall-clock time (s), logging excluded
    sweeps: np.ndarray  # (n,) cumulative VBD sweeps (0 for Newton)
    termination: str = "running"  # why the solve stopped: a StopReason name, lowercase

    @property
    def relative_error(self) -> np.ndarray:
        """(n,) error relative to the initial one, e / e0 (zeros if e0 is 0)."""
        e0 = self.error[0] if self.error.size else 1.0
        return self.error / e0 if e0 > 0 else np.zeros_like(self.error)

    def time_to_reduction(self, p: float) -> float:
        """Solver time until the error first drops to (1 - p) e0.

        Args:
            p (float): Error reduction, as a fraction (0.95 = 95% of the initial error removed).

        Returns:
            float: Time in seconds at the first entry that reaches it, or NaN if none does.
        """
        hit = np.nonzero(self.relative_error <= 1.0 - p)[0]
        return float(self.time[hit[0]]) if hit.size else math.nan

    def iterations_to_reduction(self, p: float) -> float:
        """Like :meth:`time_to_reduction`, but the number of solver iterations (NaN if never reached)."""
        hit = np.nonzero(self.relative_error <= 1.0 - p)[0]
        return float(hit[0]) if hit.size else math.nan


# --- Solves ---


def prepare_static_problem(state: SimulationState, fext_scale: float) -> np.ndarray:
    """Switch ``state.fem`` to static mode with the external load scaled by ``fext_scale``.

    Args:
        state (SimulationState): Its GPU FEM is modified in place; a ``reset()`` builds a new, dynamic
            one.
        fext_scale (float): Factor on the external load (0 = no loads, so the equilibrium is X).

    Returns:
        np.ndarray: The original external load, (N, 3), so callers can restore it.
    """
    fem = state.fem
    fext0 = fem.data.fext.numpy().copy()
    fem.static = True
    fem.data.fext.assign((fext0 * np.float32(fext_scale)).astype(np.float32))
    return fext0


def prepare_dynamic_problem(state: SimulationState, h: float) -> None:
    """Set up ``state`` for one-step dynamic benchmark problems with time step ``h``.

    One BDF1 step of size ``h`` and no substeps, so ``fem.h2 = h^2`` and the solvers minimize
    1/2 |x - xtilde|_M^2 + h^2 Psi(x), the problem files' objective scaled by h^2. Resets the state,
    which rebuilds its GPU objects in dynamic mode.

    Args:
        state (SimulationState): The simulation (contact detection should already be deactivated).
        h (float): The problems' time step (s).
    """
    state.dt = float(h)
    state.substeps = 1
    state.bdf_scheme = 1
    state.reset()


class BenchmarkSolver:
    """Runs logged benchmark solves of one solver on one ``SimulationState``, one problem at a time.

    A problem is static (``fem.static``: release ``x0``, error to the rest shape) or dynamic (one BDF1
    step from ``x0`` with velocity ``v0`` and inertial target ``xtilde``, error to a reference
    solution). For VBD-family solvers, the whole solve is captured once (on the first :meth:`solve`)
    and replayed for every problem; the state must not be reset in between. The solver's own outer
    loop and tests aren't used: a solve is one subproblem of iterations driven by ``spec``'s budgets
    and stopped by the benchmark's error, so ``gtol``, ``ftol``, ``n_max_iters`` and
    ``n_subproblem_max_iters`` don't apply.
    """

    def __init__(self, state: SimulationState, solver: SolverType, spec: StaticSolveSpec):
        """
        Args:
            state (SimulationState): A contact-free state, prepared for the problems' kind
                (:func:`prepare_static_problem` or :func:`prepare_dynamic_problem`).
            solver (SolverType): The solver to run.
            spec (StaticSolveSpec): Stopping criteria: ``target_reduction``, ``max_iterations``,
                ``max_seconds`` and the replay ``block``.
        """
        self.state = state
        self.solver = solver
        self.spec = spec
        schedule = (
            None
            if solver == SolverType.Newton
            else state.solvers[solver].schedule(state.fem, state.contact, state.params[solver])
        )
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

        Static problems run the problem's setup, which prescribes the Dirichlet nodes (kinematic
        constraints at their frozen time). Dynamic problems skip it: their inertial target comes from
        the problem file (:meth:`solve`), and their Dirichlet nodes are fixed at rest.
        """
        state, params = self.state, self.state.params[self.solver]
        if state.fem.static:
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
            schedule.iteration(0, kp)  # one static subproblem, k = 0
            log.record(schedule.sweeps_per_iteration)

        def block():
            for kp in range(schedule.prefix, schedule.prefix + self._block):
                schedule.iteration(0, kp)
                log.record(schedule.sweeps_per_iteration)

        wp.capture_while(log.keep, while_body=block)

    def _solve_newton(self):
        state, params = self.state, self.state.params[SolverType.Newton]
        newton = gpu.newton.solver
        if state.fem.static:
            state.fem.setup_time_integration_optimization()
        newton.initialize_solve(state.fem, state.contact, state.detector, params)
        saved = (params.data.n_subproblem_max_iters, params.data.gtol2, params.data.abs_eps_lin)
        # One Newton step per entry, and no absolute tolerances: the benchmark's error stops the solve.
        # With h^2-scaled dynamics at small h, |g| can start below gtol, and after a step below CG's
        # atol, so Newton would idle (gtol) or take zero steps (atol). CG keeps its relative tolerance.
        params.data.n_subproblem_max_iters = 1
        params.data.gtol2 = 0.0
        params.data.abs_eps_lin = 0.0
        try:
            log = self.log
            log.reset()
            log.record(0)
            while int(log.keep.numpy()[0]) != 0:
                status = newton.solve_subproblem(state.fem, state.contact, params)
                if status == newton.SubproblemStatus.STEPPED:
                    log.record(0)
                    continue
                # A rejected step (or a zero gradient) leaves x unchanged and is final: stop with that status
                reason = {
                    newton.SubproblemStatus.NON_DESCENT: StopReason.NEWTON_NON_DESCENT,
                    newton.SubproblemStatus.LINE_SEARCH_FAILED: StopReason.NEWTON_LINE_SEARCH_FAILED,
                    newton.SubproblemStatus.CONVERGED: StopReason.NEWTON_ZERO_GRADIENT,
                }[status]
                log.stop.fill_(int(reason))
                break
        finally:
            (params.data.n_subproblem_max_iters, params.data.gtol2, params.data.abs_eps_lin) = saved

    def _load(self, x0, reference, v0, xtilde) -> None:
        fem = self.state.fem
        fem.data.x.assign(np.asarray(x0, dtype=np.float32))
        self.log.reference.assign(fem.data.X if reference is None else np.asarray(reference, dtype=np.float32))
        if fem.static:
            fem.data.v.zero_()
            return
        if v0 is None or xtilde is None:
            raise ValueError("Dynamic problems need v0 and xtilde")
        fem.data.v.assign(np.asarray(v0, dtype=np.float32))
        fem.data.xtilde.assign(np.asarray(xtilde, dtype=np.float32))
        # The step starts from (x0, v0): write the BDF history in place (detection holds xt handles).
        fem.bdf.xt[0, :] = cp.asarray(np.asarray(x0, dtype=np.float32).ravel())[cp.newaxis, :]
        fem.bdf.xt[1, :] = cp.asarray(np.asarray(v0, dtype=np.float32).ravel())[cp.newaxis, :]

    def solve(
        self,
        x0: np.ndarray,
        reference: np.ndarray | None = None,
        v0: np.ndarray | None = None,
        xtilde: np.ndarray | None = None,
    ) -> StaticSolveLog:
        """Solve one problem from ``x0`` and log until a stop criterion.

        Args:
            x0 (np.ndarray): (N, 3) initial positions, e.g. ``batch.x[s]``.
            reference (np.ndarray | None): (N, 3) solution the error is measured against, e.g.
                ``batch.reference(s)``; the rest shape X if None.
            v0 (np.ndarray | None): (N, 3) initial velocities (dynamic problems).
            xtilde (np.ndarray | None): (N, 3) inertial target of the step (dynamic problems).

        Returns:
            StaticSolveLog: The solve's per-iteration log.
        """
        self._load(x0, reference, v0, xtilde)
        if self._schedule is None:
            self._solve_newton()
        else:
            if self._graph is None:
                self._begin()
                with wp.ScopedCapture() as capture:
                    self._begin()
                    self._record_vbd_family()
                self._graph = capture.graph
                self._load(x0, reference, v0, xtilde)
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

    def test_reference(self):
        """Errors are measured against the log's reference, e.g. a dynamic step's x_ref."""
        fem = self._make_fem()
        X = fem.data.X.numpy()
        m = fem.data.m.numpy()
        rng = np.random.default_rng(1)
        ref = (X + 0.1 * rng.standard_normal(X.shape)).astype(np.float32)
        x = (X + 0.1 * rng.standard_normal(X.shape)).astype(np.float32)
        ref[0] = x[0] = X[0]  # the Dirichlet node
        log = ErrorLog(fem, capacity=4)
        log.reference.assign(ref)
        fem.data.x.assign(x)
        log.reset()
        log.record(0)
        e = log.read().error[0]
        expected = np.sqrt(np.sum(m[1:, None] * (x[1:].astype(np.float64) - ref[1:]) ** 2))
        self.assertAlmostEqual(e / expected, 1.0, places=5)


if __name__ == "__main__":
    wp.init()
    unittest.main()
