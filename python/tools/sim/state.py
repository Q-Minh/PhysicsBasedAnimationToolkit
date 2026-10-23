# type: ignore
"""
GPU simulation state shared by the online (ui.py) and offline (simulate.py) simulators.

Holds SimulationState (the CPU pbatoolkit objects and their Warp GPU mirrors, solvers and contact
detection), the loaders that build it from HDF5 archives, and the HDF5 (de)serialization of
simulation parameters and trajectories. Nothing here depends on Polyscope, imgui or tkinter.
"""

import enum
import gc
import warnings
import h5py
import warp as wp
from pbatoolkit import pbat
import numpy as np
from . import gpu
from . import colliders as colliders_
from .gpu.common import profiling
from .status import StatusLog


def parse_archive_path(spec: str) -> tuple[str, str]:
    """Parse 'file.h5:group/path' into (file_path, group_path).

    Splits on the last ':' only, so that Windows drive letters (e.g. 'C:/data/x.h5:fem') are kept.
    """
    parts = spec.rsplit(":", 1)
    if len(parts) == 2 and len(parts[0]) > 1:
        return parts[0], parts[1]
    else:
        raise ValueError("Invalid archive path format.")


def load_fem_dynamics(spec: str) -> pbat.sim.dynamics.FemElastoDynamics:
    """Deserialize FemElastoDynamics from an h5 archive at file:group."""
    file_path, group_path = parse_archive_path(spec)
    archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
    fem_cpu = pbat.sim.dynamics.FemElastoDynamics()
    fem_cpu.deserialize(archive[group_path] if group_path else "/")
    archive = None
    gc.collect()
    return fem_cpu


def load_kinematic_constraints(
    spec: str, fem: pbat.sim.dynamics.FemElastoDynamics
) -> tuple[
    list[gpu.kinematics.constraints.KinematicConstraint],
    float,
    gpu.kinematics.constraints.KinematicAnchors | None,
]:
    """Load the kinematic constraints of the Dump whose FEM is at file:group.

    Reads the `kinematics` group at the root of the file, written by the CPU editor's Dump. A Dump
    without one gets permanently fixed constraints on the nodes with nonzero dmask.

    Returns:
        The constraints, the initial simulation time t0 in seconds and the anchors of the runs open at t0.
    """
    file_path, _ = parse_archive_path(spec)
    with h5py.File(file_path, "r") as f:
        if "kinematics" in f:
            return gpu.kinematics.constraints.read_kinematic_constraints(f["kinematics"])
    return gpu.kinematics.constraints.fixed_constraints_from_dmask(fem.dmask), 0.0, None


def load_scene(path: str) -> tuple[
    pbat.sim.dynamics.FemElastoDynamics,
    tuple[
        list[gpu.kinematics.constraints.KinematicConstraint],
        float,
        gpu.kinematics.constraints.KinematicAnchors | None,
    ],
    "colliders_.ContactGeometry | None",
]:
    """Build the FEM problem, its kinematic constraints and its contact geometry from a CPU editor scene
    file.

    Reads the tetrahedral elastodynamic bodies, static mesh and plane colliders and transform library that the CPU
    offline simulator reads (``python -m python.tools.vbd.simulate --scene``), and builds the FEM as the
    editor does. The transforms become kinematic constraints through the editor Dump's format.

    Static colliders join the contact surface as pinned, element-less FEM nodes (see colliders.py).

    Returns:
        The FEM problem (dmask holds the nodes constrained at t = 0), its kinematic constraints as returned
        by load_kinematic_constraints, and the contact geometry (colliders.ContactGeometry; None without
        static colliders, i.e. the tetrahedral mesh boundary).
    """
    from ..vbd import scenario

    bodies, colliders, transform_library, names = scenario.read_scene(path)
    fem, _, offsets = scenario.make_fem_dynamics_object(
        bodies, get_external_info=True, static_points=colliders_.collider_points(colliders)
    )
    with h5py.File("kinematics.h5", "w", driver="core", backing_store=False) as f:
        grp = f.create_group("kinematics")
        transform_library.export_kinematic_constraints(grp, names, offsets, 0, 1.0)
        constraints, t0, anchors = gpu.kinematics.constraints.read_kinematic_constraints(grp)
    contact_geometry = None
    if len(colliders) > 0:
        constraint, contact_geometry = colliders_.add_colliders(fem, offsets, colliders)
        constraints.append(constraint)
    dmask = np.zeros(fem.X.shape[1], dtype=np.int64)
    for c in constraints:
        if c.begin <= 0.0 < c.begin + c.duration:
            dmask[c.nodes] = 1
    fem.constrain(dmask)
    return fem, (constraints, t0, anchors), contact_geometry


def load_problem(
    fem_spec: str | None = None, scene: str | None = None
) -> tuple[pbat.sim.dynamics.FemElastoDynamics, tuple, "colliders_.ContactGeometry | None"]:
    """Load the FEM problem, its kinematic constraints and its contact geometry from either an editor Dump's
    FEM group (``file.h5:fem``) or a CPU editor scene file. This is what ui.py and simulate.py do for
    ``--fem-elasto-dynamics`` and ``--scene``.

    Returns:
        ``(fem, kinematics, contact_geometry)``, to pass on as ``SimulationState(fem, params, kinematics,
        contact_geometry)``. ``kinematics`` is ``(constraints, t0, anchors)`` (see
        load_kinematic_constraints). ``contact_geometry`` is None (the tetrahedral mesh boundary) unless a
        scene has static colliders.
    """
    if (fem_spec is None) == (scene is None):
        raise ValueError("Pass exactly one of fem_spec and scene")
    if scene is not None:
        return load_scene(scene)
    fem_cpu = load_fem_dynamics(fem_spec)
    return fem_cpu, load_kinematic_constraints(fem_spec, fem_cpu), None


def load_newton_params(
    fem: pbat.sim.dynamics.FemElastoDynamics,
) -> pbat.sim.algorithm.newton.Params:
    """Construct Newton params from fem mesh with default optimizer settings."""
    return pbat.sim.algorithm.newton.Params().with_optimizer(
        pbat.math.optimization.Newton(
            line_search=pbat.math.optimization.BackTrackingLineSearch()
        )
    )


def load_vbd_params(
    fem: pbat.sim.dynamics.FemElastoDynamics, spec: str | None = None
) -> pbat.sim.algorithm.vbd.Params:
    """Construct VBD params from fem mesh, optionally deserializing from file:group."""
    n_nodes = fem.X.shape[1]
    GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
        fem.E, n_nodes
    )
    GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(fem.E, n_nodes)
    params_cpu = pbat.sim.algorithm.vbd.Params()
    if spec is not None:
        try:
            file_path, group_path = parse_archive_path(spec)
            archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
            params_cpu.deserialize(archive[group_path] if group_path else "/")
            archive = None
            gc.collect()
        except Exception:
            pass  # Use defaults if deserialization fails
    params_cpu.with_vertex_element_adjacency_graph(
        GVGp, GVGe, GVGilocal
    ).with_vertex_colors(GVVp, GVVadj, colors).construct()
    return params_cpu


def load_chebyshev_params(
    fem: pbat.sim.dynamics.FemElastoDynamics, spec: str | None = None
) -> gpu.vbd.params.ChebyshevCpuParams:
    """Construct Chebyshev-accelerated VBD CPU params from fem mesh, optionally deserializing
    the underlying VBD params from file:group."""
    vbd_params_cpu = load_vbd_params(fem, spec=spec)
    return gpu.vbd.params.ChebyshevCpuParams(vbd_params_cpu)


def load_anderson_params(
    fem: pbat.sim.dynamics.FemElastoDynamics, spec: str | None = None
) -> gpu.vbd.params.AndersonCpuParams:
    """Construct Anderson-accelerated VBD CPU params from fem mesh, optionally deserializing
    the underlying VBD params from file:group."""
    vbd_params_cpu = load_vbd_params(fem, spec=spec)
    return gpu.vbd.params.AndersonCpuParams(vbd_params_cpu)


def load_params(
    fem: pbat.sim.dynamics.FemElastoDynamics,
) -> dict["SolverType", object]:
    """CPU parameter objects of every solver, with the GPU simulators' defaults.

    They differ from the C++ defaults (which the CPU editor keeps), as chosen by the maintainer:

    - VBD family: ``n_max_iters = 1`` (one augmented-Lagrangian pass; vertex SDF detection doesn't need
      several detection passes per step) and ``n_subproblem_max_iters = 100`` sweeps.
    - AAAVBD: ``gammadown = 0.99``, ``gammadownS = 0.95`` (the paper's values), ``gammaup = gammaupS =
      0.5``, ``betaG0 = 0.5``, ``rhohat = rhohatS = 0.005``; ``warm_start_beta`` = TimeStep (set in
      :class:`SimulationState`). Its params are a :class:`gpu.vbd.params.AaaVbdCpuParams`, which adds the
      GPU-only ``betaup`` and ``lipschitz_normalization`` (defaults as in C++).
    - Newton: ``n_max_iters = 1`` outer iteration, 10 Newton steps.

    See the sim-run skill's "Tuning" section for scene-specific budgets.
    """
    params = {
        SolverType.VBD: load_vbd_params(fem),
        SolverType.AAAVBD: gpu.vbd.params.AaaVbdCpuParams(load_vbd_params(fem)),
    }
    params[SolverType.Chebyshev] = load_chebyshev_params(fem)
    params[SolverType.AndersonSolver] = load_anderson_params(fem)
    params[SolverType.Newton] = load_newton_params(fem)
    for vbd in [
        params[SolverType.VBD],
        params[SolverType.AAAVBD].vbd_params,
        params[SolverType.Chebyshev].vbd_params,
        params[SolverType.AndersonSolver].vbd_params,
    ]:
        vbd.n_max_iters = 1
        vbd.n_subproblem_max_iters = 100
    aaa = params[SolverType.AAAVBD]
    aaa.betaG0 = 0.5
    aaa.gammadown, aaa.gammadownS = 0.99, 0.95
    aaa.gammaup, aaa.gammaupS = 0.5, 0.5
    aaa.rhohat, aaa.rhohatS = 0.005, 0.005
    params[SolverType.Newton].n_max_iters = 1
    params[SolverType.Newton].newton.n_max_iters = 10
    return params


class SolverType(enum.Enum):
    VBD = 0
    AAAVBD = 1
    Chebyshev = 2
    AndersonSolver = 3
    Newton = 4


class CDType(enum.Enum):
    OGC = 0
    VertexSdf = 1


def should_save(t: int, dt: float, fps: float) -> bool:
    """Return True if simulation step t crosses a new video-frame boundary at the given fps."""
    if t == 0:
        return True
    return int(t * dt * fps) > int((t - 1) * dt * fps)


class SimulationState:
    def __init__(
        self,
        fem_cpu: pbat.sim.dynamics.FemElastoDynamics,
        params_cpu: dict[
            SolverType, pbat.sim.algorithm.vbd.Params | pbat.sim.algorithm.newton.Params
        ],
        kinematics: (
            tuple[
                list[gpu.kinematics.constraints.KinematicConstraint],
                float,
                gpu.kinematics.constraints.KinematicAnchors | None,
            ]
            | None
        ) = None,
        contact_geometry: "colliders_.ContactGeometry | None" = None,
    ):
        """
        Args:
            fem_cpu: The FEM problem.
            params_cpu: CPU solver parameters per solver (see load_params). A plain
                ``pbat.sim.algorithm.vbd.Params`` for AAAVBD is wrapped in a ``gpu.vbd.params.AaaVbdCpuParams``
                with the GPU-only parameters' defaults.
            kinematics: Kinematic constraints (see load_kinematic_constraints), defaulting to fem_cpu's
                dmask as permanently fixed nodes.
            contact_geometry: Contact surface over fem_cpu's nodes, with the depths of plane colliders
                (colliders.ContactGeometry). By default it is the boundary of the tetrahedral mesh, which
                must then be sorted by connected components. load_scene passes one that also holds the
                static mesh and plane colliders.
        """
        self.fem_cpu = fem_cpu
        # Initial velocities of the loaded problem (the Dump's v or the scene's per-body v0), restored on reset
        self.v0 = fem_cpu.v.copy()
        self.params_cpu = params_cpu
        if not isinstance(params_cpu[SolverType.AAAVBD], gpu.vbd.params.AaaVbdCpuParams):
            params_cpu[SolverType.AAAVBD] = gpu.vbd.params.AaaVbdCpuParams(params_cpu[SolverType.AAAVBD])
        # The GPU AAAVBD warm starts its stencil gradient augmentation scales betaG across time steps
        # by default, whereas the C++ default (Subproblem) resets them at every time step. Configs
        # applied after construction (deserialize_params) can still choose another mode.
        self.params_cpu[SolverType.AAAVBD].warm_start_beta = (
            pbat.sim.algorithm.vbd.StencilGradientBetaWarmStartMask.TimeStep
        )
        # Kinematic constraints (see load_kinematic_constraints), defaulting to fem_cpu's dmask as
        # permanently fixed nodes. Every GPU FEM gets its own KinematicConstraintSet built from them.
        self.kinematic_constraints, self.kinematics_t0, self.kinematics_anchors = (
            kinematics
            if kinematics is not None
            else (
                gpu.kinematics.constraints.fixed_constraints_from_dmask(fem_cpu.dmask),
                0.0,
                None,
            )
        )

        # Time integration (modifiable from UI)
        self.dt: float = 1e-2
        self.substeps: int = 1
        self.bdf_scheme: int = 1
        self.init_strategy = (
            pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.TrajectoryWithExternalLoad
        )

        # Solver selection
        self.solver: SolverType = SolverType.AAAVBD

        # Configure time integration
        self.fem_cpu.set_time_integration_scheme(self.dt, self.bdf_scheme)
        self.fem_cpu.set_initial_conditions(fem_cpu.x, fem_cpu.v)

        # Build GPU mirrors
        self.fem = gpu.elasticity.fem.FemElastoDynamics(fem_cpu)
        self._attach_kinematics()
        self.params = {
            SolverType.VBD: gpu.vbd.params.Params(self.params_cpu[SolverType.VBD]),
            SolverType.AAAVBD: gpu.vbd.params.Params(
                self.params_cpu[SolverType.AAAVBD].vbd_params,
                warm_start_beta=self.params_cpu[SolverType.AAAVBD].warm_start_beta,
                betaup=self.params_cpu[SolverType.AAAVBD].betaup,
                lipschitz_normalization=self.params_cpu[SolverType.AAAVBD].lipschitz_normalization,
            ),
        }
        self.params[SolverType.Chebyshev] = gpu.vbd.params.ChebyshevParams(
            self.params_cpu[SolverType.Chebyshev].vbd_params,
            self.params_cpu[SolverType.Chebyshev].cheb_params,
        )
        self.params[SolverType.AndersonSolver] = gpu.vbd.params.AndersonParams(
            self.params_cpu[SolverType.AndersonSolver].vbd_params,
            self.params_cpu[SolverType.AndersonSolver].anderson_params,
            self.params_cpu[SolverType.AndersonSolver].linear_solver,
            self.params_cpu[SolverType.AndersonSolver].vbd_iters_per_anderson_iter,
        )
        self.capture = None

        # Build collision geometry
        self.contact_geometry = contact_geometry
        multimesh_cpu = None if contact_geometry is None else contact_geometry.mesh
        if multimesh_cpu is None:
            multimesh_cpu = pbat.sim.contact.MultiMesh()
            Xordering, Eordering, XCC, ECC, n_components = (
                pbat.graph.sorted_connected_component_ordering(fem_cpu.X, fem_cpu.E)
            )
            is_X_sorted = np.all(Xordering[:-1] <= Xordering[1:])
            is_E_sorted = np.all(Eordering[:-1] <= Eordering[1:])
            if not is_X_sorted or not is_E_sorted:
                raise ValueError(
                    "The FemElastoDynamics is expected to already be sorted by connected components."
                )
            multimesh_cpu.construct_from_tetrahedral_mesh(
                fem_cpu.E, XCC, n_components=n_components
            )
        self.multimesh = gpu.contact.multimesh.MultiMesh(multimesh_cpu)
        self.contact_storage_params = gpu.contact.mesh.pairs.Params()
        contact_pair_storage = gpu.contact.mesh.pairs.ContactPairs(
            self.multimesh, self.contact_storage_params
        )
        self.contact_params = gpu.contact.dynamics.Params()
        self.contact = gpu.contact.dynamics.MeshDynamics(
            self.dt, contact_pair_storage, self.contact_params
        )
        self.newton_linear_solver_params = gpu.newton.solver.LinearSolverParams()
        self.newton_gpu_params = gpu.newton.solver.NewtonGpuParams()
        self.convergence_params = gpu.vbd.convergence.ConvergenceParams()
        newton_params = gpu.newton.solver.Params(
            self.params_cpu[SolverType.Newton], self.newton_linear_solver_params
        )
        newton_params.construct(self.fem, self.contact)
        self.params[SolverType.Newton] = newton_params
        # Per-solve statuses, recorded by step() (see status.py)
        self.status_log = StatusLog()
        self._warned_overflows: set[str] = set()  # contact types whose dropped pairs were reported
        self._construct_convergence_check()
        # Contact detection algorithm selection
        self.cd_type: CDType = CDType.VertexSdf
        self.cd_params = {
            CDType.OGC: gpu.contact.mesh.ogc.OgcParams(),
            CDType.VertexSdf: gpu.contact.mesh.sd.Params(),
        }
        self.detector = self._make_contact_detector(contact_pair_storage)
        self.solvers = {
            SolverType.VBD: gpu.vbd.solver.VbdSolver(),
            SolverType.AndersonSolver: gpu.vbd.aasolver.AndersonSolver(),
            SolverType.AAAVBD: gpu.vbd.aaasolver.AaaVbdSolver(),
            SolverType.Chebyshev: gpu.vbd.chebsolver.ChebyshevSolver(),
            SolverType.Newton: gpu.newton.solver.NewtonSolver(),
        }
        self.contact_browser = gpu.contact.debug.contact.ContactBrowser(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.contact_overview = gpu.contact.debug.contact.ContactOverview(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        # Simulation state
        self.simulate: bool = False
        self.t: int = 0
        self.until_seconds: float = -1.0

    def _attach_kinematics(self):
        """Give the current GPU FEM a fresh KinematicConstraintSet, starting at kinematics_t0."""
        self.kinematics = gpu.kinematics.constraints.KinematicConstraintSet(
            self.kinematic_constraints,
            self.fem.data.x.shape[0],
            self.fem.bdf.h,
            self.kinematics_t0,
            self.kinematics_anchors,
        )
        self.fem.set_kinematic_constraints(self.kinematics)

    def _construct_convergence_check(self):
        """Share one device-side convergence test among the VBD-family and Newton solvers' params."""
        # step() reads the status ring every step, so it needs one row per substep
        self.convergence = gpu.vbd.convergence.ConvergenceCheck(
            self.fem,
            self.contact,
            self.convergence_params,
            status_capacity=self.substeps,
        )
        for s in (
            SolverType.VBD,
            SolverType.AAAVBD,
            SolverType.Chebyshev,
            SolverType.AndersonSolver,
            SolverType.Newton,
        ):
            self.params[s].convergence = self.convergence

    def _make_contact_detector(
        self, contact_pair_storage: gpu.contact.mesh.pairs.ContactPairs
    ) -> gpu.contact.mesh.cd.ContactDetection:
        """Instantiate and register the currently selected contact detection algorithm."""
        # TODO: OGC crashes once bodies touch (seen 2026-10-18 on CubeFallingOnCube, before and after the kinematic
        # constraints), so contact scenes use VertexSdf. Low priority.
        if self.cd_type == CDType.OGC:
            detector = gpu.contact.mesh.ogc.Ogc(self.cd_params[CDType.OGC])
        elif self.cd_type == CDType.VertexSdf:
            detector = gpu.contact.mesh.sd.Sd(self.cd_params[CDType.VertexSdf])
        else:
            raise ValueError(f"Unknown CDType: {self.cd_type}")
        detector.register_handles(
            self.fem.xt,
            None,
            self.fem.data.x,
            self.fem.data.xtilde,
            self.multimesh,
            contact_pair_storage,
            self.fem.data.dmask,
        )
        if self.cd_type == CDType.VertexSdf:
            detector.set_contact_margin(self.contact.params.dmin)
        if self.contact_geometry is not None and self.cd_type == CDType.VertexSdf:
            detector.set_plane_depths(self.contact_geometry.plane_depths)
        if self.cd_type == CDType.VertexSdf and self.multimesh.data.FP.shape[0] - 1 <= 1:
            # One contact component and no self-contact: nothing can collide
            detector.contact_free = True
            if not getattr(self, "_reported_contact_free", False):
                print("Contact detection is off: the scene has a single body and no colliders.")
                self._reported_contact_free = True
        return detector

    def step_init_strategy(self) -> "pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization":
        """The time step's initial-guess strategy: Newton's own (``Solver.Newton.init_strategy``) when Newton
        is the active solver, else the shared ``Integration.init_strategy``."""
        if self.solver == SolverType.Newton:
            return self.newton_gpu_params.init_strategy
        return self.init_strategy

    def _advance(self):
        """Run ``substeps`` time steps: setup, solve and step for each."""
        init_strategy = self.step_init_strategy()
        for s in range(self.substeps):
            with profiling.scope("step.setup"):
                self.fem.setup_time_integration_optimization(init_strategy)
            with profiling.scope("solve"):
                self.solvers[self.solver].solve(
                    self.fem, self.contact, self.detector, self.params[self.solver]
                )
            with profiling.scope("step.advance"):
                self.fem.step()

    def step(self):
        # Profiling times eager launches (see gpu/common/profiling.py), so profiled steps bypass the
        # captured graph.
        if (
            self.solvers[self.solver].supports_graph_capture
            and not profiling.is_enabled()
        ):
            if getattr(self, "capture", None) is None:
                with wp.ScopedCapture() as capture:
                    self._advance()
                self.capture = capture
                # Capturing only records the work: launch it, or this step would be skipped.
                wp.capture_launch(self.capture.graph)  # type: ignore
            else:
                wp.capture_launch(self.capture.graph)  # type: ignore
        else:
            self._advance()
        vbd_family = self.solver != SolverType.Newton
        if vbd_family:
            # Async copy into pinned memory, covered by the synchronization below
            self.convergence.enqueue_status_readback()
        self.contact.contacts.enqueue_requested_peak_readback()
        self.t += 1
        wp.synchronize()
        self._record_status(vbd_family)
        self._warn_dropped_contacts()
        self.contact_browser.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.contact_overview.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )

    def _warn_dropped_contacts(self):
        """Warn, once per contact type, when a detection requested more pairs than the storage holds: the
        excess pairs are dropped (host-only, after the step's sync)."""
        pairs = self.contact.contacts
        for name, requested, capacity in zip(("vv", "ve", "vf", "ee"), pairs.requested_peaks(), pairs.capacity):
            if requested > capacity and name not in self._warned_overflows:
                self._warned_overflows.add(name)
                warnings.warn(
                    f"Time step {self.t - 1}: contact detection found {requested} {name.upper()} pairs, more than "
                    f"the capacity {capacity}; the excess pairs are dropped. Raise "
                    f"Contact.ContactStorage.n_{name}_contact_capacity (now "
                    f"{getattr(self.contact_storage_params, f'n_{name}_contact_capacity'):g}).",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _record_status(self, vbd_family: bool):
        """Append this step's solve statuses to ``status_log`` (host-only, after the step's sync)."""
        params = self.params[self.solver]
        if vbd_family:
            rows, lost = self.convergence.read_status()
            n_max = int(params.data.n_max_iters)
            self.status_log.add_vbd(
                self.t - 1,
                int(self.solver.value),
                rows,
                lost,
                n_max,
                n_max * int(params.data.n_subproblem_max_iters),
            )
        else:
            records, params.solve_records = params.solve_records, []
            self.status_log.add_newton(
                self.t - 1,
                records,
                int(params.data.n_max_iters),
                int(params.data.n_subproblem_max_iters),
                int(params.data.n_lin_max_iters),
            )

    def reset(self):
        self.t = 0
        self.status_log.clear()
        self._warned_overflows.clear()
        sdt = self.dt / self.substeps
        self.fem_cpu.set_time_integration_scheme(sdt, self.bdf_scheme)
        self.fem_cpu.set_initial_conditions(self.fem_cpu.X, self.v0)
        self.fem = gpu.elasticity.fem.FemElastoDynamics(self.fem_cpu)
        self._attach_kinematics()
        self.params = {
            SolverType.VBD: gpu.vbd.params.Params(self.params_cpu[SolverType.VBD]),
            SolverType.AAAVBD: gpu.vbd.params.Params(
                self.params_cpu[SolverType.AAAVBD].vbd_params,
                warm_start_beta=self.params_cpu[SolverType.AAAVBD].warm_start_beta,
                betaup=self.params_cpu[SolverType.AAAVBD].betaup,
                lipschitz_normalization=self.params_cpu[SolverType.AAAVBD].lipschitz_normalization,
            ),
        }
        self.params[SolverType.Chebyshev] = gpu.vbd.params.ChebyshevParams(
            self.params_cpu[SolverType.Chebyshev].vbd_params,
            self.params_cpu[SolverType.Chebyshev].cheb_params,
        )
        self.params[SolverType.AndersonSolver] = gpu.vbd.params.AndersonParams(
            self.params_cpu[SolverType.AndersonSolver].vbd_params,
            self.params_cpu[SolverType.AndersonSolver].anderson_params,
            self.params_cpu[SolverType.AndersonSolver].linear_solver,
            self.params_cpu[SolverType.AndersonSolver].vbd_iters_per_anderson_iter,
        )
        contact_pair_storage = gpu.contact.mesh.pairs.ContactPairs(
            self.multimesh, self.contact_storage_params
        )
        self.contact = gpu.contact.dynamics.MeshDynamics(
            sdt, contact_pair_storage, self.contact_params
        )
        newton_params = gpu.newton.solver.Params(
            self.params_cpu[SolverType.Newton], self.newton_linear_solver_params
        )
        newton_params.construct(self.fem, self.contact)
        self.params[SolverType.Newton] = newton_params
        self._construct_convergence_check()
        self.detector = self._make_contact_detector(contact_pair_storage)
        self.solvers = {
            SolverType.VBD: gpu.vbd.solver.VbdSolver(),
            SolverType.AAAVBD: gpu.vbd.aaasolver.AaaVbdSolver(),
            SolverType.Chebyshev: gpu.vbd.chebsolver.ChebyshevSolver(),
            SolverType.AndersonSolver: gpu.vbd.aasolver.AndersonSolver(),
            SolverType.Newton: gpu.newton.solver.NewtonSolver(),
        }
        self.contact_browser.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.contact_overview.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.capture = None


def init_h5_serialization(
    fem_cpu: pbat.sim.dynamics.FemElastoDynamics, path: str, fps: float | None = None
) -> h5py.File:
    """Create an HDF5 file and write FemElastoDynamics static data."""
    h5f = h5py.File(path, "w")
    grp = h5f.create_group("FemElastoDynamics")
    grp.create_dataset("X", data=fem_cpu.X.T)  # (N, 3) rest positions
    grp.create_dataset("E", data=fem_cpu.E.T)  # (E, 4) element connectivity
    grp.create_dataset("lame_mu", data=fem_cpu.lamegU[0, :])  # (Q,) 1st Lame parameter
    grp.create_dataset(
        "lame_lambda", data=fem_cpu.lamegU[1, :]
    )  # (Q,) 2nd Lame parameter
    grp.create_dataset("m", data=fem_cpu.m)  # (N,) lumped masses
    # TODO: Kinematic constraints change the Dirichlet set during a run, but outputs store only this initial mask, so
    # the editor's trajectory viewer can't show which nodes are held or released at a given frame. Write the mask per
    # frame (or the constraint windows) if that's needed.
    grp.create_dataset("dmask", data=fem_cpu.dmask)  # (N,) Dirichlet mask
    h5f.create_group("sim")
    grp = h5f.create_group("params")
    if fps is not None:
        grp.attrs["fps"] = fps
    return h5f


def serialize_params(state: "SimulationState", f: h5py.File) -> None:
    """Write all simulation parameters to an open HDF5 file."""
    intg = f.create_group("Integration")
    intg.attrs["dt"] = state.dt
    intg.attrs["substeps"] = state.substeps
    intg.attrs["bdf_scheme"] = state.bdf_scheme
    intg.attrs["init_strategy"] = state.init_strategy.value
    gpu.vbd.params.serialize_vbd_cpu_params(
        state.params_cpu[SolverType.VBD], f.create_group("Solver/VBD")
    )
    gpu.vbd.params.serialize_aaavbd_cpu_params(
        state.params_cpu[SolverType.AAAVBD], f.create_group("Solver/AAAVBD")
    )
    gpu.vbd.params.serialize_chebyshev_cpu_params(
        state.params_cpu[SolverType.Chebyshev].vbd_params,
        state.params_cpu[SolverType.Chebyshev].cheb_params,
        f.create_group("Solver/Chebyshev"),
    )
    gpu.vbd.params.serialize_anderson_cpu_params(
        state.params_cpu[SolverType.AndersonSolver], f.create_group("Solver/AndersonSolver")
    )
    gpu.newton.solver.serialize_newton_cpu_params(
        state.params_cpu[SolverType.Newton], f.create_group("Solver/Newton")
    )
    state.newton_gpu_params.serialize(f["Solver/Newton"])
    state.newton_linear_solver_params.serialize(
        f.create_group("Solver/Newton/LinearSolver")
    )
    state.convergence_params.serialize(f.create_group("Solver/Convergence"))
    state.cd_params[CDType.OGC].serialize(f.create_group("Contact/CDType/OGC"))
    state.cd_params[CDType.VertexSdf].serialize(
        f.create_group("Contact/CDType/VertexSdf")
    )
    state.contact_storage_params.serialize(f.create_group("Contact/ContactStorage"))
    state.contact_params.serialize(f.create_group("Contact/ContactDynamics"))


def deserialize_params(state: "SimulationState", f: h5py.File) -> None:
    """Read simulation parameters from an open HDF5 file into state, skipping absent groups."""
    if "Integration" in f:
        intg = f["Integration"]
        if "dt" in intg.attrs:
            state.dt = float(intg.attrs["dt"])
        if "substeps" in intg.attrs:
            state.substeps = int(intg.attrs["substeps"])
        if "bdf_scheme" in intg.attrs:
            state.bdf_scheme = int(intg.attrs["bdf_scheme"])
        if "init_strategy" in intg.attrs:
            state.init_strategy = (
                pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization(
                    int(intg.attrs["init_strategy"])
                )
            )
    if "Solver/VBD" in f:
        gpu.vbd.params.deserialize_vbd_cpu_params(state.params_cpu[SolverType.VBD], f["Solver/VBD"])
    if "Solver/AAAVBD" in f:
        gpu.vbd.params.deserialize_aaavbd_cpu_params(
            state.params_cpu[SolverType.AAAVBD], f["Solver/AAAVBD"]
        )
    if "Solver/Chebyshev" in f:
        gpu.vbd.params.deserialize_chebyshev_cpu_params(
            state.params_cpu[SolverType.Chebyshev].vbd_params,
            state.params_cpu[SolverType.Chebyshev].cheb_params,
            f["Solver/Chebyshev"],
        )
    if "Solver/AndersonSolver" in f:
        gpu.vbd.params.deserialize_anderson_cpu_params(
            state.params_cpu[SolverType.AndersonSolver], f["Solver/AndersonSolver"]
        )
    if "Solver/Newton" in f:
        gpu.newton.solver.deserialize_newton_cpu_params(
            state.params_cpu[SolverType.Newton], f["Solver/Newton"]
        )
        state.newton_gpu_params.deserialize(f["Solver/Newton"])
    if "Solver/Newton/LinearSolver" in f:
        state.newton_linear_solver_params.deserialize(f["Solver/Newton/LinearSolver"])
    if "Solver/Convergence" in f:
        state.convergence_params.deserialize(f["Solver/Convergence"])
    if "Contact/CDType/OGC" in f:
        state.cd_params[CDType.OGC].deserialize(f["Contact/CDType/OGC"])
    if "Contact/CDType/VertexSdf" in f:
        state.cd_params[CDType.VertexSdf].deserialize(f["Contact/CDType/VertexSdf"])
    if "Contact/ContactStorage" in f:
        state.contact_storage_params.deserialize(f["Contact/ContactStorage"])
    if "Contact/ContactDynamics" in f:
        state.contact_params.deserialize(f["Contact/ContactDynamics"])


import unittest


class TestNewtonInitStrategy(unittest.TestCase):
    """Newton's own initial guess (``Solver.Newton.init_strategy``) vs the shared ``Integration.init_strategy``."""

    @staticmethod
    def _make_state() -> SimulationState:
        from pbatoolkit import pypbat

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C = np.array([[0, 1, 2, 3]])
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(V.T, C.T)
        fem_cpu.set_mass_matrix(1e3)
        mu, llambda = pypbat.fem.lame_coefficients(1e6, 0.45)
        fem_cpu.set_elastic_energy(mu, llambda)  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        return SimulationState(fem_cpu, load_params(fem_cpu))

    def test_newton_uses_its_own_strategy(self):
        from unittest import mock

        Init = pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization
        state = self._make_state()
        self.assertEqual(state.newton_gpu_params.init_strategy, Init.Position)  # the default
        state.init_strategy = Init.TrajectoryWithExternalLoad
        for solver in SolverType:
            state.solver = solver
            expected = Init.Position if solver == SolverType.Newton else Init.TrajectoryWithExternalLoad
            # The strategy _advance hands to the time step's setup, with the solve and step stubbed out
            with (
                mock.patch.object(state.fem, "setup_time_integration_optimization") as setup,
                mock.patch.object(state.fem, "step"),
                mock.patch.dict(state.solvers, {solver: mock.Mock()}),
            ):
                state._advance()
            setup.assert_called_once_with(expected)

    def test_serialization_round_trip(self):
        import uuid

        Init = pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization
        a, b = self._make_state(), self._make_state()
        a.newton_gpu_params.init_strategy = Init.FreeTrajectory
        with h5py.File(f"{uuid.uuid4().hex}.h5", "w", driver="core", backing_store=False) as f:
            serialize_params(a, f)
            self.assertEqual(int(f["Solver/Newton"].attrs["init_strategy"]), Init.FreeTrajectory.value)
            deserialize_params(b, f)
            self.assertEqual(b.newton_gpu_params.init_strategy, Init.FreeTrajectory)
            self.assertEqual(b.init_strategy, a.init_strategy)
            # Files that predate the parameter load with the default
            del f["Solver/Newton"].attrs["init_strategy"]
            c = self._make_state()
            deserialize_params(c, f)
            self.assertEqual(c.newton_gpu_params.init_strategy, Init.Position)


class TestDroppedContacts(unittest.TestCase):
    """A warning, when detection requests more pairs than the storage holds."""

    def test_warns_about_dropped_pairs(self):
        from pbatoolkit import pypbat

        # Two unit cubes, the top one 1 mm above the fixed bottom one: their 8 facing vertices are VV pairs, and a
        # zero VV multiplier leaves 1 slot
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        X = np.vstack([V, V + [0.0, 0.0, 1.001]]).astype(np.float64)
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(X.T, np.vstack([C, C + 8]).T)
        fem_cpu.set_mass_matrix(1e3)
        fem_cpu.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))  # pyright: ignore[reportArgumentType]
        fem_cpu.set_external_load(1e3 * np.array([0.0, 0.0, -9.81]))
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        fem_cpu.constrain(np.repeat([1, 0], 8))
        state = SimulationState(fem_cpu, load_params(fem_cpu))
        state.contact_storage_params.n_vv_contact_capacity = 0.0
        state.reset()
        pairs = state.contact.contacts
        self.assertEqual(pairs.capacity[0], 1)
        with self.assertWarnsRegex(RuntimeWarning, "VV pairs, more than the capacity 1;"):
            state.step()
        self.assertGreaterEqual(pairs.requested_peaks()[0], 8)

    def test_small_meshes_get_the_minimum_capacity(self):
        """The default multipliers of a 16-vertex scene (0.05 x 16 VV slots) give the minimum capacity, 64 slots per
        type, and a zero multiplier 1 slot."""
        from pbatoolkit import pypbat

        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        C = np.array([[0, 1, 3, 5], [3, 2, 0, 6], [5, 4, 6, 0], [6, 7, 5, 3], [0, 5, 3, 6]])
        X = np.vstack([V, V + [0.0, 0.0, 1.001]]).astype(np.float64)
        fem_cpu = pbat.sim.dynamics.FemElastoDynamics(X.T, np.vstack([C, C + 8]).T)
        fem_cpu.set_mass_matrix(1e3)
        fem_cpu.set_elastic_energy(*pypbat.fem.lame_coefficients(1e6, 0.45))  # pyright: ignore[reportArgumentType]
        fem_cpu.set_time_integration_scheme(dt=1e-2, s=1)
        state = SimulationState(fem_cpu, load_params(fem_cpu))
        self.assertEqual(state.multimesh.n_verts, 16)
        self.assertEqual(state.contact.contacts.capacity, (64, 64, 64, 64))
        state.contact_storage_params.n_ee_contact_capacity = 0.0
        state.reset()
        self.assertEqual(state.contact.contacts.capacity, (64, 64, 64, 1))


if __name__ == "__main__":
    unittest.main()
