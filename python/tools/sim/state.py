# type: ignore
"""
GPU simulation state shared by the online (ui.py) and offline (simulate.py) simulators.

Holds SimulationState (the CPU pbatoolkit objects and their Warp GPU mirrors, solvers and contact
detection), the loaders that build it from HDF5 archives, and the HDF5 (de)serialization of
simulation parameters and trajectories. Nothing here depends on Polyscope, imgui or tkinter.
"""

import enum
import gc
import h5py
import warp as wp
from pbatoolkit import pbat
import numpy as np
from . import gpu
from .gpu.common import profiling


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
    ):
        self.fem_cpu = fem_cpu
        self.params_cpu = params_cpu
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
            s: gpu.vbd.params.Params(p)
            for s, p in params_cpu.items()
            if s in (SolverType.VBD, SolverType.AAAVBD)
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
        newton_params = gpu.newton.solver.Params(self.params_cpu[SolverType.Newton])
        newton_params.construct(self.fem, self.contact)
        self.params[SolverType.Newton] = newton_params
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

    def _make_contact_detector(
        self, contact_pair_storage: gpu.contact.mesh.pairs.ContactPairs
    ) -> gpu.contact.mesh.cd.ContactDetection:
        """Instantiate and register the currently selected contact detection algorithm."""
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
        return detector

    def _advance(self):
        """Run ``substeps`` time steps: setup, solve and step for each."""
        for s in range(self.substeps):
            with profiling.scope("step.setup"):
                self.fem.setup_time_integration_optimization(self.init_strategy)
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
        self.t += 1
        wp.synchronize()
        self.contact_browser.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.contact_overview.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )

    def reset(self):
        self.t = 0
        sdt = self.dt / self.substeps
        self.fem_cpu.set_time_integration_scheme(sdt, self.bdf_scheme)
        self.fem_cpu.set_initial_conditions(self.fem_cpu.X, self.fem_cpu.v * 0.0)
        self.fem = gpu.elasticity.fem.FemElastoDynamics(self.fem_cpu)
        self._attach_kinematics()
        self.params = {
            s: gpu.vbd.params.Params(p)
            for s, p in self.params_cpu.items()
            if s in (SolverType.VBD, SolverType.AAAVBD)
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
        newton_params = gpu.newton.solver.Params(self.params_cpu[SolverType.Newton])
        newton_params.construct(self.fem, self.contact)
        self.params[SolverType.Newton] = newton_params
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
    for stype in [SolverType.VBD, SolverType.AAAVBD]:
        gpu.vbd.params.serialize_vbd_cpu_params(
            state.params_cpu[stype], f.create_group(f"Solver/{stype.name}")
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
    for stype in [SolverType.VBD, SolverType.AAAVBD]:
        key = f"Solver/{stype.name}"
        if key in f:
            gpu.vbd.params.deserialize_vbd_cpu_params(state.params_cpu[stype], f[key])
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
    if "Contact/CDType/OGC" in f:
        state.cd_params[CDType.OGC].deserialize(f["Contact/CDType/OGC"])
    if "Contact/CDType/VertexSdf" in f:
        state.cd_params[CDType.VertexSdf].deserialize(f["Contact/CDType/VertexSdf"])
    if "Contact/ContactStorage" in f:
        state.contact_storage_params.deserialize(f["Contact/ContactStorage"])
    if "Contact/ContactDynamics" in f:
        state.contact_params.deserialize(f["Contact/ContactDynamics"])
