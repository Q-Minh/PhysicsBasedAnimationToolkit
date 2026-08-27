# type: ignore
"""
Lightweight GPU VBD simulation UI using Polyscope + imgui.

Usage:
    python -m python.tools.sim.ui --fem path/to/fem.h5:hdf5/group/to/fem --vbd-params path/to/params.h5:hdf5/group/to/vbdparams
"""

import argparse
import inspect
import enum
import gc
import h5py
import tkinter as tk
from tkinter import filedialog
import warp as wp
import polyscope as ps
import polyscope.imgui as imgui
from pbatoolkit import pbat
import numpy as np
from . import gpu


def try_draw_tooltip(obj, name):
    if imgui.IsItemHovered():
        imgui.BeginTooltip()
        imgui.SetTooltip(getattr(type(obj), name, "").__doc__ or "")
        imgui.EndTooltip()


def draw_params(obj, sub_params: dict | None = None, _depth=0):
    for name, value in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        if (
            isinstance(getattr(type(obj), name, None), property)
            and getattr(type(obj), name).fset is None
        ):
            continue
        if value is None or callable(value):
            continue
        if isinstance(value, float):
            _, new_value = imgui.InputFloat(name, value, format="%.6f")
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, bool):
            _, new_value = imgui.Checkbox(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, int):
            _, new_value = imgui.InputInt(name, value)
            try_draw_tooltip(obj, name)
            setattr(obj, name, new_value)
        elif isinstance(value, enum.Enum):
            enum_values = list(type(value))
            selected_idx = enum_values.index(value)
            _, selected_idx = imgui.Combo(
                name, selected_idx, [ev.name for ev in enum_values]
            )
            try_draw_tooltip(obj, name)
            setattr(obj, name, enum_values[selected_idx])
        elif sub_params is None and _depth < 4 and not isinstance(value, np.ndarray):
            if imgui.TreeNode(name):
                imgui.PushID(name)
                draw_params(value, None, _depth + 1)
                imgui.PopID()
                imgui.TreePop()
    if sub_params:
        for sub_name, nested_sub_params in sub_params.items():
            sub_value = getattr(obj, sub_name, None)
            if sub_value is None:
                continue
            if imgui.TreeNode(sub_name):
                imgui.PushID(sub_name)
                draw_params(sub_value, nested_sub_params or None, _depth + 1)
                imgui.PopID()
                imgui.TreePop()


def parse_archive_path(spec: str) -> tuple[str, str]:
    """Parse 'file.h5:group/path' into (file_path, group_path)."""
    parts = spec.split(":")
    if len(parts) == 2:
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


class SolverType(enum.Enum):
    VBD = 0
    AAAVBD = 1
    Chebyshev = 2
    # TODO: Add AndersonSolver to the enum when implemented
    # ...
    Newton = 3


class CDType(enum.Enum):
    OGC = 0
    VertexSdf = 1


SOLVER_SUB_PARAMS: dict[SolverType, dict] = {
    SolverType.VBD: {},
    SolverType.AAAVBD: {},
    SolverType.Chebyshev: {"vbd_params": None},
    # TODO: Add sub-params for AndersonSolver when implemented
    # ...
    SolverType.Newton: {"newton": {"line_search": {}}},
}


def _should_save(t: int, dt: float, fps: float) -> bool:
    """Return True if simulation step t crosses a new video-frame boundary at the given fps."""
    if t == 0:
        return True
    return int(t * dt * fps) > int((t - 1) * dt * fps)


class UIState:
    def __init__(self):
        self.request_reset: bool = False
        self.item_width: int = 250
        self.screenshot_after_step: bool = False
        self.screenshot_fps: float = 60.0
        self.screenshot_frame: int = 0
        self.debug_tab_active: bool = False
        self.serialization_path: str | None = None
        self.serialization_h5: h5py.File | None = None
        self.serialization_time_since_flush: float = 0.0


class SimulationState:
    def __init__(
        self,
        fem_cpu: pbat.sim.dynamics.FemElastoDynamics,
        params_cpu: dict[
            SolverType, pbat.sim.algorithm.vbd.Params | pbat.sim.algorithm.newton.Params
        ],
    ):
        self.fem_cpu = fem_cpu
        self.params_cpu = params_cpu

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
        self.params = {
            s: gpu.vbd.params.Params(p)
            for s, p in params_cpu.items()
            if s in (SolverType.VBD, SolverType.AAAVBD)
        }
        self.params[SolverType.Chebyshev] = gpu.vbd.params.ChebyshevParams(
            self.params_cpu[SolverType.Chebyshev].vbd_params,
            self.params_cpu[SolverType.Chebyshev].cheb_params,
        )
        # TODO: Add params to self.params for AndersonSolver
        # ...
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
            # TODO: Add AndersonSolver here when implemented
            # ...
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
        )
        return detector

    def step(self):
        if self.solvers[self.solver].supports_graph_capture:
            if getattr(self, "capture", None) is None:
                with wp.ScopedCapture() as capture:
                    for s in range(self.substeps):
                        self.fem.setup_time_integration_optimization(self.init_strategy)
                        self.solvers[self.solver].solve(
                            self.fem,
                            self.contact,
                            self.detector,
                            self.params[self.solver],
                        )
                        self.fem.step()
                self.capture = capture
            else:
                # with wp.ScopedTimer(
                #     "PBAT Step",
                #     detailed=True,
                #     use_nvtx=True,
                #     synchronize=True,
                #     cuda_filter=wp.TIMING_ALL,
                # ):
                wp.capture_launch(self.capture.graph)  # type: ignore
        else:
            for s in range(self.substeps):
                self.fem.setup_time_integration_optimization(self.init_strategy)
                self.solvers[self.solver].solve(
                    self.fem, self.contact, self.detector, self.params[self.solver]
                )
                self.fem.step()
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
        self.params = {
            s: gpu.vbd.params.Params(p)
            for s, p in self.params_cpu.items()
            if s in (SolverType.VBD, SolverType.AAAVBD)
        }
        self.params[SolverType.Chebyshev] = gpu.vbd.params.ChebyshevParams(
            self.params_cpu[SolverType.Chebyshev].vbd_params,
            self.params_cpu[SolverType.Chebyshev].cheb_params,
        )
        # TODO: Add params to self.params for AndersonSolver
        # ...
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
            SolverType.Newton: gpu.newton.solver.NewtonSolver(),
        }
        self.contact_browser.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.contact_overview.update(
            self.fem.data.x, self.multimesh, self.contact.contacts
        )
        self.capture = None


def _init_h5_serialization(
    fem_cpu: pbat.sim.dynamics.FemElastoDynamics, path: str, ui_state: UIState = None
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
    if ui_state is not None:
        grp.attrs["fps"] = ui_state.screenshot_fps
    return h5f


def _serialize_params(state: "SimulationState", f: h5py.File) -> None:
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
    gpu.newton.solver.serialize_newton_cpu_params(
        state.params_cpu[SolverType.Newton], f.create_group("Solver/Newton")
    )
    state.cd_params[CDType.OGC].serialize(f.create_group("Contact/CDType/OGC"))
    state.cd_params[CDType.VertexSdf].serialize(
        f.create_group("Contact/CDType/VertexSdf")
    )
    state.contact_storage_params.serialize(f.create_group("Contact/ContactStorage"))
    state.contact_params.serialize(f.create_group("Contact/ContactDynamics"))


def _deserialize_params(state: "SimulationState", f: h5py.File) -> None:
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


def _serialize_simulation(state: SimulationState, ui_state: UIState):
    if ui_state.serialization_h5 is not None:
        x = state.fem.data.x.numpy()
        ui_state.serialization_h5.create_dataset(
            f"sim/{ui_state.screenshot_frame:08d}/x", data=x
        )
        ui_state.serialization_time_since_flush += imgui.GetIO().DeltaTime
        if ui_state.serialization_time_since_flush >= 1.0:
            ui_state.serialization_h5.flush()
            ui_state.serialization_time_since_flush = 0.0


def _import_parameters_from_file(
    path: str, state: SimulationState, ui_state: UIState, mesh_name: str
):
    with h5py.File(path, "r") as f:
        _deserialize_params(state, f)
    state.reset()
    _update_mesh(state, mesh_name)
    ui_state.request_reset = True


def make_callback(
    state: SimulationState, ui_state: UIState, mesh_name: str = "FEM Mesh"
):
    def callback():
        ui_state.request_reset = False

        imgui.PushItemWidth(ui_state.item_width)

        if imgui.BeginTabBar("MainTabs"):
            if imgui.BeginTabItem("Simulation", True)[0]:
                imgui.Text(f"Step: {state.t}  Time: {state.t * state.dt:.4f}s")
                imgui.Separator()

                # --- Integration controls ---
                if imgui.TreeNode("Integration"):
                    _, state.dt = imgui.InputFloat("dt", state.dt, format="%.5f")
                    _, state.substeps = imgui.InputInt("Substeps", state.substeps)
                    _, state.bdf_scheme = imgui.InputInt("BDF order", state.bdf_scheme)
                    state.bdf_scheme = max(1, min(6, state.bdf_scheme))
                    init_strategies = list(
                        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization
                    )
                    idx = init_strategies.index(state.init_strategy)
                    _, idx = imgui.Combo(
                        "Init Strategy", idx, [s.name for s in init_strategies]
                    )
                    state.init_strategy = init_strategies[idx]
                    imgui.TreePop()

                # --- Solver selection ---
                if imgui.TreeNode("Solver"):
                    solvers = list(SolverType)
                    solver_idx = solvers.index(state.solver)
                    _, solver_idx = imgui.Combo(
                        "Solver", solver_idx, [s.name for s in solvers]
                    )
                    new_solver = solvers[solver_idx]
                    if new_solver != state.solver:
                        state.solver = new_solver
                        state.reset()
                        ui_state.request_reset = True
                    if imgui.TreeNode("Params"):
                        draw_params(
                            state.params_cpu[state.solver],
                            SOLVER_SUB_PARAMS.get(state.solver),
                        )
                        imgui.TreePop()
                    imgui.TreePop()

                # --- Contact parameters ---
                if imgui.TreeNode("Contact"):
                    if imgui.TreeNode("Statistics"):
                        nvv, nve, nvf, nee = state.contact.contacts.num_contacts
                        imgui.Text(f"# Vertex-Vertex Contacts: {nvv}")
                        imgui.Text(f"# Vertex-Edge Contacts: {nve}")
                        imgui.Text(f"# Vertex-Face Contacts: {nvf}")
                        imgui.Text(f"# Edge-Edge Contacts: {nee}")
                        imgui.TreePop()
                    if imgui.TreeNode("Storage"):
                        draw_params(state.contact_storage_params)
                        imgui.TreePop()
                    if imgui.TreeNode("Detection"):
                        cd_types = list(CDType)
                        cd_idx = cd_types.index(state.cd_type)
                        _, cd_idx = imgui.Combo(
                            "Algorithm", cd_idx, [c.name for c in cd_types]
                        )
                        new_cd = cd_types[cd_idx]
                        if new_cd != state.cd_type:
                            state.cd_type = new_cd
                            state.reset()
                            ui_state.request_reset = True
                        if imgui.TreeNode("Params"):
                            draw_params(state.cd_params[state.cd_type])
                            imgui.TreePop()
                        imgui.TreePop()
                    if imgui.TreeNode("Dynamics"):
                        draw_params(state.contact_params)
                        imgui.TreePop()
                    imgui.TreePop()

                # --- Parameters I/O ---
                if imgui.TreeNode("I/O"):
                    if imgui.Button("Export Parameters"):
                        root = tk.Tk()
                        root.withdraw()
                        path = filedialog.asksaveasfilename(
                            title="Export simulation parameters",
                            defaultextension=".h5",
                            filetypes=[
                                ("HDF5 files", "*.h5 *.hdf5"),
                                ("All files", "*.*"),
                            ],
                        )
                        root.destroy()
                        if path:
                            with h5py.File(path, "w") as f:
                                _serialize_params(state, f)
                    if imgui.Button("Import Parameters"):
                        root = tk.Tk()
                        root.withdraw()
                        path = filedialog.askopenfilename(
                            title="Import simulation parameters",
                            defaultextension=".h5",
                            filetypes=[
                                ("HDF5 files", "*.h5 *.hdf5"),
                                ("All files", "*.*"),
                            ],
                        )
                        root.destroy()
                        if path:
                            _import_parameters_from_file(
                                path, state, ui_state, mesh_name
                            )
                    imgui.TreePop()

                # --- Serialization ---
                if imgui.TreeNode("Serialization"):
                    if ui_state.serialization_h5 is None:
                        if imgui.Button("Start##Serialization"):
                            root = tk.Tk()
                            root.withdraw()
                            path = filedialog.asksaveasfilename(
                                title="Save simulation serialization",
                                defaultextension=".h5",
                                filetypes=[
                                    ("HDF5 files", "*.h5 *.hdf5"),
                                    ("All files", "*.*"),
                                ],
                            )
                            root.destroy()
                            if path:
                                ui_state.serialization_h5 = _init_h5_serialization(
                                    state.fem_cpu, path, ui_state
                                )
                                ui_state.serialization_path = path
                    else:
                        imgui.TextWrapped(
                            f"Serializing to {ui_state.serialization_path}"
                        )
                        if imgui.Button("Stop##Serialization"):
                            ui_state.serialization_h5.close()
                            ui_state.serialization_h5 = None
                            ui_state.serialization_path = None
                    imgui.TreePop()

                # --- Simulation controls ---
                imgui.Separator()

                # --- Screenshot ---

                # if ui_state.screenshot_after_step:

                changed, ui_state.screenshot_fps = imgui.InputFloat(
                    "fps##screenshot", ui_state.screenshot_fps, format="%.1f"
                )
                if changed:
                    ui_state.screenshot_fps = max(0.1, ui_state.screenshot_fps)
                    if ui_state.serialization_h5 is not None:
                        ui_state.serialization_h5.get("params").attrs[
                            "fps"
                        ] = ui_state.screenshot_fps

                _, ui_state.screenshot_after_step = imgui.Checkbox(
                    "Screenshot", ui_state.screenshot_after_step
                )

                _, state.simulate = imgui.Checkbox("Simulate", state.simulate)
                imgui.SameLine()
                _, state.until_seconds = imgui.InputFloat(
                    "Until (s)", state.until_seconds, format="%.3f"
                )
                if (
                    state.until_seconds >= 0.0
                    and state.t * state.dt >= state.until_seconds
                ):
                    state.simulate = False

                if imgui.Button("Reset") or ui_state.request_reset:
                    if ui_state.serialization_h5 is not None:
                        ui_state.serialization_h5.close()
                        ui_state.serialization_h5 = None
                        ui_state.serialization_path = None
                    state.reset()
                    ui_state.screenshot_frame = 0
                    _update_mesh(state, mesh_name)

                # --- Continuous simulation ---
                request_step = state.simulate or imgui.Button("Step")
                if request_step:
                    if (
                        ui_state.screenshot_after_step
                        or ui_state.serialization_h5 is not None
                    ):
                        if state.t == 0:
                            if ui_state.screenshot_after_step:
                                ps.screenshot(
                                    "{:08d}.png".format(ui_state.screenshot_frame)
                                )
                            if ui_state.serialization_h5 is not None:
                                _serialize_simulation(state, ui_state)
                            ui_state.screenshot_frame += 1

                    state.step()
                    _update_mesh(state, mesh_name)

                    if (
                        ui_state.screenshot_after_step
                        or ui_state.serialization_h5 is not None
                    ):
                        if _should_save(state.t, state.dt, ui_state.screenshot_fps):
                            if ui_state.screenshot_after_step:
                                ps.screenshot(
                                    "{:08d}.png".format(ui_state.screenshot_frame)
                                )
                            if ui_state.serialization_h5 is not None:
                                _serialize_simulation(state, ui_state)
                            ui_state.screenshot_frame += 1

                imgui.EndTabItem()

            if imgui.BeginTabItem("Debug", True)[0]:
                ui_state.debug_tab_active = True
                if imgui.TreeNode("Contact"):
                    if imgui.TreeNode("Overview"):
                        state.contact_overview.draw()
                        imgui.TreePop()
                    if imgui.TreeNode("Browser"):
                        state.contact_browser.draw()
                        imgui.TreePop()
                    imgui.TreePop()
                imgui.EndTabItem()
            elif ui_state.debug_tab_active:
                state.contact_browser.clear()
                state.contact_overview.remove()
                ui_state.debug_tab_active = False

            imgui.EndTabBar()

        imgui.PopItemWidth()

    return callback


def _update_mesh(state: SimulationState, mesh_name: str):
    x = state.fem.data.x.numpy()  # (N, 3)
    ps.get_volume_mesh(mesh_name).update_vertex_positions(x)
    ps.get_surface_mesh(mesh_name).update_vertex_positions(x)


def parse_args():
    parser = argparse.ArgumentParser(description="GPU VBD Simulation UI")
    parser.add_argument(
        "--fem-elasto-dynamics",
        type=str,
        required=True,
        help="file.h5:group/path to a serialized FemElastoDynamics object.",
        dest="fem_elasto_dynamics",
    )
    parser.add_argument(
        "--sim-params",
        type=str,
        required=False,
        help="file.h5 to simulation parameters to load (optional).",
        dest="sim_params",
    )
    return parser.parse_args()


def main():
    # wp.config.mode = "debug"
    # wp.config.verify_cuda = True
    # wp.config.print_launches = True
    # wp.config.verify_fp = True
    # wp.config.cache_kernels = False
    wp.init()
    args = parse_args()
    fem_cpu = load_fem_dynamics(args.fem_elasto_dynamics)
    params_cpu = {
        s: load_vbd_params(fem_cpu) for s in [SolverType.VBD, SolverType.AAAVBD]
    }
    params_cpu[SolverType.Chebyshev] = load_chebyshev_params(fem_cpu)
    params_cpu[SolverType.Newton] = load_newton_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu)
    ui_state = UIState()
    mesh_name = "Mesh"

    # Setup polyscope
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Simulator")
    ps.init()

    vm = ps.register_volume_mesh(mesh_name, fem_cpu.X.T, fem_cpu.E.T)
    vm.add_scalar_quantity(
        "mug", fem_cpu.lamegU[0, :], defined_on="cells", enabled=True, cmap="blues"
    )
    vm.add_scalar_quantity(
        "lambdag", fem_cpu.lamegU[1, :], defined_on="cells", enabled=False, cmap="blues"
    )
    vm.add_scalar_quantity(
        "m(i)", fem_cpu.m, defined_on="vertices", enabled=False, cmap="reds"
    )
    sm = ps.register_surface_mesh(
        mesh_name, fem_cpu.X.T, state.multimesh.data.F.numpy()
    )
    sm.set_enabled(False)

    if args.sim_params:
        _import_parameters_from_file(args.sim_params, state, ui_state, mesh_name)

    ps.set_user_callback(make_callback(state, ui_state, mesh_name))
    ps.show()


if __name__ == "__main__":
    main()
