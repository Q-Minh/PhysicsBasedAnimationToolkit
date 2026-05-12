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


def draw_params(obj):
    for name, value in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        if (
            isinstance(getattr(type(obj), name, None), property)
            and getattr(type(obj), name).fset is None
        ):
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


def parse_archive_path(spec: str) -> tuple[str, str]:
    """Parse 'file.h5:group/path' into (file_path, group_path)."""
    parts = spec.split(":")
    if len(parts) >= 2:
        return parts[0], ":".join(parts[1:])
    return parts[0], ""


def load_fem_dynamics(spec: str) -> pbat.sim.dynamics.FemElastoDynamics:
    """Deserialize FemElastoDynamics from an h5 archive at file:group."""
    file_path, group_path = parse_archive_path(spec)
    archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
    fem_cpu = pbat.sim.dynamics.FemElastoDynamics()
    fem_cpu.deserialize(archive[group_path] if group_path else "/")
    archive = None
    gc.collect()
    return fem_cpu


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


class SolverType(enum.Enum):
    VBD = 0
    AAAVBD = 1


class UIState:
    def __init__(self):
        self.request_reset: bool = False
        self.item_width: int = 250
        self.screenshot_after_step: bool = False
        self.debug_tab_active: bool = False


class SimulationState:
    def __init__(
        self,
        fem_cpu: pbat.sim.dynamics.FemElastoDynamics,
        params_cpu: dict[SolverType, pbat.sim.algorithm.vbd.Params],
    ):
        self.fem_cpu = fem_cpu
        self.params_cpu = params_cpu

        # Time integration (modifiable from UI)
        self.dt: float = 1e-2
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
        self.params = {s: gpu.vbd.params.Params(p) for s, p in params_cpu.items()}
        self.capture = None

        # Build collision geometry
        n_nodes = fem_cpu.X.shape[1]
        multimesh_cpu = pbat.sim.contact.MultiMesh()
        multimesh_cpu.construct_from_tetrahedral_mesh(
            fem_cpu.E, np.full(n_nodes, 0, dtype=np.int64), n_components=1
        )
        self.multimesh = gpu.contact.multimesh.MultiMesh(multimesh_cpu)
        self.ogc_params = gpu.contact.ogc.OgcParams()
        ogc = gpu.contact.ogc.Ogc(
            self.fem.data.x,
            self.multimesh,
            self.ogc_params,
        )
        ogc.enable_adaptive_query_radius(self.fem.xt, self.fem.data.xtilde)
        self.contact_params = gpu.contact.dynamics.Params()
        self.contact = gpu.contact.dynamics.MeshDynamics(ogc, self.contact_params)
        # NOTE: This will need to be updated if adding new Solvers with
        # different storage location for Qnk, Qfk
        self.contact.enable_adaptive_penalty_parameters(
            self.params[self.solver].data.Qnk, self.params[self.solver].data.Qfk
        )
        self.solvers = {
            SolverType.VBD: gpu.vbd.solver.VbdSolver(),
            SolverType.AAAVBD: gpu.vbd.aaasolver.AaaVbdSolver(),
        }
        self.contact_browser = gpu.contact.debug.ogc.OgcContactBrowser(
            self.fem.data.x, self.contact.ogc
        )

        # Simulation state
        self.simulate: bool = False
        self.t: int = 0
        self.until_t: int = -1

    def step(self):
        self.fem.setup_time_integration_optimization(self.init_strategy)
        self.solvers[self.solver].solve(
            self.fem, self.params[self.solver], self.contact
        )
        self.fem.step()
        self.t += 1

        self.contact_browser.update(self.fem.data.x, self.contact.ogc)

    def reset(self):
        self.t = 0
        self.fem_cpu.set_time_integration_scheme(self.dt, self.bdf_scheme)
        self.fem_cpu.set_initial_conditions(self.fem_cpu.X, self.fem_cpu.v * 0.0)
        self.fem = gpu.elasticity.fem.FemElastoDynamics(self.fem_cpu)
        self.params = {s: gpu.vbd.params.Params(p) for s, p in self.params_cpu.items()}
        ogc = gpu.contact.ogc.Ogc(
            self.fem.data.x,
            self.multimesh,
            self.ogc_params,
        )
        ogc.enable_adaptive_query_radius(self.fem.xt, self.fem.data.xtilde)
        self.contact = gpu.contact.dynamics.MeshDynamics(ogc, self.contact_params)
        # NOTE: This will need to be updated if adding new Solvers with
        # different storage location for Qnk, Qfk
        self.contact.enable_adaptive_penalty_parameters(
            self.params[self.solver].data.Qnk, self.params[self.solver].data.Qfk
        )
        self.solvers = {
            SolverType.VBD: gpu.vbd.solver.VbdSolver(),
            SolverType.AAAVBD: gpu.vbd.aaasolver.AaaVbdSolver(),
        }
        self.contact_browser.update(self.fem.data.x, self.contact.ogc)
        self.capture = None


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
                        draw_params(state.params_cpu[state.solver])
                        imgui.TreePop()
                    imgui.TreePop()

                # --- Contact parameters ---
                if imgui.TreeNode("Contact"):
                    if imgui.TreeNode("Statistics"):
                        nvv, nve, nvf, nee = state.contact.ogc.num_contacts
                        imgui.Text(f"# Vertex-Vertex Contacts: {nvv}")
                        imgui.Text(f"# Vertex-Edge Contacts: {nve}")
                        imgui.Text(f"# Vertex-Face Contacts: {nvf}")
                        imgui.Text(f"# Edge-Edge Contacts: {nee}")
                        imgui.TreePop()
                    if imgui.TreeNode("OGC"):
                        draw_params(state.ogc_params)
                        imgui.TreePop()
                    if imgui.TreeNode("Dynamics"):
                        draw_params(state.contact_params)
                        imgui.TreePop()
                    imgui.TreePop()

                imgui.Separator()

                # --- Simulation controls ---
                _, ui_state.screenshot_after_step = imgui.Checkbox(
                    "Screenshot", ui_state.screenshot_after_step
                )
                _, state.simulate = imgui.Checkbox("Simulate", state.simulate)
                imgui.SameLine()
                _, state.until_t = imgui.InputInt("Until", state.until_t)
                if state.t == state.until_t:
                    state.simulate = False

                if imgui.Button("Reset") or ui_state.request_reset:
                    state.reset()
                    _update_mesh(state, mesh_name)

                # --- Continuous simulation ---
                request_step = state.simulate or imgui.Button("Step")
                if request_step:
                    if ui_state.screenshot_after_step and state.t == 0:
                        ps.screenshot("{:08d}.png".format(state.t))
                    # try:
                    state.step()
                    # except Exception as e:
                    #     ps.error("Simulation step failed: {}".format(e))
                    _update_mesh(state, mesh_name)
                    if ui_state.screenshot_after_step:
                        ps.screenshot("{:08d}.png".format(state.t))

                imgui.EndTabItem()

            if imgui.BeginTabItem("Debug", True)[0]:
                ui_state.debug_tab_active = True
                if imgui.TreeNode("Contact"):
                    state.contact_browser.draw()
                    imgui.TreePop()
                imgui.EndTabItem()
            elif ui_state.debug_tab_active:
                state.contact_browser.clear()
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
        "--vbd-params",
        type=str,
        default=None,
        help="file.h5:group/path to serialized VBD Params (optional, uses defaults otherwise).",
        dest="vbd_params",
    )
    return parser.parse_args()


def main():
    # wp.config.mode = "debug"
    # wp.config.verify_cuda = True
    # wp.config.print_launches = True
    # wp.config.verify_fp = True
    wp.init()
    args = parse_args()
    fem_cpu = load_fem_dynamics(args.fem_elasto_dynamics)
    params_cpu = {s: load_vbd_params(fem_cpu, args.vbd_params) for s in SolverType}
    state = SimulationState(fem_cpu, params_cpu)
    # Setup polyscope
    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Simulator")
    ps.init()
    mesh_name = "Mesh"
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
    ps.set_user_callback(make_callback(state, UIState(), mesh_name))
    ps.show()


if __name__ == "__main__":
    main()
