# type: ignore
"""
Lightweight GPU VBD simulation UI using Polyscope + imgui.

Usage:
    python -m python.tools.sim.ui --fem-elasto-dynamics path/to/fem.h5:hdf5/group/to/fem [--sim-params params.toml ...] [--set KEY=VALUE]
"""

import argparse
import h5py
import tkinter as tk
from tkinter import filedialog
import warp as wp
import polyscope as ps
import polyscope.imgui as imgui
from pbatoolkit import pbat
import numpy as np
from . import config, gpu
from .panels.common import draw_params, try_draw_tooltip
from .colliders import collider_faces
from .panels.kinematics import draw_kinematics_panel
from .panels.profiling import draw_profiling_panel, profile_step
from .panels.static import draw_static_panel
from .panels.status import draw_status_panel
from .state import (
    SimulationState,
    SolverType,
    CDType,
    load_problem,
    load_params,
    should_save,
    init_h5_serialization,
    serialize_params,
)


SOLVER_SUB_PARAMS: dict[SolverType, dict] = {
    SolverType.VBD: {},
    SolverType.AAAVBD: {"vbd_params": None},
    SolverType.Chebyshev: {"vbd_params": None},
    SolverType.AndersonSolver: {"vbd_params": None},
    SolverType.Newton: {"newton": {"line_search": {}}},
}
# Parameters a solver ignores, hidden from its Params panel. The stencil gradient acceleration
# parameters of pbat.sim.algorithm.vbd.Params belong to AAAVBD only.
SOLVER_HIDDEN_PARAMS: dict[SolverType, frozenset] = {
    s: gpu.vbd.params.STENCIL_GRADIENT_ACCELERATION_KEYS
    for s in (SolverType.VBD, SolverType.Chebyshev, SolverType.AndersonSolver)
}


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
    paths: list[str],
    state: SimulationState,
    ui_state: UIState,
    mesh_name: str,
    overrides: list[str] = [],
):
    config.load_sim_params(state, paths, overrides)
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
                        params_cpu = state.params_cpu[state.solver]
                        if state.solver == SolverType.AAAVBD:
                            # AAAVBD's own parameters at the top (its GPU-only fields, then the stencil
                            # gradient acceleration fields of the wrapped vbd::Params); the fields it
                            # shares with VBD stay in the vbd_params sub-tree
                            stencil = gpu.vbd.params.STENCIL_GRADIENT_ACCELERATION_KEYS
                            draw_params(params_cpu, exclude=frozenset({"vbd_params"}))
                            draw_params(params_cpu.vbd_params, only=stencil)
                            draw_params(params_cpu, SOLVER_SUB_PARAMS[state.solver], exclude=stencil)
                        else:
                            draw_params(
                                params_cpu,
                                SOLVER_SUB_PARAMS.get(state.solver),
                                exclude=SOLVER_HIDDEN_PARAMS.get(state.solver, frozenset()),
                            )
                        # Own ID scopes: Newton's params have a linear_solver combo at this level
                        if state.solver == SolverType.Newton:
                            imgui.PushID("newton_linear_solver_params")
                            if imgui.TreeNode("LinearSolver"):
                                draw_params(state.newton_linear_solver_params)
                                imgui.TreePop()
                            imgui.PopID()
                        # Shared by every solver, Newton included (check_interval is VBD-family only)
                        imgui.PushID("convergence_params")
                        if imgui.TreeNode("Convergence"):
                            draw_params(state.convergence_params)
                            imgui.TreePop()
                        imgui.PopID()
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
                            defaultextension=".toml",
                            filetypes=[
                                ("TOML configs", "*.toml"),
                                ("HDF5 files", "*.h5 *.hdf5"),
                                ("All files", "*.*"),
                            ],
                        )
                        root.destroy()
                        if path and path.lower().endswith(".toml"):
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(config.dumps(config.params_dict(state)))
                        elif path:
                            with h5py.File(path, "w") as f:
                                serialize_params(state, f)
                    if imgui.Button("Import Parameters"):
                        root = tk.Tk()
                        root.withdraw()
                        path = filedialog.askopenfilename(
                            title="Import simulation parameters",
                            defaultextension=".toml",
                            filetypes=[
                                ("Parameter files", "*.toml *.h5 *.hdf5"),
                                ("All files", "*.*"),
                            ],
                        )
                        root.destroy()
                        if path:
                            _import_parameters_from_file(
                                [path], state, ui_state, mesh_name
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
                                ui_state.serialization_h5 = init_h5_serialization(
                                    state.fem_cpu, path, ui_state.screenshot_fps
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

                # --- Feature panels ---
                # One draw_<feature>_panel(state, ui_state) call per feature, implemented in
                # python/tools/sim/panels/<feature>.py (see panels/__init__.py).
                draw_kinematics_panel(state, ui_state)
                draw_profiling_panel(state, ui_state)
                draw_static_panel(state, ui_state)
                draw_status_panel(state, ui_state)

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

                    with profile_step(state):
                        state.step()
                    _update_mesh(state, mesh_name)

                    if (
                        ui_state.screenshot_after_step
                        or ui_state.serialization_h5 is not None
                    ):
                        if should_save(state.t, state.dt, ui_state.screenshot_fps):
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
    problem = parser.add_mutually_exclusive_group(required=True)
    problem.add_argument(
        "--fem-elasto-dynamics",
        type=str,
        help="file.h5:group/path to a serialized FemElastoDynamics object.",
        dest="fem_elasto_dynamics",
    )
    problem.add_argument(
        "--scene",
        type=str,
        help="CPU editor scene file (.h5). Static mesh colliders are ignored.",
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
    return parser.parse_args()


def main():
    # wp.config.mode = "debug"
    # wp.config.verify_cuda = True
    # wp.config.print_launches = True
    # wp.config.verify_fp = True
    # wp.config.cache_kernels = False
    wp.init()
    args = parse_args()
    fem_cpu, kinematics, contact_geometry = load_problem(args.fem_elasto_dynamics, args.scene)
    params_cpu = load_params(fem_cpu)
    state = SimulationState(fem_cpu, params_cpu, kinematics, contact_geometry)
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
    # Static colliders never move, so they are drawn once
    static_faces = collider_faces(
        fem_cpu.E, state.multimesh.data.F.numpy(), fem_cpu.X.shape[1]
    )
    if static_faces.shape[0] > 0:
        ps.register_surface_mesh("Static colliders", fem_cpu.X.T, static_faces)
        # Frame the bodies, not the (possibly huge, e.g. infinite plane) colliders
        XB = fem_cpu.X[:, np.unique(fem_cpu.E)]
        low, high = XB.min(axis=1), XB.max(axis=1)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(low, high)
        ps.set_length_scale(float(np.linalg.norm(high - low)))

    if args.sim_params or args.overrides:
        _import_parameters_from_file(
            args.sim_params, state, ui_state, mesh_name, args.overrides
        )

    ps.set_user_callback(make_callback(state, ui_state, mesh_name))
    ps.show()


if __name__ == "__main__":
    main()
