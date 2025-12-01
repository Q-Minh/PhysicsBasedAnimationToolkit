# type: ignore
import numpy as np
from pbatoolkit import pbat, pypbat
import polyscope as ps
import polyscope.imgui as imgui
import tkinter as tk
from tkinter import filedialog
import h5py as h5
import gc
from .solver import Solver
from .contact import Contact
from .convergence import Convergence
from .trajectory import Trajectory
from .utils import transform_library as tlib
from . import material


class Simulation:
    _fem_dynamics: pbat.sim.dynamics.FemElastoDynamics
    _dt: float
    _bdf_scheme: int
    _fem_dynamics_init_strategy: (
        pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization
    )
    _t: int
    _profiler: pypbat.profiling.Profiler

    _fem_dynamics_vm: ps.VolumeMesh
    _fem_dynamics_dirichlet_pc: ps.PointCloud
    _simulate: bool
    _solver: Solver
    _contact: Contact
    _convergence: Convergence
    _trajectory: Trajectory
    _transform_library: tlib.TransformLibrary
    _tet_elastic_body_names: list[str]
    _XP: np.ndarray[int]
    _v0: np.ndarray[float]
    _xD: np.ndarray[float]

    def __init__(self):
        self._fem_dynamics = pbat.sim.dynamics.FemElastoDynamics()
        self._profiler = pypbat.profiling.Profiler()
        self._fem_dynamics_vm = None
        self._fem_dynamics_dirichlet_pc = None
        self._simulate = False
        self._solver = Solver()
        self._contact = Contact()
        self._convergence = Convergence()
        self._trajectory = Trajectory()
        self._transform_library = tlib.TransformLibrary()
        self._tet_elastic_body_names = []
        self._XP = np.array([], dtype=int)
        self._v0 = np.array([], dtype=float)
        self._xD = np.array([], dtype=float)
        self._dt = 1e-2
        self._bdf_scheme = 1
        self._t = 0
        self._fem_dynamics_init_strategy = (
            pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization.Position
        )

    def draw(self):
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        tab_flags = (
            imgui.ImGuiTabBarFlags_Reorderable
            | imgui.ImGuiTabBarFlags_FittingPolicyScroll
            | imgui.ImGuiTabBarFlags_TabListPopupButton
        )
        if imgui.BeginTabBar("Sim bar", tab_flags):
            if imgui.BeginTabItem("Integration", True, tab_flags)[0]:
                self._draw_integration_ui(default_button_size)
                imgui.EndTabItem()
            if imgui.BeginTabItem("Solver", True, tab_flags)[0]:
                self._solver.draw()
                imgui.EndTabItem()
            if imgui.BeginTabItem("Contact", True, tab_flags)[0]:
                self._contact.draw()
                imgui.EndTabItem()
            if imgui.BeginTabItem("Convergence", True, tab_flags)[0]:
                self._draw_convergence_ui()
                imgui.EndTabItem()
            if imgui.BeginTabItem("I/O", True, tab_flags)[0]:
                self._draw_io_ui(default_button_size)
                imgui.EndTabItem()
            imgui.EndTabBar()

    def set_visible(self, visible: bool):
        if self._fem_dynamics_vm is not None:
            self._fem_dynamics_vm.set_enabled(visible)
        if self._fem_dynamics_dirichlet_pc is not None:
            self._fem_dynamics_dirichlet_pc.set_enabled(visible)
        self._solver.set_visible(visible)
        self._contact.set_visible(visible)

    def on_simulation_scenario_created(
        self,
        tet_elastic_body_names: list[str],
        XP: np.ndarray[int],
        fem_dynamics: pbat.sim.dynamics.FemElastoDynamics,
        contact_dynamics: pbat.sim.contact.MeshDynamics,
        transform_library: tlib.TransformLibrary = None,
    ):
        self._fem_dynamics = fem_dynamics
        self._v0 = self._fem_dynamics.v.copy()  # store initial velocity for reset
        self._xD = self._fem_dynamics.x.copy()  # store initial position for reset
        self._contact.on_new_contact_dynamics(contact_dynamics)
        self._transform_library = transform_library
        self._tet_elastic_body_names = tet_elastic_body_names
        self._XP = XP
        self._solver.on_simulation_scenario_created(
            self._fem_dynamics, contact_dynamics
        )
        self._fem_dynamics_vm = ps.register_volume_mesh(
            "FEM Elasto Dynamics", self._fem_dynamics.X.T, self._fem_dynamics.E.T
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Mass",
            np.log10(self._fem_dynamics.m + 1),
            defined_on="vertices",
            cmap=material.mass_density_log10_cmap(),
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame mu",
            np.log10(self._fem_dynamics.lamegU[0, :] + 1),
            defined_on="cells",
            cmap=material.lame_parameters_cmap(),
            enabled=True,
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame lambda",
            np.log10(self._fem_dynamics.lamegU[1, :] + 1),
            defined_on="cells",
            cmap=material.lame_parameters_cmap(),
        )
        self._constrain()
        d_nodes = self._fem_dynamics.dirichlet_nodes
        self._fem_dynamics_dirichlet_pc = ps.register_point_cloud(
            "sim - Dirichlet", self._fem_dynamics.X[:, d_nodes].T
        )
        self._reset_sim()

    def _reset_sim(self):
        self._t = 0
        if self._fem_dynamics is not None:
            self._fem_dynamics.set_time_integration_scheme(
                dt=self._dt, s=self._bdf_scheme
            )
            self._fem_dynamics.set_initial_conditions(self._fem_dynamics.X, self._v0)
            self._constrain()
            self._update_visuals_after_position_change()

    def _step(self):
        if self._fem_dynamics is None:
            ps.error("No simulation scenario loaded!")
            return
        self._apply_procedural_constraints()
        self._profiler.begin_frame("Physics")
        self._fem_dynamics.setup_time_integration_optimization(
            initialization_strategy=self._fem_dynamics_init_strategy
        )
        self._solver.solve(
            self._fem_dynamics,
            self._contact.contact_dynamics,
        )
        self._fem_dynamics.step()
        self._profiler.end_frame("Physics")
        self._update_visuals_after_position_change()
        self._t += 1
        self._trajectory.t = self._t

    def _update_visuals_after_position_change(self):
        if self._fem_dynamics_vm is not None:
            self._fem_dynamics_vm.update_vertex_positions(self._fem_dynamics.x.T)
        if self._fem_dynamics_dirichlet_pc is not None:
            d_nodes = self._fem_dynamics.dirichlet_nodes
            if d_nodes.shape[0] != self._fem_dynamics_dirichlet_pc.n_points():
                self._fem_dynamics_dirichlet_pc = ps.register_point_cloud(
                    "sim - Dirichlet", self._fem_dynamics.X[:, d_nodes].T
                )
            else:
                self._fem_dynamics_dirichlet_pc.update_point_positions(
                    self._fem_dynamics.x[:, d_nodes].T
                )

    def _draw_integration_ui(self, button_size):
        imgui.PushID("Integration")
        init_strategies = list(
            pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization
        )
        selected_idx = init_strategies.index(self._fem_dynamics_init_strategy)
        _, selected_idx = imgui.Combo(
            "Initialization Strategy",
            selected_idx,
            [s.name for s in init_strategies],
        )
        self._fem_dynamics_init_strategy = init_strategies[selected_idx]
        _, self._dt = imgui.InputFloat("Time Step", self._dt, 1e-5, 1.0, "%.5f")
        _, self._bdf_scheme = imgui.InputInt("BDF Scheme", self._bdf_scheme, step=1)
        self._bdf_scheme = max(1, min(6, self._bdf_scheme))

        _, self._simulate = imgui.Checkbox("Simulate", self._simulate)
        step = imgui.Button("Step", button_size)
        reset = imgui.Button("Reset", button_size)
        imgui.Text(f"Time step={self._t}, t={self._t * self._dt:.4f}s")
        if reset:
            self._reset_sim()

        if step or self._simulate:
            self._step()

        imgui.PopID()

    def _draw_convergence_ui(self):
        if self._convergence.is_convergence_analysis_requested:
            self._apply_procedural_constraints()
            self._fem_dynamics.setup_time_integration_optimization(
                initialization_strategy=self._fem_dynamics_init_strategy
            )
            self._convergence.analyze_convergence(
                self._solver.selected,
                self._solver.solvers,
                self._fem_dynamics,
                self._contact.contact_dynamics,
            )
            self._fem_dynamics.step()
            self._update_visuals_after_position_change()
            self._t += 1
            self._trajectory.t = self._t
        self._convergence.draw()

    def _draw_io_ui(self, button_size):
        imgui.PushID("IO")
        export = imgui.Button("Export Scenario", button_size)
        if imgui.IsItemHovered():
            imgui.BeginTooltip()
            imgui.SetTooltip("Export the simulation scenario.")
            imgui.EndTooltip()
        if export:
            self._serialize_simulation_scenario()
        load_scenario = imgui.Button("Load Scenario", button_size)
        if imgui.IsItemHovered():
            imgui.BeginTooltip()
            imgui.SetTooltip("Load a simulation scenario or trajectory.")
            imgui.EndTooltip()
        if load_scenario:
            self._deserialize_simulation()
        self._trajectory.draw()
        if self._trajectory.dirty:
            if self._fem_dynamics is not None:
                self._trajectory.undirty(
                    lambda archive: self._fem_dynamics.deserialize(archive)
                )
                self._update_visuals_after_position_change()
                self._t = self._trajectory.t
            else:
                self._trajectory.undirty(lambda archive: None)
        imgui.PopID()

    def _serialize_simulation_scenario(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Save scenario (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5;*.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.Overwrite)
                self._fem_dynamics.serialize(archive)
                # TODO: Implement contact dynamics serialization
                # self._contact.contact_dynamics.serialize(archive)
                self._solver.serialize(archive["Solver"])
                archive = None
                gc.collect()  # Force garbage collection to close the archive...
                with h5.File(file_path, "a") as f:
                    tlib_group = f.create_group("TransformLibrary")
                    self._transform_library.serialize(tlib_group)
                    f["tet_elastic_body_names"] = self._tet_elastic_body_names
                    f["XP"] = self._XP
                    f.attrs["dt"] = self._dt
                    f.attrs["bdf_scheme"] = self._bdf_scheme
                    f.attrs["fem_dynamics_init_strategy"] = (
                        self._fem_dynamics_init_strategy.value
                    )
        except Exception as e:
            ps.error(f"Error saving scenario:\n{e}")
        finally:
            root.destroy()

    def _deserialize_simulation(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Load scenario or trajectory (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5;*.hdf5"), ("All files", "*.*")],
        )
        try:
            if not file_path:
                return
            archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
            self._fem_dynamics.deserialize(archive)
            # TODO: Implement contact dynamics deserialization
            # self._contact.contact_dynamics.deserialize(archive)
            self._solver.deserialize(archive["Solver"])
            # TODO:
            # When the offline simulation runner will be implemented, we will
            # assume that it saved the simulation trajectory (i.e. every time
            # step's FemElastoDynamics) to the archive as the ground truth.
            # At that point, we will also load that trajectory here, and enable
            # playback of the loaded trajectory in the UI, and simulating from
            # any given time step in the loaded trajectory.
            archive = None
            gc.collect()  # Force garbage collection to close the archive...
            with h5.File(file_path, "r") as f:
                tlib_group = f["TransformLibrary"]
                self._transform_library.deserialize(tlib_group)
                self._tet_elastic_body_names = (
                    f["tet_elastic_body_names"][:].astype(str).tolist()
                )
                self._XP = f["XP"][:].astype(int)
                self._dt = f.attrs["dt"]
                self._bdf_scheme = f.attrs["bdf_scheme"]
                self._fem_dynamics_init_strategy = (
                    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization(
                        f.attrs["fem_dynamics_init_strategy"]
                    )
                )
            # NOTE:
            # Because contact dynamics deserialization is not implemented yet,
            # if we load a simulation scenario that does not correspond to the
            # current scene, then the contact dynamics will not be properly
            # initialized. Thus, the next time we perform a simulation step, i.e.
            # via self._step(), the app will crash. However, if we load a scene first,
            # and then switch to simulation mode, and load a simulation scenario that
            # corresponds to the current scene, then everything will work fine. In the
            # future, we will address this by implementing contact dynamics serialization/deserialization.
            self.on_simulation_scenario_created(
                self._tet_elastic_body_names,
                self._XP,
                self._fem_dynamics,
                self._contact.contact_dynamics,
                self._transform_library,
            )
        except Exception as e:
            ps.error(f"Error loading scenario:\n{e}")
            return
        finally:
            root.destroy()

    def _constrain(self):
        fem = self._fem_dynamics
        fem.dmask = np.zeros_like(fem.dmask, dtype=int)
        for name, dnodes in self._transform_library.all_transformed_nodes(
            self._t, self._dt
        ):
            fem.dmask[dnodes] = 1
        fem.constrain(fem.dmask)

    def _apply_procedural_constraints(self):
        fem = self._fem_dynamics
        self._constrain()
        self._xD = fem.x
        for start, end, name in zip(
            self._XP[:-1], self._XP[1:], self._tet_elastic_body_names
        ):
            self._xD[:, start:end] = self._transform_library.apply(
                name, self._xD[:, start:end], self._t, self._dt
            )
        fem.x = self._xD
