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
    _dmin: float = float("inf")

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
        self._dmin = float("inf")

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
        device_config = pbat.geometry.DeviceConfig()
        device = pbat.geometry.Device(device_config)
        contact_dynamics.initialize(device)
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
            if self._contact.contact_dynamics is not None:
                self._contact.contact_dynamics.compute_displacement_bounds(
                    self._fem_dynamics.X
                )

    def _step(self):
        if self._fem_dynamics is None:
            ps.error("No simulation scenario loaded!")
            return
        self._apply_procedural_constraints()
        self._profiler.begin_frame("Physics")
        self._fem_dynamics.setup_time_integration_optimization(
            initialization_strategy=self._fem_dynamics_init_strategy
        )
        try:
            self._solver.solve(
                self._fem_dynamics,
                self._contact.contact_dynamics,
            )
        except Exception as e:
            ps.error(f"Simulation step failed:\n{e}")
            self._simulate = False
            self._profiler.end_frame("Physics")
            return
        self._fem_dynamics.step()
        self._profiler.end_frame("Physics")
        self._update_visuals_after_position_change()
        self._t += 1

    def _update_visuals_after_position_change(self):
        if self._fem_dynamics_vm is not None:
            self._fem_dynamics_vm.update_vertex_positions(self._fem_dynamics.x.T)
            bv = self._contact.contact_dynamics.ogc_state.bv
            bvmax = bv.max()
            bx = np.full(self._fem_dynamics.x.shape[1], bvmax)
            bx[self._contact.contact_dynamics.dynamic_meshes.V] = bv
            r = self._contact.contact_dynamics.params.ogc_params.r
            self._fem_dynamics_vm.add_scalar_quantity(
                "-bv", -bx, defined_on="vertices", cmap="coolwarm", vminmax=(-r, 0)
            )
            self._fem_dynamics_vm.add_scalar_quantity(
                "bvr",
                bx <= self._contact.contact_dynamics.params.ogc_params.r,
                defined_on="vertices",
                cmap="reds",
                vminmax=(0, 1),
            )
            x = self._fem_dynamics.x
            bdf: pbat.sim.integration.Bdf = self._fem_dynamics.bdf
            xt = -bdf.inertia().reshape((3, -1), order="F")
            bt = bdf.beta_tilde
            if self._contact.requires_force_display:
                self._contact.on_contact_force_display_requested(
                    x,
                    xt,
                    bt,
                )
            if self._contact.requires_stencil_display:
                self._contact.on_stencil_display_requested(x, xt, bt)
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
        if imgui.Button("Dump", button_size):
            self._serialize_problem()
        imgui.Text(f"Time step={self._t}, t={self._t * self._dt:.4f}s")
        if reset:
            self._reset_sim()

        if step or self._simulate:
            self._step()

        dmin = self._contact.contact_dynamics.ogc_state.bv.min()
        if self._dmin != dmin:
            self._dmin = dmin
        imgui.Text(f"Minimum displacement bound: {self._dmin:.16f}")
        imgui.Text(
            f"Query radius: {self._contact.contact_dynamics.params.ogc_params.rq:.6f}"
        )
        imgui.Text(f"# contacts: {self._contact.contact_dynamics.num_contacts}")
        self._draw_trajectory_ui()
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
        self._convergence.draw()

    def _draw_trajectory_ui(self):
        self._trajectory.set_timestep(self._dt)
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

    def _draw_io_ui(self, button_size):
        imgui.PushID("IO")
        export = imgui.Button("Export Sim Parameters", button_size)
        if imgui.IsItemHovered():
            imgui.BeginTooltip()
            imgui.SetTooltip("Export the solver parameters of this simulation.")
            imgui.EndTooltip()
        if export:
            self._serialize_simulation_parameters()
        load_scenario = imgui.Button("Load Sim Parameters", button_size)
        if imgui.IsItemHovered():
            imgui.BeginTooltip()
            imgui.SetTooltip("Load solver parameters of a simulation or trajectory.")
            imgui.EndTooltip()
        if load_scenario:
            self._deserialize_simulation_parameters()
        imgui.PopID()

    def _serialize_simulation_parameters(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Save scenario (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.Overwrite)
                self._contact.serialize(archive["Contact"])
                self._solver.serialize(archive["Solver"])
                archive = None
                gc.collect()  # Force garbage collection to close the archive...
                with h5.File(file_path, "a") as f:
                    f.attrs["dt"] = self._dt
                    f.attrs["bdf_scheme"] = self._bdf_scheme
                    f.attrs["fem_dynamics_init_strategy"] = (
                        self._fem_dynamics_init_strategy.value
                    )
        except Exception as e:
            ps.error(f"Error saving scenario:\n{e}")
        finally:
            root.destroy()

    def _deserialize_simulation_parameters(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Load scenario or trajectory (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        try:
            if not file_path:
                return
            archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.ReadOnly)
            self._contact.deserialize(archive["Contact"])
            self._solver.deserialize(archive["Solver"])
            archive = None
            gc.collect()  # Force garbage collection to close the archive...
            with h5.File(file_path, "r") as f:
                self._dt = f.attrs["dt"]
                self._bdf_scheme = f.attrs["bdf_scheme"]
                self._fem_dynamics_init_strategy = (
                    pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization(
                        f.attrs["fem_dynamics_init_strategy"]
                    )
                )
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
        for start, tup in zip(
            self._XP[:-1],
            self._transform_library.all_transformed_nodes(self._t, self._dt),
        ):
            _, dnodes = tup
            if len(dnodes) != 0:
                fem.dmask[start + dnodes] = 1
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

    def _serialize_problem(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            title="Save problem (HDF5)",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        try:
            if file_path:
                archive = pbat.io.Archive(file_path, flags=pbat.io.AccessMode.Overwrite)
                xt = self._fem_dynamics.x.copy()
                self._apply_procedural_constraints()
                self._solver.serialize_problem(
                    archive, self._fem_dynamics, self._contact.contact_dynamics
                )
                archive.write_metadata(
                    "initialization_strategy",
                    int(self._fem_dynamics_init_strategy.value),
                )
                archive = None
                gc.collect()  # Force garbage collection to close the archive...
                self._fem_dynamics.x = xt
        except Exception as e:
            ps.error(f"Error saving problem:\n{e}")
        finally:
            root.destroy()
