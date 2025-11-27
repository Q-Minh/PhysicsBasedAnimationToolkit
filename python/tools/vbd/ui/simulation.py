# type: ignore
import numpy as np
from pbatoolkit import pbat, pypbat
import polyscope as ps
import polyscope.imgui as imgui
from .solver import Solver
from .contact import Contact
from ..utils import transform_library as tlib


class Simulation:
    _fem_dynamics: pbat.sim.dynamics.FemElastoDynamics
    _dt: float
    _bdf_scheme: int
    _profiler: pypbat.profiling.Profiler

    _fem_dynamics_vm: ps.VolumeMesh
    _fem_dynamics_dirichlet_pc: ps.PointCloud
    _simulate: bool
    _solver: Solver
    _contact: Contact
    _transform_library: tlib.TransformLibrary
    _v0: np.ndarray

    def __init__(self):
        self._fem_dynamics = pbat.sim.dynamics.FemElastoDynamics()
        self._profiler = pypbat.profiling.Profiler()
        self._fem_dynamics_vm = None
        self._fem_dynamics_dirichlet_pc = None
        self._simulate = False
        self._solver = Solver()
        self._contact = Contact()
        self._transform_library = tlib.TransformLibrary()
        self._v0 = np.array([])
        self._dt = 1e-2
        self._bdf_scheme = 1

    def draw(self):
        default_button_size = [imgui.GetWindowWidth() / 2.1, 0]
        tab_flags = (
            imgui.ImGuiTabBarFlags_Reorderable
            | imgui.ImGuiTabBarFlags_FittingPolicyScroll
            | imgui.ImGuiTabBarFlags_TabListPopupButton
        )
        if imgui.BeginTabBar("Sim bar", tab_flags):
            if imgui.BeginTabItem("Solver", True, tab_flags)[0]:
                self._solver.draw()
                imgui.EndTabItem()
            if imgui.BeginTabItem("Contact", True, tab_flags)[0]:
                self._contact.draw()
                imgui.EndTabItem()
            if imgui.BeginTabItem("ODE", True, tab_flags)[0]:
                imgui.PushID("ODE")
                _, self._dt = imgui.InputFloat("Time Step", self._dt, 1e-5, 1.0, "%.5f")
                _, self._bdf_scheme = imgui.InputInt(
                    "BDF Scheme", self._bdf_scheme, step=1
                )
                self._bdf_scheme = max(1, min(6, self._bdf_scheme))
                imgui.PopID()
                imgui.EndTabItem()
            imgui.EndTabBar()

        _, self._simulate = imgui.Checkbox("Simulate", self._simulate)
        step = imgui.Button("Step", default_button_size)
        reset = imgui.Button("Reset", default_button_size)
        if reset:
            self._reset_sim()

        if step or self._simulate:
            self._step()

    def set_visible(self, visible: bool):
        if self._fem_dynamics_vm is not None:
            self._fem_dynamics_vm.set_enabled(visible)
        if self._fem_dynamics_dirichlet_pc is not None:
            self._fem_dynamics_dirichlet_pc.set_enabled(visible)
        self._solver.set_visible(visible)
        self._contact.set_visible(visible)

    def on_simulation_scenario_created(
        self,
        fem_dynamics: pbat.sim.dynamics.FemElastoDynamics,
        contact_dynamics: pbat.sim.contact.MeshDynamics,
        transform_library: tlib.TransformLibrary = None,
    ):
        self._fem_dynamics = fem_dynamics
        self._v0 = self._fem_dynamics.v.copy()  # store initial velocity for reset
        self._contact.on_new_contact_dynamics(contact_dynamics)
        self._transform_library = transform_library
        self._reset_sim()
        self._fem_dynamics_vm = ps.register_volume_mesh(
            "FEM Elasto Dynamics", self._fem_dynamics.X.T, self._fem_dynamics.E.T
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Mass", np.log10(self._fem_dynamics.m + 1), defined_on="vertices"
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame mu",
            np.log10(self._fem_dynamics.lamegU[0, :] + 1),
            defined_on="cells",
            enabled=True,
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame lambda",
            np.log10(self._fem_dynamics.lamegU[1, :] + 1),
            defined_on="cells",
        )
        d_nodes = self._fem_dynamics.dirichlet_nodes
        self._fem_dynamics_dirichlet_pc = ps.register_point_cloud(
            "sim - Dirichlet", self._fem_dynamics.X[:, d_nodes].T
        )
        self._fem_dynamics_dirichlet_pc.add_scalar_quantity(
            "Group", self._fem_dynamics.dmask[d_nodes], cmap="turbo", enabled=True
        )

    def _reset_sim(self):
        self._fem_dynamics.set_time_integration_scheme(dt=self._dt, s=self._bdf_scheme)
        self._fem_dynamics.set_initial_conditions(self._fem_dynamics.X, self._v0)

    def _step(self):
        ps.error("Simulation stepping not implemented yet")
