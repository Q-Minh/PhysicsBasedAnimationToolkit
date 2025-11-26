# type: ignore
import numpy as np
from pbatoolkit import pbat, pypbat
import polyscope as ps
import polyscope.imgui as imgui
from .solver import Solver
from .contact import Contact


class Simulation:
    _fem_dynamics: pbat.sim.dynamics.FemElastoDynamics
    _profiler: pypbat.profiling.Profiler

    _fem_dynamics_vm: ps.VolumeMesh
    _fem_dynamics_dirichlet_pc: ps.PointCloud
    _simulate: bool
    _solver: Solver
    _contact: Contact

    def __init__(self):
        self._fem_dynamics = pbat.sim.dynamics.FemElastoDynamics()
        self._profiler = pypbat.profiling.Profiler()
        self._fem_dynamics_vm = None
        self._fem_dynamics_dirichlet_pc = None
        self._simulate = False
        self._solver = Solver()
        self._contact = Contact()

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
            imgui.EndTabBar()

        _, self._simulate = imgui.Checkbox("Simulate", self._simulate)
        step = imgui.Button("Step", default_button_size)
        reset = imgui.Button("Reset", default_button_size)
        if reset:
            # TODO: We need to rebuild the simulation from the scene tab...
            ps.error("Reset not implemented yet")

        if step or self._simulate:
            # TODO
            ps.error("Simulation stepping not implemented yet")

    def on_fem_elasto_dynamics_created(
        self, fem_dynamics: pbat.sim.dynamics.FemElastoDynamics
    ):
        self._fem_dynamics = fem_dynamics
        if self._fem_dynamics_vm is not None:
            ps.remove_volume_mesh(self._fem_dynamics_vm.name)
        self._fem_dynamics_vm = ps.register_volume_mesh(
            "FEM Elasto Dynamics", self._fem_dynamics.X.T, self._fem_dynamics.E.T
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Mass", np.log10(self._fem_dynamics.m + 1), defined_on="vertices"
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame mu", self._fem_dynamics.lamegU[0, :], defined_on="cells"
        )
        self._fem_dynamics_vm.add_scalar_quantity(
            "Lame lambda",
            self._fem_dynamics.lamegU[1, :],
            defined_on="cells",
        )
        d_nodes = self._fem_dynamics.dirichlet_nodes
        self._fem_dynamics_dirichlet_pc = ps.register_point_cloud(
            "Dirichlet nodes", self._fem_dynamics.X[:, d_nodes].T
        )
        self._fem_dynamics_dirichlet_pc.add_scalar_quantity(
            "Group", self._fem_dynamics.dmask[d_nodes], cmap="turbo"
        )
