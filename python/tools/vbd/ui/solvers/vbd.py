# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .serialize import serialize_solver_iteration
from .base import BaseSolver
import numpy as np


class VbdSolver(BaseSolver):
    _params: ParameterObject

    def __init__(self):
        super().__init__("VBD")
        self._params = ParameterObject(pbat.sim.algorithm.vbd.Params())

    def draw(self):
        imgui.PushID(self._name)
        if imgui.TreeNode(f"{type(self._params.params).__name__}"):
            self._params.draw()
            imgui.TreePop()
        imgui.PopID()

    def on_simulation_scenario_created(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        n_nodes = fem.X.shape[1]
        GVGp, GVGe, GVGilocal = pbat.sim.algorithm.vbd.vertex_element_adjacency_graph(
            fem.E, n_nodes
        )
        colors = pbat.sim.algorithm.vbd.vertex_colors(fem.E, n_nodes)
        params.with_vertex_element_adjacency_graph(
            GVGp, GVGe, GVGilocal
        ).with_vertex_colors(colors).construct()

    def integrate(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        init: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
        archive: pbat.io.Archive | None = None,
    ):
        fem.setup_time_integration_optimization(initialization_strategy=init)
        # bt = fem.bdf.beta_tilde
        # xtilde = fem.x + bt * fem.v + bt**2 * fem.aext()
        # fem.xtilde = xtilde
        # fem.x = fem.xtilde
        grp = None
        if archive is not None:
            grp = archive["pbat.sim.algorithm.vbd.Integrate"]
            fem.serialize(grp)
        self._solve(fem, contact, archive=grp)
        fem.step()

    def _solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        archive: pbat.io.Archive | None = None,
    ):
        grp = None
        if archive is not None:
            grp = archive["pbat.sim.algorithm.vbd.Solve"]
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        # pbat.sim.algorithm.vbd.initialize_solve(fem, contact, params)
        for k in range(params.n_max_iters):
            # if grp is not None:
            #     serialize_solver_iteration(fem, k, grp)
            # if (k + 1) % 5 == 0:
            #     contact.update_environment_contact_constraints(fem.x)
            pbat.sim.algorithm.vbd.iterate(fem, contact, params)
        fem.back_substitute_integrated_positions_into_velocities()
        if grp is not None:
            serialize_solver_iteration(fem, params.n_max_iters, grp, post_solve=True)
