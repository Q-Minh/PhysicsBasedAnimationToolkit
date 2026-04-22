# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .base import BaseSolver
import typing
import numpy as np


class VbdSolver(BaseSolver):
    _params: ParameterObject
    _step_size: float

    def __init__(self):
        super().__init__("VBD")
        self._params = ParameterObject(pbat.sim.algorithm.vbd.Params())
        self._step_size = 1.0

    def draw(self):
        imgui.PushID(self._name)
        if imgui.TreeNode(f"{type(self._params.params).__name__}"):
            self._params.draw()
            imgui.TreePop()
        if imgui.TreeNode(f"Control"):
            _, self._step_size = imgui.InputFloat("Step Size", self._step_size)
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
        GVVp, GVVadj, colors = pbat.sim.algorithm.vbd.vertex_colors(fem.E, n_nodes)
        params.with_vertex_element_adjacency_graph(
            GVGp, GVGe, GVGilocal
        ).with_vertex_colors(GVVp, GVVadj, colors).construct()

    def solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        callback: typing.Callable[None, None] | None = None,
    ):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        if callback is None:
            callback = lambda: None
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, params)
        pbat.sim.algorithm.vbd.solve(fem, contact, params)
        # callback()
        # for k in range(params.n_max_iters):
        #     if contact.requires_constraint_set_update:
        #         contact.update_constraint_set(fem.x)
        #     xk = fem.x.copy()
        #     pbat.sim.algorithm.vbd.iterate(fem, contact, params)
        #     fem.x = xk + self._step_size * (fem.x - xk)
        #     fem.x = contact.restore_feasibility(fem.x, fem.dmask)
        #     callback()
        # fem.back_substitute_integrated_positions_into_velocities()

    def serialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.serialize(archive, minimal=True)

    def deserialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.deserialize(archive)
        params.construct()

    def serialize_problem(
        self,
        archive: pbat.io.Archive,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        fem.serialize(archive["fem"])
        contact.serialize(archive["contact"])
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.serialize(archive["vbd/params"], minimal=False)
