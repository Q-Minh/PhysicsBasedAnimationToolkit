# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope.imgui as imgui
from .base import BaseSolver
import typing


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

    def solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        callback: typing.Callable[None, None] | None = None,
    ):
        if callback is None:
            callback = lambda: None
        callback()
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        for k in range(params.n_max_iters):
            if contact.requires_bounds_computation:
                contact.compute_displacement_bounds(fem.x)
            pbat.sim.algorithm.vbd.iterate(fem, contact, params)
            fem.x = contact.truncate_displacement(fem.x, fem.dmask)
            callback()
        fem.back_substitute_integrated_positions_into_velocities()

    def serialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.serialize(archive)

    def deserialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.deserialize(archive)

    def serialize_problem(
        self,
        archive: pbat.io.Archive,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        fem.serialize(archive["fem"])
        contact.serialize(archive["contact"])
        params: pbat.sim.algorithm.vbd.Params = self._params.params
        params.serialize(archive["vbd/params"])
