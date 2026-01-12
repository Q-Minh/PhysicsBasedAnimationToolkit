# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .base import BaseSolver
import typing
import numpy as np


class Params:
    _vbd_params: pbat.sim.algorithm.vbd.Params
    _broyden_params: pbat.sim.algorithm.vbd.BroydenParams

    def __init__(self):
        self._vbd_params = pbat.sim.algorithm.vbd.Params()
        self._broyden_params = pbat.sim.algorithm.vbd.BroydenParams()

    @property
    def vbd_params(self):
        return self._vbd_params

    @vbd_params.setter
    def vbd_params(self, value: pbat.sim.algorithm.vbd.Params):
        self._vbd_params = value

    @property
    def broyden_params(self):
        return self._broyden_params

    @broyden_params.setter
    def broyden_params(self, value: pbat.sim.algorithm.vbd.BroydenParams):
        self._broyden_params = value


class BroydenSolver(BaseSolver):
    _params: ParameterObject

    def __init__(self):
        super().__init__("Broyden")
        self._params = ParameterObject(
            Params(), {"vbd_params": None, "broyden_params": None}
        )

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
        params: pbat.sim.algorithm.vbd.Params = self._params.params.vbd_params
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
        params: Params = self._params.params
        vbd = params.vbd_params
        broyden = params.broyden_params
        if callback is None:
            callback = lambda: None
        # Hack to register the initial iterate, because Anderson's initialize_solve
        # computes both the initial iterate and the first iteration.
        xcpy = fem.x.copy()
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd)
        callback()
        fem.x = xcpy
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, broyden)
        callback()
        while vbd.k < vbd.n_max_iters:
            if contact.requires_bounds_computation:
                contact.compute_displacement_bounds(fem.x)
            pbat.sim.algorithm.vbd.iterate(fem, contact, vbd, broyden)
            fem.x = contact.truncate_displaced_positions(fem.x, fem.dmask)
            callback()
        fem.back_substitute_integrated_positions_into_velocities()

    def serialize(self, archive: pbat.io.Archive):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        broyden: pbat.sim.algorithm.vbd.BroydenParams = params.broyden_params
        vbd.serialize(archive)
        broyden.serialize(archive)

    def deserialize(self, archive: pbat.io.Archive):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        broyden: pbat.sim.algorithm.vbd.BroydenParams = params.broyden_params
        vbd.deserialize(archive)
        broyden.deserialize(archive)

    def serialize_problem(
        self,
        archive: pbat.io.Archive,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        broyden: pbat.sim.algorithm.vbd.BroydenParams = params.broyden_params
        fem.serialize(archive["fem"])
        contact.serialize(archive["contact"])
        vbd.serialize(archive["vbd/params"])
        broyden.serialize(archive["vbd/broyden_params"])
