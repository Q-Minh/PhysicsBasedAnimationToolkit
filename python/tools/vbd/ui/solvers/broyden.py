# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .serialize import serialize_solver_iteration
from .base import BaseSolver


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
        if imgui.TreeNode("Parameters"):
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

    def integrate(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        init: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
        archive: pbat.io.Archive | None = None,
    ):
        # fem.setup_time_integration_optimization(initialization_strategy=init)
        params: Params = self._params.params
        vbd = params.vbd_params
        broyden = params.broyden_params
        vbd.strategy = init
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, broyden)
        grp = (
            archive["pbat.sim.algorithm.vbd.Broyden.Integrate"]
            if archive is not None
            else None
        )
        if grp is not None:
            fem.serialize(grp)
        self._solve(fem, contact, archive=grp)
        fem.step()

    def _solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        archive: pbat.io.Archive | None = None,
    ):
        grp = (
            archive["pbat.sim.algorithm.vbd.Broyden.Solve"]
            if archive is not None
            else None
        )
        params: Params = self._params.params
        vbd = params.vbd_params
        broyden = params.broyden_params
        # pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, broyden)
        while broyden.k < vbd.n_max_iters:
            if grp is not None:
                serialize_solver_iteration(fem, broyden.k, grp)
            pbat.sim.algorithm.vbd.iterate(fem, contact, vbd, broyden)
        fem.back_substitute_integrated_positions_into_velocities()
        if grp is not None:
            serialize_solver_iteration(fem, broyden.k, grp, post_solve=True)
