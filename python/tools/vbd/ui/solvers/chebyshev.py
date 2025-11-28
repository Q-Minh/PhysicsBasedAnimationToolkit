# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .serialize import serialize_solver_iteration
from .base import BaseSolver


class Params:
    _vbd_params: pbat.sim.algorithm.vbd.Params
    _chebyshev_params: pbat.sim.algorithm.vbd.ChebyshevParams

    def __init__(self):
        self._vbd_params = pbat.sim.algorithm.vbd.Params()
        self._chebyshev_params = pbat.sim.algorithm.vbd.ChebyshevParams()

    @property
    def vbd_params(self):
        return self._vbd_params

    @vbd_params.setter
    def vbd_params(self, value: pbat.sim.algorithm.vbd.Params):
        self._vbd_params = value

    @property
    def chebyshev_params(self):
        return self._chebyshev_params

    @chebyshev_params.setter
    def chebyshev_params(self, value: pbat.sim.algorithm.vbd.ChebyshevParams):
        self._chebyshev_params = value


class ChebyshevSolver(BaseSolver):
    _params: ParameterObject

    def __init__(self):
        super().__init__("Chebyshev")
        self._params = ParameterObject(
            Params(), {"vbd_params": None, "chebyshev_params": None}
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

    def integrate(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        init: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
        archive: pbat.io.Archive | None = None,
    ):
        params: Params = self._params.params
        vbd = params.vbd_params
        chebyshev = params.chebyshev_params
        vbd.strategy = init
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, chebyshev)
        grp = (
            archive["pbat.sim.algorithm.vbd.Chebyshev.Integrate"]
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
            archive["pbat.sim.algorithm.vbd.Chebyshev.Solve"]
            if archive is not None
            else None
        )
        params: Params = self._params.params
        vbd = params.vbd_params
        chebyshev = params.chebyshev_params
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, chebyshev)
        while chebyshev.k < vbd.n_max_iters:
            if grp is not None:
                serialize_solver_iteration(fem, chebyshev.k, grp)
            pbat.sim.algorithm.vbd.iterate(fem, contact, vbd, chebyshev)
        fem.back_substitute_integrated_positions_into_velocities()
        if grp is not None:
            serialize_solver_iteration(fem, chebyshev.k, grp, post_solve=True)
