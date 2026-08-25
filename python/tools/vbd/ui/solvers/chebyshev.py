# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .base import BaseSolver
import typing


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
        params: Params = self._params.params
        vbd = params.vbd_params
        chebyshev = params.chebyshev_params
        if callback is None:
            callback = lambda: None
        callback()
        # Update Chebyshev solver for most recent SAL contact framework
        pbat.sim.algorithm.vbd.initialize_solve(fem, contact, vbd, chebyshev)
        callback()
        for k in range(vbd.n_max_iters):
            callback()
            pbat.sim.algorithm.vbd.linearize_constraints(fem, contact)
            converged = pbat.sim.algorithm.vbd.check_convergence(fem, contact, vbd)
            if converged:
                break
            pbat.sim.algorithm.vbd.prepare_subproblem(fem, contact, vbd)
            for kp in range(vbd.n_subproblem_max_iters):
                pbat.sim.algorithm.vbd.iterate(fem, contact, vbd, chebyshev)
            pbat.sim.algorithm.vbd.finalize_subproblem(fem, contact, vbd)
        callback()
        fem.back_substitute_integrated_positions_into_velocities()

    def serialize(self, archive: pbat.io.Archive):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        chebyshev: pbat.sim.algorithm.vbd.ChebyshevParams = params.chebyshev_params
        vbd.serialize(archive, minimal=True)
        chebyshev.serialize(archive, minimal=True)

    def deserialize(self, archive: pbat.io.Archive):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        chebyshev: pbat.sim.algorithm.vbd.ChebyshevParams = params.chebyshev_params
        vbd.deserialize(archive)
        chebyshev.deserialize(archive)

    def serialize_problem(
        self,
        archive: pbat.io.Archive,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        params: Params = self._params.params
        vbd: pbat.sim.algorithm.vbd.Params = params.vbd_params
        chebyshev: pbat.sim.algorithm.vbd.ChebyshevParams = params.chebyshev_params
        fem.serialize(archive["fem"])
        contact.serialize(archive["contact"])
        vbd.serialize(archive["vbd/params"], minimal=False)
        chebyshev.serialize(archive["vbd/chebyshev_params"], minimal=False)
