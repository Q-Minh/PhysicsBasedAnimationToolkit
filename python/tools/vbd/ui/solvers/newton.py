# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .serialize import serialize_solver_iteration
from .base import BaseSolver
import typing
import numpy as np


class NewtonSolver(BaseSolver):
    _params: ParameterObject
    _linsol: pbat.sim.algorithm.newton.ELinearSolver
    _linsol_max_iters: int
    _linsol_tol: float

    def __init__(self):
        super().__init__("Newton")
        self._params = ParameterObject(
            pbat.sim.algorithm.newton.Params().with_optimizer(
                pbat.math.optimization.Newton(
                    line_search=pbat.math.optimization.BackTrackingLineSearch(),
                )
            ),
            {"newton": {"line_search": None}},
        )
        params: pbat.sim.algorithm.newton.Params = self._params.params
        self._linsol = params.linear_solver
        self._linsol_max_iters = 300
        self._linsol_tol = 1e-6

    def draw(self):
        imgui.PushID(self._name)
        if imgui.TreeNode(f"{type(self._params.params).__name__}"):
            self._params.draw()
            if imgui.TreeNode("linear solver parameters"):
                max_iters_changed, self._linsol_max_iters = imgui.InputInt(
                    "Max Iters", self._linsol_max_iters, 1, 10
                )
                tol_changed, self._linsol_tol = imgui.InputFloat(
                    "Tolerance", self._linsol_tol, 1e-6, 1e-3, "%.6f"
                )
                params: pbat.sim.algorithm.newton.Params = self._params.params
                linsol_changed = self._linsol != params.linear_solver
                if max_iters_changed or tol_changed or linsol_changed:
                    self._linsol = params.linear_solver
                    params.with_linear_solver(
                        linear_solver=self._linsol,
                        max_iters=self._linsol_max_iters,
                        tol=self._linsol_tol,
                    ).construct()
                imgui.TreePop()
            imgui.TreePop()
        imgui.PopID()

    def on_simulation_scenario_created(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        pass

    def solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        callback: typing.Callable[None, None] | None = None,
    ):
        if callback is None:
            callback = lambda: None
        params: pbat.sim.algorithm.newton.Params = self._params.params
        newton: pbat.math.optimization.Newton = params.newton
        # pbat.sim.algorithm.newton.initialize_solve(fem, contact, params)
        # TODO: Do this conditionally only
        newton.k = 0
        newton.gk = np.zeros(fem.x.shape[0] * fem.x.shape[1])
        contact.ogc_state.bv = np.full_like(contact.ogc_state.bv, 1e10)
        callback()
        pbat.sim.algorithm.newton.prepare_next_iteration(fem, contact, params)
        while newton.k < newton.n_max_iters:
            if newton.gknorm2 < newton.gtol2:
                break
            if not pbat.sim.algorithm.newton.iterate(fem, contact, params):
                break
            callback()
            pbat.sim.algorithm.newton.prepare_next_iteration(fem, contact, params)
        fem.back_substitute_integrated_positions_into_velocities()

    def serialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.newton.Params = self._params.params
        params.serialize(archive)

    def deserialize(self, archive: pbat.io.Archive):
        params: pbat.sim.algorithm.newton.Params = self._params.params
        params.deserialize(archive)

    def serialize_problem(
        self,
        archive: pbat.io.Archive,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        fem.serialize(archive["fem"])
        contact.serialize(archive["contact"])
        params: pbat.sim.algorithm.newton.Params = self._params.params
        params.serialize(archive["newton/params"])
