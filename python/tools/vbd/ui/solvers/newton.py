# type: ignore
from pbatoolkit import pbat
from ..params import ParameterObject
import polyscope as ps
import polyscope.imgui as imgui
from .serialize import serialize_solver_iteration
from .base import BaseSolver


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
        # TODO: Call the params.with_linear_solver function to construct the linear solver
        pass

    def integrate(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        init: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
        archive: pbat.io.Archive | None = None,
    ):
        fem.setup_time_integration_optimization(initialization_strategy=init)
        grp = (
            archive["pbat.sim.algorithm.newton.Integrate"]
            if archive is not None
            else None
        )
        if grp is not None:
            fem.serialize(grp)
        self._solve(fem, contact, archive=grp)
        fem.back_substitute_integrated_positions_into_velocities()
        fem.step()

    def _solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        archive: pbat.io.Archive | None = None,
    ):
        grp = (
            archive["pbat.sim.algorithm.newton.Solve"] if archive is not None else None
        )
        params: pbat.sim.algorithm.newton.Params = self._params.params
        newton: pbat.math.optimization.Newton = params.newton
        pbat.sim.algorithm.newton.initialize_solve(fem, params)
        while newton.k < newton.n_max_iters:
            if grp is not None:
                serialize_solver_iteration(
                    fem, newton.k, newton, grp, f_serialize_more=self._serialize_more
                )
            if newton.gknorm2 < newton.gtol2:
                break
            if not pbat.sim.algorithm.newton.iterate(fem, params):
                break
            pbat.sim.algorithm.newton.prepare_next_iteration(fem, params)
        fem.back_substitute_integrated_positions_into_velocities()
        if grp is not None:
            serialize_solver_iteration(
                fem,
                newton.k,
                newton,
                grp,
                f_serialize_more=self._serialize_more,
                post_solve=True,
            )

    def _serialize_more(self, arc: pbat.io.Archive):
        newton: pbat.math.optimization.Newton = self._params.params
        newton.serialize(arc)
