# type: ignore

import enum
import typing
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
from .draw_parameter_object import ParameterObject


class SolverType(enum.Enum):
    VBD = "VBD"
    ANDERSON = "Anderson"
    BROYDEN = "Broyden"
    CHEBYSHEV = "Chebyshev"
    NEWTON = "Newton"


class Solver:
    _solver: SolverType
    _solvers: dict[SolverType, typing.Any]
    _solver_params: dict[SolverType, ParameterObject]

    def __init__(self):
        self._solver = SolverType.VBD
        self._solver_params = {
            SolverType.VBD: ParameterObject(pbat.sim.algorithm.vbd.Params()),
            SolverType.ANDERSON: ParameterObject(
                pbat.sim.algorithm.vbd.AndersonParams()
            ),
            SolverType.BROYDEN: ParameterObject(pbat.sim.algorithm.vbd.BroydenParams()),
            SolverType.CHEBYSHEV: ParameterObject(
                pbat.sim.algorithm.vbd.ChebyshevParams()
            ),
            SolverType.NEWTON: ParameterObject(
                pbat.sim.algorithm.newton.Params().with_optimizer(
                    pbat.math.optimization.Newton(
                        line_search=pbat.math.optimization.BackTrackingLineSearch(),
                    )
                ),
                {"newton": {"line_search": None}},
            ),
        }

    def draw(self):
        imgui.PushID("Solver")
        solver_types = list(SolverType)
        selected_idx = solver_types.index(self._solver)
        _, selected_idx = imgui.Combo(
            "Solver Type",
            selected_idx,
            [st.name for st in solver_types],
        )
        self._solver = solver_types[selected_idx]
        param_obj = self._solver_params[self._solver]
        if imgui.TreeNode("Parameters"):
            imgui.PushID(selected_idx)
            param_obj.draw()
            imgui.PopID()
            imgui.TreePop()
        imgui.PopID()
