# type: ignore

import enum
import typing
from pbatoolkit import pbat
import polyscope as ps
import polyscope.imgui as imgui
from .params import ParameterObject
from .solvers import vbd, anderson, broyden, chebyshev, newton, base


class Solver:
    _solver: (
        vbd.VbdSolver
        | anderson.AndersonSolver
        | broyden.BroydenSolver
        | chebyshev.ChebyshevSolver
        | newton.NewtonSolver
    )
    _solvers: list[base.BaseSolver]

    def __init__(self):
        self._solvers = [
            vbd.VbdSolver(),
            anderson.AndersonSolver(),
            broyden.BroydenSolver(),
            chebyshev.ChebyshevSolver(),
            newton.NewtonSolver(),
        ]
        self._solver = self._solvers[0]

    def draw(self):
        imgui.PushID("Solver")
        selected_idx = self._solvers.index(self._solver)
        _, selected_idx = imgui.Combo(
            "Solver Type",
            selected_idx,
            [st.name for st in self._solvers],
        )
        self._solver = self._solvers[selected_idx]
        self._solver.draw()
        imgui.PopID()

    def on_simulation_scenario_created(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        for solver in self._solvers:
            solver.on_simulation_scenario_created(fem, contact)

    def step(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
        init: pbat.sim.dynamics.EFemElastoDynamicsTimeStepInitialization,
        archive: pbat.io.Archive | None = None,
    ):
        self._solver.integrate(fem, contact, init, archive)

    def set_visible(self, visible: bool):
        pass
