# type: ignore
from abc import ABC, abstractmethod
from pbatoolkit import pbat


class BaseSolver(ABC):
    _name: str

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def on_simulation_scenario_created(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        pass

    @abstractmethod
    def solve(
        self,
        fem: pbat.sim.dynamics.FemElastoDynamics,
        contact: pbat.sim.contact.MeshDynamics,
    ):
        pass

    @property
    def name(self) -> str:
        return self._name
