# type: ignore
from abc import ABC, abstractmethod
from pbatoolkit import pbat
import typing


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
        callback: typing.Callable[None, None] | None = None,
    ):
        """Solve the simulation step.

        Args:
            fem (pbat.sim.dynamics.FemElastoDynamics): The FEM dynamics object.
            contact (pbat.sim.contact.MeshDynamics): The contact dynamics object.
            callback (typing.Callable[None, None] | None, optional): A callback function to be called on each iteration. Defaults to None.
        """
        pass

    @abstractmethod
    def serialize(self, archive: pbat.io.Archive):
        pass

    @abstractmethod
    def deserialize(self, archive: pbat.io.Archive):
        pass

    @property
    def name(self) -> str:
        return self._name
