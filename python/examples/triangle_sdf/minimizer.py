from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

class Minimizer(ABC):
    label: str
    barycentric: bool

    @abstractmethod
    def setup(self, f:Callable[[np.ndarray], float], g: Callable[[np.ndarray], np.ndarray], **kwargs)-> None:
        pass

    @abstractmethod
    def updateParams(self, **kwargs)-> None:
        pass

    @abstractmethod
    def step(self, x: np.ndarray)-> np.ndarray:
        pass