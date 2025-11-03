from typing import Callable
import numpy as np

from ..minimizer import Minimizer


class FW(Minimizer):
    barycentric = False
    label = "Frank-Wolfe"

    f: Callable[[np.ndarray], float]
    g: Callable[[np.ndarray], np.ndarray]
    vertices: np.ndarray
    t: int

    def setup(self, f, g, **kwargs):
        self.f = f
        self.g = g
        self.vertices = kwargs["vertices"]
        self.t = 0
    
    def updateParams(self, **kwargs):
        self.vertices = kwargs["vertices"]
        
    def step(self, x):
        grad = self.g(x)

        d = self.vertices.T @ grad
        # min_v = sorted(zip(vertices,d),key=lambda el: el[1])[0][0]
        min_v = self.vertices[:,np.argmin(d)]
        # print("min_v",min_v)
        # print("vertices",self.vertices)
        alpha = 2./(self.t + 2)
        xkp1 = x + alpha * (min_v - x)

        self.t =self.t + 1
        return xkp1