from typing import Callable
import numpy as np

from ..minimizer import Minimizer


class PGD(Minimizer):
    barycentric = True
    label = "PGD"

    f: Callable[[np.ndarray], float]
    g: Callable[[np.ndarray], np.ndarray]
    eta: float

    def setup(self, f, g, **kwargs):
        self.f = f
        self.g = g
        self.eta = kwargs.get("eta",1e-3)
    
    def updateParams(self, **kwargs):
        self.eta = kwargs.get("eta", self.eta)
        
    def step(self, x):
        gk = self.g(x)
        xkp1 = x - self.eta * gk
        xkp1[0] = 0. if xkp1[0] < 0. else xkp1[0]
        xkp1[1] = 0. if xkp1[1] < 0. else xkp1[1]

        # projection might be wrong
        if xkp1[0] + xkp1[1] > 1:
            # xkp1 /= xkp1[0] + xkp1[1]
            alpha = np.array([1.,0.])
            v = np.array([-1.,1.])
            x = xkp1 - alpha
            t  = np.clip(np.dot(v,x)/np.dot(v,v),0.,1.)
            xkp1 = np.array(alpha) + t * v

        return xkp1