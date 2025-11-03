from typing import Callable
import numpy as np

from ..minimizer import Minimizer


class SR1(Minimizer):
    barycentric = True
    label = "SR1"

    f: Callable[[np.ndarray], float]
    g: Callable[[np.ndarray], np.ndarray]
    triangle: np.ndarray
    fk: float
    gk: np.ndarray
    sigmaB: float
    sigmaR: float
    eta: float
    Bk: np.ndarray
    Rk: float
    r: float
    trlo: float
    trhi: float
    trbound: float
    trgrow: float
    trshrink: float
    eps: float
    elen: list[float]

    
    def setup(self, f, g, **kwargs):
        self.f = f
        self.g = g

        self.fk = 0.0
        self.gk = np.zeros(2)

        self.triangle = kwargs['triangle']

        self.sigmaB = kwargs.get('sigmaB',1e-1)
        self.sigmaR = kwargs.get('sigmaR',1e-1)
        self.eta = kwargs.get('eta',1e-3)
        self.r = kwargs.get('r',1e-8)
        self.trlo = kwargs.get('trlo',0.1)
        self.trhi = kwargs.get('trhi',0.75)
        self.trbound = kwargs.get('trbound',0.8)
        self.trgrow = kwargs.get('trgrow',2.0)
        self.trshrink = kwargs.get('trshrink',0.5)

        A, B, C = self.triangle[:, 0], self.triangle[:, 1], self.triangle[:, 2]
        self.elen = [
                np.linalg.norm(A - B),
                np.linalg.norm(A - C),
                np.linalg.norm(B - C),
            ]

        self.Bk = np.eye(2) * self.sigmaB * max(self.elen)
        self.Rk = self.sigmaR * max(self.elen)
        self.eps = 1e-4
    
    def updateParams(self, **kwargs):
        pass

    def step(self, xk):
        self.fk = self.f(xk)
        self.gk = self.g(xk)
        xkp1 = self.solve_quadratic_in_reference_triangle_2d(xk, self.gk, self.Bk)
        sk = xkp1 - xk
        # NOTE: Uncomment to use 2-norm, i.e. ball trust region
        # if np.dot(sk, sk) > Rk * Rk:
        #     sk = sk * Rk / np.linalg.norm(sk)
        # Use max-norm, i.e. box trust region
        lensk = np.linalg.norm(sk, np.inf)
        # TODO: Vectorize this branch
        if lensk > self.Rk:
            sk = sk * self.Rk / lensk
            xkp1 = xk + sk
            lensk = np.linalg.norm(sk, np.inf)
        gkp1 = self.g(xkp1)
        yk = gkp1 - self.gk
        fkp1 = self.f(xkp1)
        ared = self.fk - fkp1
        Bksk = self.Bk @ sk
        skTBksk = sk.T @ Bksk
        mkp1 = self.gk.T @ sk + 0.5 * skTBksk
        pred = -mkp1
        rho = ared / (pred + self.eps)
        Rkp1 = self.Rk
        # TODO: Vectorize these branches
        if rho > self.trhi and lensk >= self.trbound * self.Rk:
            Rkp1 = self.trgrow * self.Rk
        elif rho < self.trlo:
            Rkp1 = self.trshrink * self.Rk
        vk = yk - Bksk
        den = np.dot(vk, sk)
        # stable = den**2 >= r * np.dot(sk, sk) * np.dot(vk, vk)
        # Bkp1 = Bk + np.outer(vk, vk) / den if stable else Bk
        # Update inverse hessian estimate and keep positive definite
        Bkp1 = self.Bk
        skTyk = np.dot(sk, yk)
        # NOTE: For a GPU implementation, we probably also want to
        # vectorize these branches
        if skTyk > skTBksk:
            Bkp1 = self.Bk + np.outer(vk, vk) / den
        if rho <= self.eta:
            xkp1 = xk
            fkp1 = self.fk
            gkp1 = self.gk
        
        if np.linalg.norm(xk - xkp1) > 0.0:
            self.fk, self.gk, self.Bk, self.Rk = fkp1, gkp1, Bkp1, Rkp1
        return xkp1


    def solve_quadratic_in_reference_triangle_2d(self, xk, gk, Bk) -> np.ndarray:
        xstar = xk - np.linalg.solve(Bk, gk)
        feasible = (xstar >= 0).all() and (xstar <= 1).all() and (xstar.sum() <= 1)
        if not feasible:
            # Derivation and CSE by-hand for the quadratic
            # f(t) = 0.5 (x0 + t dx - xk)^T Bk (x0 + t dx - xk) + gk^T (x0 + t dx - xk)
            # where x = x0 + t dx is constrained to one of the triangle edges.
            # Edge 1: x0 = [0,0], dx = [0,1]
            # Edge 2: x0 = [0,0], dx = [1,0]
            # Edge 3: x0 = [0,1], dx = [1,-1]
            gkTxk = gk.T @ xk
            Bkxk = Bk @ xk
            xkTBkxk = xk.T @ Bkxk
            a1 = -gkTxk + 0.5 * xkTBkxk
            b1 = gk[1] - Bkxk[1]
            c1 = Bk[1, 1]
            a2 = a1
            b2 = gk[0] - Bkxk[0]
            c2 = Bk[0, 0]
            xk0 = np.array([-xk[0], 1 - xk[1]])
            a3 = gk.T @ xk0 + 0.5 * xk0.T @ Bk @ xk0
            b3 = (gk[0] - gk[1]) + (Bk[0, 1] - Bk[1, 1]) - (Bkxk[0] - Bkxk[1])
            c3 = Bk[0, 0] - 2 * Bk[0, 1] + Bk[1, 1]
            # Minimize quadratic a_i + b_i t + 1/2 c_i t^2 assuming c_i > 0
            tmin1 = min(max(-b1 / c1, 0.0), 1.0)
            tmin2 = min(max(-b2 / c2, 0.0), 1.0)
            tmin3 = min(max(-b3 / c3, 0.0), 1.0)
            fmins = [
                a1 + b1 * tmin1 + 0.5 * c1 * tmin1**2,
                a2 + b2 * tmin2 + 0.5 * c2 * tmin2**2,
                a3 + b3 * tmin3 + 0.5 * c3 * tmin3**2,
            ]
            # Vectorized argmin
            argmin = np.array(
                [
                    fmins[0] <= fmins[1] and fmins[0] <= fmins[2],
                    fmins[1] <= fmins[0] and fmins[1] <= fmins[2],
                    fmins[2] <= fmins[0] and fmins[2] <= fmins[1],
                ]
            )
            xstars = np.array(
                [
                    [0.0, tmin1],
                    [tmin2, 0.0],
                    [tmin3, 1.0 - tmin3],
                ]
            )
            xstar = xstars.T @ argmin / argmin.sum()
            # Non-vectorized argmin
            # imin = np.argmin(fmins)
            # if imin == 0:
            #     xstar = np.array([0.0, tmin1])
            # elif imin == 1:
            #     xstar = np.array([tmin2, 0.0])
            # else:
            #     xstar = np.array([tmin3, 1.0 - tmin3])
        return xstar