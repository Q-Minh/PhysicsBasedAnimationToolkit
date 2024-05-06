from ..codegen import codegen
import sympy as sp


def IC(F):
    return (F.transpose() * F).trace()


def IIC(F):
    FTF = F.transpose() * F
    return (FTF.transpose() * FTF).trace()


def IIIC(F):
    return (F.transpose() * F).det()


def I1(S):
    return S.trace()


def I2(F):
    return (F.transpose() * F).trace()


def I3(F):
    return F.det()


def stvk(F, mu, llambda):
    E = (F.transpose() * F - sp.eye(F.shape[0])) / 2
    return mu*(E.transpose()*E).trace() + (llambda / 2) * E.trace()**2


def neohookean(F, mu, llambda):
    gamma = 1 + mu/llambda
    d = F.shape[0]
    return (mu/2) * (I2(F) - d) + (llambda / 2) * (I3(F) - gamma)**2


if __name__ == "__main__":
    din, dout = 3, 3
    ne = 4
    mu, llambda = sp.symbols("\\mu \\lambda", real=True)
    F = sp.Matrix(sp.MatrixSymbol("F", din, dout))
    S = sp.Matrix(sp.MatrixSymbol("S", din, dout))
    u = sp.Matrix(sp.MatrixSymbol("u", dout, ne))
    gradphi = sp.Matrix(sp.MatrixSymbol("GN", ne, din))

    
