from .. import codegen as cg
import sympy as sp


def cross(A, B):
    return sp.Matrix(
        [
            A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0],
        ]
    )


def dot(A, B):
    return sum(A[i] * B[i] for i in range(A.shape[0]))


if __name__ == "__main__":
    sp.init_printing()
    dims = 3
    XT = sp.MatrixSymbol("XT", dims, 1)
    AT = sp.MatrixSymbol("AT", dims, 1)
    BT = sp.MatrixSymbol("BT", dims, 1)
    CT = sp.MatrixSymbol("CT", dims, 1)
    X = sp.MatrixSymbol("X", dims, 1)
    A = sp.MatrixSymbol("A", dims, 1)
    B = sp.MatrixSymbol("B", dims, 1)
    C = sp.MatrixSymbol("C", dims, 1)
    t = sp.Symbol("t", real=True)

    x = XT + t * (X - XT)
    a = AT + t * (A - XT)
    b = BT + t * (B - XT)
    c = CT + t * (C - XT)
    n = cross(b - a, c - a)
    q = x - a
    f = dot(n, q)
    P = sp.Poly(f, t)
    degree = sp.degree(P, gen=t)
    sigma = sp.MatrixSymbol("sigma", degree + 1, 1)
    coeffs = sp.Matrix(list(reversed(P.coeffs())))
    print("Point-Triangle CCD univariate polynomial:\n")
    code = cg.codegen(coeffs, lhs=sigma, csesymbol="z", scalar_type="TScalar")
    print(f"{code}\n\n")

    P1T = sp.MatrixSymbol("P1T", dims, 1)
    Q1T = sp.MatrixSymbol("Q1T", dims, 1)
    P2T = sp.MatrixSymbol("P2T", dims, 1)
    Q2T = sp.MatrixSymbol("Q2T", dims, 1)
    P1 = sp.MatrixSymbol("P1", dims, 1)
    Q1 = sp.MatrixSymbol("Q1", dims, 1)
    P2 = sp.MatrixSymbol("P2", dims, 1)
    Q2 = sp.MatrixSymbol("Q2", dims, 1)
    p1 = P1T + t * (P1 - P1T)
    q1 = Q1T + t * (Q1 - Q1T)
    p2 = P2T + t * (P2 - P2T)
    q2 = Q2T + t * (Q2 - P2T)
    n = cross(q1 - p1, q2 - p2)
    q = p2 - p1
    f = dot(n, q)
    P = sp.Poly(f, t)
    degree = sp.degree(P, gen=t)
    sigma = sp.MatrixSymbol("sigma", degree + 1, 1)
    coeffs = sp.Matrix(list(reversed(P.coeffs())))
    print("Edge-Edge CCD univariate polynomial:\n")
    code = cg.codegen(coeffs, lhs=sigma, csesymbol="z", scalar_type="TScalar")
    print(f"{code}\n\n")
