import sympy as sp
from sympy.printing.python import PythonPrinter


def quadratic(x0, dx, B, g, t):
    x = x0 + t * dx
    f = 0.5 * x.T @ B @ x + g.T @ x
    f = sp.expand(f[0,0])
    f = sp.simplify(f)
    Q = sp.Poly(f, t)
    print("a={}".format(codegen.doprint(Q.all_coeffs()[0])))
    print("b={}".format(codegen.doprint(Q.all_coeffs()[1])))
    print("c={}".format(codegen.doprint(Q.all_coeffs()[2])))


if __name__ == "__main__":
    B = sp.MatrixSymbol("B", 2, 2)
    g = sp.MatrixSymbol("g", 2, 1)
    t = sp.symbols("t", real=True)
    codegen = PythonPrinter()
    # Plane x >= 0
    print("Plane x >= 0")
    x0 = sp.Matrix([0, 0])
    dx = sp.Matrix([0, 1])
    quadratic(x0, dx, B, g, t)
    # Plane y >= 0
    print("Plane y >= 0")
    x0 = sp.Matrix([0, 0])
    dx = sp.Matrix([1, 0])
    quadratic(x0, dx, B, g, t)
    # Plane x+y <= 1
    print("Plane x+y <= 1")
    x0 = sp.Matrix([0, 1])
    dx = sp.Matrix([1, -1])
    quadratic(x0, dx, B, g, t)
