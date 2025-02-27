from .. import codegen as cg
import sympy as sp

if __name__ == "__main__":
    sp.init_printing()
    dims = 3
    a = sp.MatrixSymbol("axi", dims, 1)
    b = sp.MatrixSymbol("bxi", dims, 1)
    c = sp.MatrixSymbol("cxi", dims, 1)
    xk = sp.MatrixSymbol("xki", dims, 1)
    t, tk, tkm2 = sp.symbols("t tk tkm2", real=True)
    tbar = (t - tk) / (tk - tkm2)
    x = a * tbar**2 + b * tbar + c
    dx = x - xk
    P = sp.Poly(sum(dx[i] ** 2 for i in range(dims)), t)
    degree = sp.degree(P, gen=t)
    sigma = sp.MatrixSymbol("sigma", degree + 1, 1)
    for i in range(sigma.shape[0]):
        code = cg.codegen(
            P.coeffs()[i], lhs=sigma[i], csesymbol="z", scalar_type="GpuScalar"
        )
        print(f"{code}\n\n")
