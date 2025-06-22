from ..codegen import cg
import sympy as sp

if __name__ == "__main__":
    d = 3
    ks, l0 = sp.symbols("k_s l_0", real=True)
    x = sp.Matrix(sp.MatrixSymbol("x", 2*d, 1))
    xi = x[0:d,0]
    xj = x[d:,0]
    xij = xi - xj
    l = sp.sqrt(xij.T @ xij)[0,0]
    E = sp.Rational(1, 2) * ks * (l - l0)**2
    gradE = sp.derive_by_array(E, x)
    hessE = sp.derive_by_array(gradE, x)[:,0,:,0]
    gradEcode = cg.codegen(gradE, lhs=sp.MatrixSymbol("gradE", *gradE.shape))
    hessEcode = cg.codegen(hessE.transpose(), lhs=sp.MatrixSymbol("hessE", x.shape[0], x.shape[0]))
    print(f"gradE code:\n{gradEcode}\n\n")
    print(f"hessE code:\n{hessEcode}\n\n")