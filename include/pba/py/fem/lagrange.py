import sympy as sp

def lagrange_shape_functions(X, x, p):
    P = sp.Matrix([[ pi.subs([(X[d], x[i][d]) for d in range(len(x[i]))]) for pi in p] for i in range(len(x))])
    W = P.inv()
    N = W.transpose() * sp.Matrix(p)
    N = N.applyfunc(lambda ni: sp.factor(ni))
    return N

# Lagrange tetrahedron (i+j+k) <= p, where p is the polynomial order of the element's function space
def tetrahedron(p=1):
    X = sp.Matrix(sp.MatrixSymbol("X", 3, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(j, p), sp.Rational(k, p)]
        for k in range(p + 1)
        for j in range(p + 1)
        for i in range(p + 1)
        if (i + j + k) <= p
    ]
    monomials = [
        X[0] ** i * X[1] ** j * X[2] ** k
        for k in range(p + 1)
        for j in range(p + 1)
        for i in range(p + 1)
        if (i + j + k) <= p
    ]
    N = lagrange_shape_functions(X, x, monomials)
    gradN = N.jacobian(X)
    return (X, x, N, gradN)

def hexahedron(p=1):
    # Lagrange hexahedron max(i,j,k) <= p, where p is the polynomial order of the any monomial in the element's function space
    X = sp.Matrix(sp.MatrixSymbol("X", 3, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(j, p), sp.Rational(k, p)]
        for k in range(p + 1)
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    monomials = [
        X[0] ** i * X[1] ** j * X[2] ** k
        for k in range(p + 1)
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    N = lagrange_shape_functions(X, x, monomials)
    gradN = N.jacobian(X)
    return (X, x, N, gradN)

# Lagrange triangle (i+j) <= p, where p is the polynomial order of the element's function space
def triangle(p=1):
    X = sp.Matrix(sp.MatrixSymbol("X", 2, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(j, p)]
        for j in range(p + 1)
        for i in range(p + 1)
        if (i + j) <= p
    ]
    monomials = [
        X[0] ** i * X[1] ** j
        for j in range(p + 1)
        for i in range(p + 1)
        if (i + j) <= p
    ]
    N = lagrange_shape_functions(X, x, monomials)
    gradN = N.jacobian(X)
    return (X, x, N, gradN)

# Lagrange quadrilateral max(i,j) <= p, where p is the polynomial order of any monomial in the element's function space
def quadrilateral(p=1):
    X = sp.Matrix(sp.MatrixSymbol("X", 2, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(j, p)]
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    monomials = [
        X[0] ** i * X[1] ** j
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    N = lagrange_shape_functions(X, x, monomials)
    gradN = N.jacobian(X)
    return (X, x, N, gradN)

# Lagrange line segment i <= p, where p is the polynomial order of the element's function space
def line(p=1):
    X = sp.Matrix(sp.MatrixSymbol("X", 1, 1))
    x = [
        [sp.Rational(i, p)]
        for i in range(p + 1)
        if (i) <= p
    ]
    monomials = [
        X[0] ** i
        for i in range(p + 1)
        if (i) <= p
    ]
    N = lagrange_shape_functions(X, x, monomials)
    gradN = N.jacobian(X)
    return (X, x, N, gradN)

if __name__ == "__main__":
    pass