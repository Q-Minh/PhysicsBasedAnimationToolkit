import sympy as sp
from .. import codegen as cg


def lagrange_shape_functions(X, x, p):
    """Computes interpolating polynomials for the data points x using polynomial basis p

    Args:
        X : Dims x 1 Symbolic independent variables of shape functions
        x : N x Dims data points to interpolate
        p : N x 1 basis polynomials

    Returns:
        sp.Matrix : N x 1 matrix of shape functions for data points x
    """
    P = sp.Matrix([[pi.subs([(X[d], x[i][d]) for d in range(len(x[i]))])
                  for pi in p] for i in range(len(x))])
    W = P.inv()
    N = W.transpose() * sp.Matrix(p)
    N = N.applyfunc(
        lambda ni: sp.factor(ni))
    return N


def tetrahedron(p=1):
    """Lagrange tetrahedron (i+j+k) <= p, where p is the polynomial order 
    of the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        tuple: The 5-tuple (X, x, N, gradN, V) of variables, data points, shape 
        functions, shape function derivatives, and vertices, respectively.
    """
    dims = 3
    X = sp.Matrix(
        sp.MatrixSymbol("X", dims, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(
            j, p), sp.Rational(k, p)]
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
    N = lagrange_shape_functions(
        X, x, monomials)
    gradN = N.jacobian(X)
    V = [x[i] for i in range(len(x)) if [
        x[i][d] for d in range(dims)].count(1) <= 1]
    return (X, x, N, gradN, V)


def hexahedron(p=1):
    """Lagrange hexahedron max(i,j,k) <= p, where p is the polynomial order of 
    any monomial in the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        tuple: The 5-tuple (X, x, N, gradN, V) of variables, data points, shape 
        functions, shape function derivatives, and vertices, respectively.
    """
    dims = 3
    X = sp.Matrix(
        sp.MatrixSymbol("X", dims, 1))
    x = [
        [sp.Rational(i, p), sp.Rational(
            j, p), sp.Rational(k, p)]
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
    N = lagrange_shape_functions(
        X, x, monomials)
    gradN = N.jacobian(X)
    V = [x[i] for i in range(len(x)) if all(
        [x[i][d] == 0 or x[i][d] == 1 for d in range(dims)])]
    return (X, x, N, gradN, V)


def triangle(p=1):
    """Lagrange triangle (i+j) <= p, where p is the polynomial order of the 
    element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        tuple: The 5-tuple (X, x, N, gradN, V) of variables, data points, shape 
        functions, shape function derivatives, and vertices, respectively.
    """
    dims = 2
    X = sp.Matrix(
        sp.MatrixSymbol("X", dims, 1))
    x = [
        [sp.Rational(i, p),
         sp.Rational(j, p)]
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
    N = lagrange_shape_functions(
        X, x, monomials)
    gradN = N.jacobian(X)
    V = [x[i] for i in range(len(x)) if [
        x[i][d] for d in range(dims)].count(1) <= 1]
    return (X, x, N, gradN, V)


def quadrilateral(p=1):
    """Lagrange quadrilateral max(i,j) <= p, where p is the polynomial order 
    of any monomial in the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        tuple: The 5-tuple (X, x, N, gradN, V) of variables, data points, shape 
        functions, shape function derivatives, and vertices, respectively.
    """
    dims = 2
    X = sp.Matrix(
        sp.MatrixSymbol("X", dims, 1))
    x = [
        [sp.Rational(i, p),
         sp.Rational(j, p)]
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    monomials = [
        X[0] ** i * X[1] ** j
        for j in range(p + 1)
        for i in range(p + 1)
    ]
    N = lagrange_shape_functions(
        X, x, monomials)
    gradN = N.jacobian(X)
    V = [x[i] for i in range(len(x)) if all(
        [x[i][d] == 0 or x[i][d] == 1 for d in range(dims)])]
    return (X, x, N, gradN, V)


def line(p=1):
    """Lagrange line segment i <= p, where p is the polynomial order of the 
    element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        tuple: The 5-tuple (X, x, N, gradN, V) of variables, data points, shape 
        functions, shape function derivatives, and vertices, respectively.
    """
    dims = 1
    X = sp.Matrix(
        sp.MatrixSymbol("X", dims, 1))
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
    N = lagrange_shape_functions(
        X, x, monomials)
    gradN = N.jacobian(X)
    V = [[0], [1]]
    return (X, x, N, gradN, V)


def codegen(felement, p: int, element: str):
    """Generate C++ header file containing implementations of the given element up to order p.

    Args:
        felement Function: Function handle to generate the element shape functions.
        p (int): Polynomial order
        element (str): Textual name of element
    """

    header = """
#ifndef PBA_CORE_FEM_{0}_H    
#define PBA_CORE_FEM_{0}_H

#include "pba/aliases.h"

#include <array>

namespace pba {{
namespace fem {{
    
template <int Order>
struct {1};
"""

    footer = """
}} // fem
}} // pba

#endif // PBA_CORE_FEM_{0}_H
"""

    template_specialization = """
template <>
struct {0}<{1}>
{{
    using AffineBase = {0}<1>;
    
    static int constexpr Order = {1};
    static int constexpr Dims  = {2};
    static int constexpr Nodes = {3};
    static int constexpr Vertices = {4};
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {{{5}}}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {{
        Eigen::Vector<TScalar, Nodes> Nm;
{6}
        return Nm;
    }}
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {{
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
{7}
        return GNm;
    }}
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, Dims> Jacobian(
        [[maybe_unused]] Vector<Dims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {{
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == Nodes);
        auto constexpr DimsOut = Derived::RowsAtCompileTime;
        Matrix<DimsOut, Dims> const J = x * GradN(X);
        return J;
    }}
}};    
"""
    with open("{}.h".format(element), 'w', encoding="utf-8") as file:
        file.write(
            header.format(element.upper(), element))
        for order in range(1, p+1):
            X, x, N, gradN, V = felement(
                order)
            gradNT = gradN.transpose()
            dims = X.shape[0]
            nodes = len(x)
            vertices = len(V)
            coordinates = ",".join(
                [str(xi[d]*order) for xi in x for d in range(dims)])
            codeN = cg.tabulate(cg.codegen(N, lhs=sp.MatrixSymbol(
                "Nm", *N.shape), scalar_type="auto"), spaces=8)
            codeGN = cg.tabulate(cg.codegen(gradNT, lhs=sp.MatrixSymbol(
                "GNp", *gradNT.shape)), spaces=8)

            file.write(template_specialization.format(element,
                                                      order,
                                                      dims,
                                                      nodes,
                                                      vertices,
                                                      coordinates,
                                                      codeN,
                                                      codeGN))

        file.write(
            footer.format(element.upper()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Lagrange Finite Element Computer",
        description="""Computes Lagrange element nodes, shape functions and their 
        derivatives and generates C++ code for their computation.""",
    )
    parser.add_argument(
        "-p",
        "--polynomial-order",
        help="Polynomial order",
        type=int,
        dest="p",
        default=2)
    parser.add_argument(
        "-d",
        "--dry-run",
        help="Prints computed shape functions to stdout",
        type=bool,
        dest="dry_run")
    args = parser.parse_args()

    p = args.p
    dry_run = args.dry_run

    sp.init_printing()
    for element in [line, triangle, quadrilateral, tetrahedron, hexahedron]:
        name = "".join([c if i > 0 else c.upper()
                        for i, c in enumerate(element.__name__)])
        if dry_run:
            for order in range(1, p+1):
                X, x, N, gradN, V = element(
                    order)
                print("P{} {}".format(
                    order, name))
                sp.pretty_print(N)
        else:
            codegen(element, p, name)
