import sympy as sp
from .. import codegen as cg
from enum import StrEnum


class Quadrature(StrEnum):
    SIMPLEX = "SymmetricSimplexPolynomialQuadratureRule"
    GAUSS = "GaussLegendreQuadrature"


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


def vertices_from_nodes(x):
    vertices = []
    for i in range(len(x)):
        is_vertex_coordinate = [
            (x[i][d] == 0 or x[i][d] == 1) for d in range(len(x[i]))]
        is_vertex = is_vertex_coordinate.count(False) == 0
        if is_vertex:
            vertices.append(i)
    return vertices


class Element:
    """
    Holds information necessary to code generate an FEM element type in C++
    """

    def __init__(self, X: sp.Matrix,
                 x: list,
                 N: sp.Matrix,
                 gradN: sp.Matrix,
                 quad: Quadrature):
        """Standard constructor

        Args:
            X (sp.Matrix): Material space variables
            x (list): Lagrange nodal coordinates
            N (sp.Matrix): Nodal shape functions
            gradN (sp.Matrix): Nodal shape function gradients (i.e. the Jacobian)
            quad (Quadrature): Quadrature scheme to use for this element
            vertices (list): Indices into nodes x revealing vertices of the element's simplex
        """
        self.X = X
        self.x = x
        self.N = N
        self.gradN = gradN
        self.quad = quad
        self.vertices = vertices_from_nodes(x)


def tetrahedron(p=1):
    """Lagrange tetrahedron (i+j+k) <= p, where p is the polynomial order 
    of the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        Element: 
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
    quad = Quadrature.SIMPLEX

    return Element(X, x, N, gradN, quad)


def hexahedron(p=1):
    """Lagrange hexahedron max(i,j,k) <= p, where p is the polynomial order of 
    any monomial in the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        Element: 
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
    quad = Quadrature.GAUSS
    return Element(X, x, N, gradN, quad)


def triangle(p=1):
    """Lagrange triangle (i+j) <= p, where p is the polynomial order of the 
    element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        Element: 
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
    quad = Quadrature.SIMPLEX
    return Element(X, x, N, gradN, quad)


def quadrilateral(p=1):
    """Lagrange quadrilateral max(i,j) <= p, where p is the polynomial order 
    of any monomial in the element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        Element: 
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
    quad = Quadrature.GAUSS
    return Element(X, x, N, gradN, quad)


def line(p=1):
    """Lagrange line segment i <= p, where p is the polynomial order of the 
    element's function space

    Args:
        p (int, optional): Polynomial order. Defaults to 1.

    Returns:
        Element: 
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
    # Could also be simplex, doesn't matter for 1D case
    quad = Quadrature.GAUSS
    return Element(X, x, N, gradN, quad)


def codegen(felement, p: int, element_name: str):
    """Generate C++ header file containing implementations of the given element up to order p.

    Args:
        felement Function: Function handle to generate the element shape functions.
        p (int): Polynomial order
        element (str): Textual name of element
    """

    header = f"""
/**
 * @file {element_name}.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief {element_name} finite element
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef PBAT_FEM_{element_name.upper()}_H
#define PBAT_FEM_{element_name.upper()}_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "QuadratureRules.h"

#include <array>

namespace pbat {{
namespace fem {{

namespace detail {{

template <int Order>
struct {element_name};

}} // namespace detail

/**
 * @brief {element_name} finite element
 * 
 * Satisfies concept CElement  
 *
 * @tparam Order Polynomial order
 */
template <int Order>
using {element_name} = typename detail::{element_name}<Order>;

namespace detail {{
"""

    footer = f"""
}} // namespace detail
}} // namespace fem
}} // namespace pbat

#endif // PBAT_FEM_{element_name.upper()}_H
"""

    with open("{}.h".format(element_name), 'w', encoding="utf-8") as file:
        file.write(header)
        for order in range(1, p+1):
            element = felement(
                order)
            X, x, N, gradN, quad = element.X, element.x, element.N, element.gradN, element.quad
            gradNT = gradN.transpose()
            dims = X.shape[0]
            nodes = len(x)
            coordinates = ",".join(
                [str(xi[d]*order) for xi in x for d in range(dims)])
            codeN = cg.tabulate(cg.codegen(N, lhs=sp.MatrixSymbol(
                "Nm", *N.shape), scalar_type="auto"), spaces=8)
            codeGN = cg.tabulate(cg.codegen(gradNT, lhs=sp.MatrixSymbol(
                "GNp", *gradNT.shape), scalar_type="auto"), spaces=8)
            vertices = ",".join([str(v)
                                for v in element.vertices])

            symX = list(X.atoms(sp.MatrixSymbol))[0]
            has_constant_jacobian = symX not in gradN.atoms(
                sp.MatrixSymbol)
            Jconst = str(has_constant_jacobian).lower()
            template_specialization = f"""
template <>
struct {element_name}<{order}>
{{
    using AffineBaseType = {element_name}<1>;

    static int constexpr kOrder = {order};
    static int constexpr kDims  = {dims};
    static int constexpr kNodes = {nodes};
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {{{coordinates}}}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {{{vertices}}}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = {Jconst};

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::{quad}<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {{
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
{codeN}
        return Nm;
    }}

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {{
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
{codeGN}
        return GNm;
    }}
}};
"""
            file.write(template_specialization)

        file.write(
            footer)


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
