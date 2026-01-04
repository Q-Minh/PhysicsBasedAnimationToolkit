#ifndef PBAT_SIM_CONTACT_FRICTION_H
#define PBAT_SIM_CONTACT_FRICTION_H

#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Concepts.h"
#include "pbat/math/linalg/mini/Geometry.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Norm.h"

#include <cassert>
#include <limits>

namespace pbat::sim::contact {

/**
 * @brief Compute tangential basis given contact normal
 * 
 * @tparam TMatrixN Matrix type for normal
 * @tparam TScalar Scalar type
 * @param n Contact normal
 * @param eps Epsilon for colinearity check
 * @return math::linalg::mini::SMatrix<TScalar, TMatrixN::kRows, 2> 
 */
template <math::linalg::mini::CMatrix TMatrixN, class TScalar = typename TMatrixN::ScalarType>
PBAT_HOST_DEVICE auto
TangentialBasis(TMatrixN const& n, TScalar eps = std::numeric_limits<TScalar>::epsilon())
    -> math::linalg::mini::SMatrix<TScalar, TMatrixN::kRows, 2>
{
    using namespace math::linalg::mini;
    static_assert(TMatrixN::kRows >= 2, "n must have at least 2 rows.");
    auto constexpr kDims = TMatrixN::kRows;
    using namespace std;
    bool bIsColinearWithX = abs(n(0)) >= TScalar(1) - eps;
    // NOTE: We vectorize the following code for `t = cross(n, e)`
    // if (bIsColinearWithX)
    // {
    //     // e = (0,1,0)
    //     t(0) = n(1)*0 - n(2)*1;
    //     t(1) = n(2)*0 - n(0)*0;
    //     t(2) = n(0)*1 - n(1)*0;
    // }
    // else
    // {
    //     // e = (1,0,0)
    //     t(0) = n(1)*0 - n(2)*0;
    //     t(1) = n(2)*1 - n(0)*0;
    //     t(2) = n(0)*0 - n(1)*1;
    // }
    SMatrix<TScalar, kDims, 2> T;
    auto t = T.Col(0);
    t(0)   = (bIsColinearWithX) * (-n(2));
    t(1)   = (not bIsColinearWithX) * (n(2));
    t(2)   = (bIsColinearWithX) * (n(0)) + (not bIsColinearWithX) * (-n(1));
    t /= Norm(t);
    T.Col(1) = Cross(n, t);
    return T;
}

/**
 * @brief Compute tangential basis for point-point contact
 *
 * @tparam TMatrixX Matrix type for first point
 * @tparam TMatrixY Matrix type for second point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` first point
 * @param y `3 x 1` second point
 * @param eps Epsilon for colinearity check
 * @return `3 x 2` tangential basis matrix
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointPointTangentialBasis(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar eps = std::numeric_limits<TScalar>::epsilon())
    -> math::linalg::mini::SMatrix<TScalar, TMatrixX::kRows, 2>
{
    using namespace math::linalg::mini;
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    auto constexpr kDims         = TMatrixX::kRows;
    SVector<TScalar, kDims> xrel = x - y;
    TScalar xreln                = Norm(xrel);
    assert(xreln > TScalar(0));
    xrel /= xreln;
    return TangentialBasis(xrel, eps);
}

/**
 * @brief Compute linear tangential operator \f$ \mathbf{T} \f$ for point-point contact s.t.
 * tangential displacements are \f$ \mathbf{T} \begin{bmatrix}\mathbf{x} \\ \mathbf{y}\end{bmatrix}
 * \f$
 *
 * @tparam TMatrixX Matrix type for first point
 * @tparam TMatrixY Matrix type for second point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` first point
 * @param y `3 x 1` second point
 * @return `2*|# dims| x 2` tangential operator
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointPointLinearTangentialOperator(TMatrixX const& x, TMatrixY const& y)
    -> math::linalg::mini::SMatrix<TScalar, TMatrixX::kRows + TMatrixY::kRows, 2>
{
    using namespace math::linalg::mini;
    SMatrix<TScalar, TMatrixX::kRows + TMatrixY::kRows, 2> T;
    auto constexpr kDims                 = TMatrixX::kRows;
    T.template Slice<kDims, 2>(0, 0)     = PointPointTangentialBasis(x, y);
    T.template Slice<kDims, 2>(kDims, 0) = -T.template Slice<kDims, 2>(0, 0);
    return T;
}

/**
 * @brief Compute the (i-th) block of the linear tangential operator \f$ \mathbf{T} \f$ returned by
 * function `PointPointLinearTangentialOperator`.
 *
 * @tparam TMatrixX Matrix type for first point
 * @tparam TMatrixY Matrix type for second point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` first point
 * @param y `3 x 1` second point
 * @param i Block row
 * @return `|# dims| x 2` tangential operator block
 * @pre `i` must be either `0` (for first point) or `1` (for second point)
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto
PointPointLinearTangentialOperatorBlock(TMatrixX const& x, TMatrixY const& y, int i)
    -> math::linalg::mini::SMatrix<TScalar, TMatrixX::kRows, 2>
{
    auto B  = PointPointTangentialBasis(x, y);
    int sgn = (i == 0) * 1 + (i == 1) * -1;
    return sgn * B;
}

/**
 * @brief Compute tangential basis for point-edge contact
 *
 * @tparam TMatrixX Matrix type for point
 * @tparam TMatrixP Matrix type for first edge point
 * @tparam TMatrixQ Matrix type for second edge point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` point
 * @param p `3 x 1` first edge point
 * @param q `3 x 1` second edge point
 * @return `3 x 2` tangential basis matrix
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixP,
    math::linalg::mini::CMatrix TMatrixQ,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto
PointEdgeTangentialBasis(TMatrixX const& x, TMatrixP const& p, TMatrixQ const& q)
    -> math::linalg::mini::SMatrix<TScalar, TMatrixX::kRows, 2>
{
    using namespace math::linalg::mini;
    static_assert(TMatrixX::kRows == TMatrixP::kRows, "x and p must have the same number of rows.");
    static_assert(TMatrixX::kRows == TMatrixQ::kRows, "x and q must have the same number of rows.");
    auto constexpr kDims = TMatrixX::kRows;
    SMatrix<TScalar, kDims, 2> T;
    auto t1     = T.Col(0);
    auto t2     = T.Col(1);
    t1          = q - p;
    TScalar t1n = Norm(t1);
    assert(t1n > TScalar(0));
    t1 /= t1n;
    t2          = x - p;
    t2          = Cross(t1, t2);
    TScalar t2n = Norm(t2);
    assert(t2n > TScalar(0));
    t2 /= t2n;
    return T;
}

/**
 * @brief Compute linear tangential operator \f$ \mathbf{T} \f$ for point-edge contact s.t.
 * tangential displacements are \f$ \mathbf{T} \begin{bmatrix}\mathbf{x} \\ \mathbf{p}
 * \\ \mathbf{q}\end{bmatrix} \f$
 *
 * @tparam TMatrixX Matrix type for point
 * @tparam TMatrixP Matrix type for first edge point
 * @tparam TMatrixQ Matrix type for second edge point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` point
 * @param p `3 x 1` first edge point
 * @param q `3 x 1` second edge point
 * @param u Closest point on edge `pq` to `x` in barycentric coordinates [0,1]
 * @return `3*|# dims| x 2` tangential operator
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixP,
    math::linalg::mini::CMatrix TMatrixQ,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointEdgeLinearTangentialOperator(
    TMatrixX const& x,
    TMatrixP const& p,
    TMatrixQ const& q,
    TScalar u)
{
    using namespace math::linalg::mini;
    static_assert(TMatrixX::kRows == TMatrixP::kRows, "x and p must have the same number of rows.");
    static_assert(TMatrixX::kRows == TMatrixQ::kRows, "x and q must have the same number of rows.");
    assert(u >= TScalar(0) and u <= TScalar(1));
    auto constexpr kDims = TMatrixX::kRows;
    SMatrix<TScalar, kDims * 3, 2> T;
    auto B                                   = T.template Slice<kDims, 2>(0, 0);
    B                                        = PointEdgeTangentialBasis(x, p, q);
    T.template Slice<kDims, 2>(kDims, 0)     = (u - TScalar(1)) * B;
    T.template Slice<kDims, 2>(2 * kDims, 0) = -u * B;
    return T;
}

/**
 * @brief Compute the (i-th) block of the linear tangential operator \f$ \mathbf{T} \f$ returned by
 * function `PointEdgeLinearTangentialOperator`.
 *
 * @tparam TMatrixX Matrix type for point
 * @tparam TMatrixP Matrix type for first edge point
 * @tparam TMatrixQ Matrix type for second edge point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` point
 * @param p `3 x 1` first edge point
 * @param q `3 x 1` second edge point
 * @param u Closest point on edge `pq` to `x` in barycentric coordinates [0,1]
 * @param i Block row
 * @return `|# dims| x 2` tangential operator
 * @pre `i` must be either `0`, `1`, or `2`
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixP,
    math::linalg::mini::CMatrix TMatrixQ,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointEdgeLinearTangentialOperatorBlock(
    TMatrixX const& x,
    TMatrixP const& p,
    TMatrixQ const& q,
    TScalar u,
    int i)
{
    assert(i >= 0 and i <= 2);
    auto B    = PointEdgeTangentialBasis(x, p, q);
    TScalar k = (i == 0) * TScalar(1) + (i == 1) * (u - TScalar(1)) + (i == 2) * (-u);
    return k * B;
}

/**
 * @brief Compute linear tangential operator \f$ \mathbf{T} \f$ for point-edge contact s.t.
 * tangential displacements are \f$ \mathbf{T} \begin{bmatrix}\mathbf{x} \\ \mathbf{p}
 * \\ \mathbf{q}\end{bmatrix} \f$, using a pre-interpolated point on the edge.
 *
 * @tparam TMatrixX Matrix type for point
 * @tparam TMatrixY Matrix type for interpolated edge point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` point
 * @param y `3 x 1` interpolated point on edge, i.e., `y = (1-u)*p + u*q`
 * @param u Barycentric coordinate of `y` on edge `pq` in [0,1]
 * @param eps Epsilon for colinearity check
 * @return `3*|# dims| x 2` tangential operator
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointEdgeLinearTangentialOperator(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar u,
    TScalar eps = std::numeric_limits<TScalar>::epsilon())
{
    using namespace math::linalg::mini;
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    assert(u >= TScalar(0) and u <= TScalar(1));
    auto constexpr kDims = TMatrixX::kRows;
    SMatrix<TScalar, kDims * 3, 2> T;
    auto B                                   = T.template Slice<kDims, 2>(0, 0);
    B                                        = PointPointTangentialBasis(x, y, eps);
    T.template Slice<kDims, 2>(kDims, 0)     = (u - TScalar(1)) * B;
    T.template Slice<kDims, 2>(2 * kDims, 0) = -u * B;
    return T;
}

/**
 * @brief Compute the (i-th) block of the linear tangential operator \f$ \mathbf{T} \f$ returned by
 * function `PointEdgeLinearTangentialOperator`, using a pre-interpolated point on the edge.
 *
 * @tparam TMatrixX Matrix type for point
 * @tparam TMatrixY Matrix type for interpolated edge point
 * @tparam TScalar Scalar type
 * @param x `3 x 1` point
 * @param y `3 x 1` interpolated point on edge, i.e., `y = (1-u)*p + u*q`
 * @param u Barycentric coordinate of `y` on edge `pq` in [0,1]
 * @param i Block row
 * @param eps Epsilon for colinearity check
 * @return `|# dims| x 2` tangential operator
 * @pre `i` must be either `0`, `1`, or `2`
 */
template <
    math::linalg::mini::CMatrix TMatrixX,
    math::linalg::mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto PointEdgeLinearTangentialOperatorBlock(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar u,
    int i,
    TScalar eps = std::numeric_limits<TScalar>::epsilon())
{
    assert(i >= 0 and i <= 2);
    auto B    = PointPointTangentialBasis(x, y, eps);
    TScalar k = (i == 0) * TScalar(1) + (i == 1) * (u - TScalar(1)) + (i == 2) * (-u);
    return k * B;
}

/**
 * @brief Compute tangential basis for point-triangle contact
 *
 * @tparam TMatrixA Matrix type for first triangle point
 * @tparam TMatrixB Matrix type for second triangle point
 * @tparam TMatrixC Matrix type for third triangle point
 * @tparam TScalar Scalar type
 * @param a `3 x 1` first triangle point
 * @param b `3 x 1` second triangle point
 * @param c `3 x 1` third triangle point
 * @return `3 x 2` tangential basis matrix
 */
template <
    math::linalg::mini::CMatrix TMatrixA,
    math::linalg::mini::CMatrix TMatrixB,
    math::linalg::mini::CMatrix TMatrixC,
    class TScalar = typename TMatrixA::ScalarType>
PBAT_HOST_DEVICE auto
PointTriangleTangentialBasis(TMatrixA const& a, TMatrixB const& b, TMatrixC const& c)
    -> math::linalg::mini::SMatrix<TScalar, TMatrixA::kRows, 2>
{
    using namespace math::linalg::mini;
    static_assert(TMatrixA::kRows == TMatrixB::kRows, "a and b must have the same number of rows.");
    static_assert(TMatrixA::kRows == TMatrixC::kRows, "a and c must have the same number of rows.");
    auto constexpr kDims = TMatrixA::kRows;
    SMatrix<TScalar, kDims, 2> T;
    auto t1                   = T.Col(0);
    auto t2                   = T.Col(1);
    t1                        = b - a;
    SVector<TScalar, kDims> n = Cross(t1, c - a);
    TScalar t1n               = Norm(t1);
    assert(t1n > TScalar(0));
    TScalar nn = Norm(n);
    assert(nn > TScalar(0));
    t1 /= t1n;
    n /= nn;
    t2 = Cross(n, t1);
    return T;
}

/**
 * @brief Compute linear tangential operator \f$ \mathbf{T} \f$ for point-triangle contact s.t.
 * tangential displacements are \f$ \mathbf{T} \begin{bmatrix}\mathbf{x} \\ \mathbf{a} \\ \mathbf{b}
 * \\ \mathbf{c}\end{bmatrix} \f$ and the closest point on triangle `abc` to `x` is given in
 * barycentric coordinates by `uvw`.
 *
 * @tparam TMatrixA Matrix type for first triangle point
 * @tparam TMatrixB Matrix type for second triangle point
 * @tparam TMatrixC Matrix type for third triangle point
 * @tparam TMatrixUvw Matrix type for barycentric coordinates of closest point
 * @tparam TScalar Scalar type
 * @param a `3 x 1` first triangle point
 * @param b `3 x 1` second triangle point
 * @param c `3 x 1` third triangle point
 * @param uvw `3 x 1` barycentric coordinates of closest point
 * @return `4*|# dims| x 2` tangential operator
 */
template <
    math::linalg::mini::CMatrix TMatrixA,
    math::linalg::mini::CMatrix TMatrixB,
    math::linalg::mini::CMatrix TMatrixC,
    math::linalg::mini::CMatrix TMatrixUvw,
    class TScalar = typename TMatrixA::ScalarType>
PBAT_HOST_DEVICE auto PointTriangleLinearTangentialOperator(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixC const& c,
    TMatrixUvw const& uvw)
{
    using namespace math::linalg::mini;
    static_assert(TMatrixA::kRows == TMatrixB::kRows, "a and b must have the same number of rows.");
    static_assert(TMatrixA::kRows == TMatrixC::kRows, "a and c must have the same number of rows.");
    static_assert(TMatrixUvw::kRows == 3, "uvw must be 3 x 1.");
    assert(uvw(0) >= TScalar(0) and uvw(0) <= TScalar(1));
    assert(uvw(1) >= TScalar(0) and uvw(1) <= TScalar(1));
    assert(uvw(2) >= TScalar(0) and uvw(2) <= TScalar(1));
    auto constexpr kDims = TMatrixA::kRows;
    SMatrix<TScalar, 4 * kDims, 2> T;
    auto B                                   = T.template Slice<kDims, 2>(0, 0);
    B                                        = PointTriangleTangentialBasis(a, b, c);
    T.template Slice<kDims, 2>(kDims, 0)     = -uvw(0) * B;
    T.template Slice<kDims, 2>(2 * kDims, 0) = -uvw(1) * B;
    T.template Slice<kDims, 2>(3 * kDims, 0) = -uvw(2) * B;
    return T;
}

/**
 * @brief Compute the (i-th) block of the linear tangential operator \f$ \mathbf{T} \f$ returned by
 * function `PointTriangleLinearTangentialOperator`.
 *
 * @tparam TMatrixA Matrix type for first triangle point
 * @tparam TMatrixB Matrix type for second triangle point
 * @tparam TMatrixC Matrix type for third triangle point
 * @tparam TMatrixUvw Matrix type for barycentric coordinates of closest point
 * @tparam TScalar Scalar type
 * @param a `3 x 1` first triangle point
 * @param b `3 x 1` second triangle point
 * @param c `3 x 1` third triangle point
 * @param uvw `3 x 1` barycentric coordinates of closest point
 * @param i Block row
 * @return `|# dims| x 2` tangential operator
 * @pre `i` must be either `0`, `1`, `2`, or `3`
 */
template <
    math::linalg::mini::CMatrix TMatrixA,
    math::linalg::mini::CMatrix TMatrixB,
    math::linalg::mini::CMatrix TMatrixC,
    math::linalg::mini::CMatrix TMatrixUvw,
    class TScalar = typename TMatrixA::ScalarType>
PBAT_HOST_DEVICE auto PointTriangleLinearTangentialOperatorBlock(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixC const& c,
    TMatrixUvw const& uvw,
    int i)
{
    assert(i >= 0 and i <= 3);
    auto B = PointTriangleTangentialBasis(a, b, c);
    TScalar k =
        (i == 0) * TScalar(1) + (i == 1) * (-uvw(0)) + (i == 2) * (-uvw(1)) + (i == 3) * (-uvw(2));
    return k * B;
}

/**
 * @brief Compute tangential basis for edge-edge contact
 *
 * @tparam TMatrixP1 Matrix type for first edge first point
 * @tparam TMatrixQ1 Matrix type for first edge second point
 * @tparam TMatrixP2 Matrix type for second edge first point
 * @tparam TMatrixQ2 Matrix type for second edge second point
 * @tparam TScalar Scalar type
 * @param p1 `3 x 1` first edge first point
 * @param q1 `3 x 1` first edge second point
 * @param p2 `3 x 1` second edge first point
 * @param q2 `3 x 1` second edge second point
 * @return `3 x 2` tangential basis matrix
 */
template <
    math::linalg::mini::CMatrix TMatrixP1,
    math::linalg::mini::CMatrix TMatrixQ1,
    math::linalg::mini::CMatrix TMatrixP2,
    math::linalg::mini::CMatrix TMatrixQ2,
    class TScalar = typename TMatrixP1::ScalarType>
PBAT_HOST_DEVICE auto EdgeEdgeTangentialBasis(
    TMatrixP1 const& p1,
    TMatrixQ1 const& q1,
    TMatrixP2 const& p2,
    TMatrixQ2 const& q2) -> math::linalg::mini::SMatrix<TScalar, TMatrixP1::kRows, 2>
{
    using namespace math::linalg::mini;
    static_assert(
        TMatrixP1::kRows == TMatrixQ1::kRows,
        "p1 and q1 must have the same number of rows.");
    static_assert(
        TMatrixP1::kRows == TMatrixP2::kRows,
        "p1 and p2 must have the same number of rows.");
    static_assert(
        TMatrixP1::kRows == TMatrixQ2::kRows,
        "p1 and q2 must have the same number of rows.");
    auto constexpr kDims = TMatrixP1::kRows;
    SMatrix<TScalar, kDims, 2> T;
    auto t1                   = T.Col(0);
    auto t2                   = T.Col(1);
    t1                        = q1 - p1;
    SVector<TScalar, kDims> n = Cross(t1, q2 - p2);
    TScalar t1n               = Norm(t1);
    assert(t1n > TScalar(0));
    TScalar nn = Norm(n);
    assert(nn > TScalar(0));
    t1 /= t1n;
    n /= nn;
    t2 = Cross(n, t1);
    return T;
}

/**
 * @brief Compute linear tangential operator \f$ \mathbf{T} \f$ for edge-edge contact s.t.
 * tangential displacements are \f$ \mathbf{T} \begin{bmatrix}\mathbf{p_1} \\ \mathbf{q_1}
 * \\ \mathbf{p_2} \\ \mathbf{q_2}\end{bmatrix} \f$, s.t. the closest points `x1,x2` on both edges
 * are expressed in barycentric coordinates by `u1` and `u2`.
 *
 * @tparam TMatrixP1 Matrix type for first edge first point
 * @tparam TMatrixQ1 Matrix type for first edge second point
 * @tparam TMatrixP2 Matrix type for second edge first point
 * @tparam TMatrixQ2 Matrix type for second edge second point
 * @tparam TScalar Scalar type
 * @param p1 `3 x 1` first edge first point
 * @param q1 `3 x 1` first edge second point
 * @param p2 `3 x 1` second edge first point
 * @param q2 `3 x 1` second edge second point
 * @param u1 Barycentric coordinate of closest point on first edge
 * @param u2 Barycentric coordinate of closest point on second edge
 * @return `4*|# dims| x 2` tangential operator
 */
template <
    math::linalg::mini::CMatrix TMatrixP1,
    math::linalg::mini::CMatrix TMatrixQ1,
    math::linalg::mini::CMatrix TMatrixP2,
    math::linalg::mini::CMatrix TMatrixQ2,
    class TScalar = typename TMatrixP1::ScalarType>
PBAT_HOST_DEVICE auto EdgeEdgeLinearTangentialOperator(
    TMatrixP1 const& p1,
    TMatrixQ1 const& q1,
    TMatrixP2 const& p2,
    TMatrixQ2 const& q2,
    TScalar u1,
    TScalar u2)
{
    using namespace math::linalg::mini;
    static_assert(
        TMatrixP1::kRows == TMatrixQ1::kRows,
        "p1 and q1 must have the same number of rows.");
    static_assert(
        TMatrixP1::kRows == TMatrixP2::kRows,
        "p1 and p2 must have the same number of rows.");
    static_assert(
        TMatrixP1::kRows == TMatrixQ2::kRows,
        "p1 and q2 must have the same number of rows.");
    assert(u1 >= TScalar(0) and u1 <= TScalar(1));
    assert(u2 >= TScalar(0) and u2 <= TScalar(1));
    auto constexpr kDims         = TMatrixP1::kRows;
    SMatrix<TScalar, kDims, 2> B = EdgeEdgeTangentialBasis(p1, q1, p2, q2);
    SMatrix<TScalar, 4 * kDims, 2> T;
    T.template Slice<kDims, 2>(0, 0)         = (TScalar(1) - u1) * B;
    T.template Slice<kDims, 2>(kDims, 0)     = u1 * B;
    T.template Slice<kDims, 2>(2 * kDims, 0) = (u2 - TScalar(1)) * B;
    T.template Slice<kDims, 2>(3 * kDims, 0) = -u2 * B;
    return T;
}

/**
 * @brief Compute the (i-th) block of the linear tangential operator \f$ \mathbf{T} \f$ returned by
 * function `EdgeEdgeLinearTangentialOperator`.
 *
 * @tparam TMatrixP1 Matrix type for first edge first point
 * @tparam TMatrixQ1 Matrix type for first edge second point
 * @tparam TMatrixP2 Matrix type for second edge first point
 * @tparam TMatrixQ2 Matrix type for second edge second point
 * @tparam TScalar Scalar type
 * @param p1 `3 x 1` first edge first point
 * @param q1 `3 x 1` first edge second point
 * @param p2 `3 x 1` second edge first point
 * @param q2 `3 x 1` second edge second point
 * @param u1 Barycentric coordinate of closest point on first edge
 * @param u2 Barycentric coordinate of closest point on second edge
 * @param i Block row
 * @return `|# dims| x 2` tangential operator
 * @pre `i` must be either `0`, `1`, `2`, or `3`
 */
template <
    math::linalg::mini::CMatrix TMatrixP1,
    math::linalg::mini::CMatrix TMatrixQ1,
    math::linalg::mini::CMatrix TMatrixP2,
    math::linalg::mini::CMatrix TMatrixQ2,
    class TScalar = typename TMatrixP1::ScalarType>
PBAT_HOST_DEVICE auto EdgeEdgeLinearTangentialOperatorBlock(
    TMatrixP1 const& p1,
    TMatrixQ1 const& q1,
    TMatrixP2 const& p2,
    TMatrixQ2 const& q2,
    TScalar u1,
    TScalar u2,
    int i)
{
    assert(i >= 0 and i <= 3);
    auto B    = EdgeEdgeTangentialBasis(p1, q1, p2, q2);
    TScalar k = (i == 0) * (TScalar(1) - u1) + (i == 1) * u1 + (i == 2) * (u2 - TScalar(1)) +
                (i == 3) * (-u2);
    return k * B;
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_FRICTION_H
