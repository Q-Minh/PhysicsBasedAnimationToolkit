/**
 * @file ClosestPointQueries.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to answer closest point queries.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_CLOSESTPOINTQUERIES_H
#define PBAT_GEOMETRY_CLOSESTPOINTQUERIES_H

#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>
#include <cassert>

namespace pbat {
namespace geometry {
/**
 * @brief This namespace contains functions to answer closest point queries.
 */
namespace ClosestPointQueries {

namespace mini = math::linalg::mini;

/**
 * @brief Obtain the point on the plane (P,n) closest to the point X.
 * @tparam TMatrixX Query point matrix type
 * @tparam TMatrixP Point on the plane matrix type
 * @tparam TMatrixN Normal of the plane matrix type
 * @param X Query point
 * @param P Point on the plane
 * @param n Normal of the plane
 * @return Point on the plane closest to X
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto PointOnPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>;

/**
 * @brief Obtain the point on the line segment PQ closest to the point X.
 * @tparam TMatrixX Query point matrix type
 * @tparam TMatrixP Start point of the line segment matrix type
 * @tparam TMatrixQ End point of the line segment matrix type
 * @param X Query point
 * @param P Start point of the line segment
 * @param Q End point of the line segment
 * @return Point on the line segment closest to X
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ>
PBAT_HOST_DEVICE auto PointOnLineSegment(TMatrixX const& X, TMatrixP const& P, TMatrixQ const& Q)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>;

/**
 * @brief Obtain the point on the axis-aligned bounding box (AABB) defined by the lower
 * and upper corners closest to the point X.
 * @tparam TMatrixX Query point matrix type
 * @tparam TMatrixL Lower corner of the AABB matrix type
 * @tparam TMatrixU Upper corner of the AABB matrix type
 * @param X Query point
 * @param L Lower corner of the AABB
 * @param U Upper corner of the AABB
 * @return Point on the AABB closest to X
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE auto
PointOnAxisAlignedBoundingBox(TMatrixX const& X, TMatrixL const& L, TMatrixU const& U)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>;

/**
 * @brief Obtain the point on the triangle ABC closest to the point P in barycentric coordinates.
 * @tparam TMatrixP Query point matrix type
 * @tparam TMatrixA Vertex A of the triangle matrix type
 * @tparam TMatrixB Vertex B of the triangle matrix type
 * @tparam TMatrixC Vertex C of the triangle matrix type
 * @param P Query point
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return Barycentric coordinates of the point in the triangle closest to P
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
UvwPointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
    -> mini::SVector<typename TMatrixP::ScalarType, 3>;

/**
 * @brief Obtain the point on the triangle ABC closest to the point P.
 * @tparam TMatrixP Query point matrix type
 * @tparam TMatrixA Vertex A of the triangle matrix type
 * @tparam TMatrixB Vertex B of the triangle matrix type
 * @tparam TMatrixC Vertex C of the triangle matrix type
 * @param P Query point
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return Point in the triangle closest to P
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
    -> mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>;

/**
 * @brief Obtain the point in the tetrahedron ABCD closest to the point P. The order of ABCD
 * must be such that all faces ABC, ACD, ADB and BDC are oriented with outwards pointing normals
 * when viewed from outside the tetrahedron.
 * @tparam TMatrixP Query point matrix type
 * @tparam TMatrixA Vertex A of the tetrahedron matrix type
 * @tparam TMatrixB Vertex B of the tetrahedron matrix type
 * @tparam TMatrixC Vertex C of the tetrahedron matrix type
 * @tparam TMatrixD Vertex D of the tetrahedron matrix type
 * @param P Query point
 * @param A Vertex A of the tetrahedron
 * @param B Vertex B of the tetrahedron
 * @param C Vertex C of the tetrahedron
 * @param D Vertex D of the tetrahedron
 * @return Point in the tetrahedron closest to P
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE auto PointInTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D) -> mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>;

/**
 * @brief Find closest points on two lines defined by points P1, Q1 and P2, Q2.
 *
 * We find closest points by minimizing
 * \f[
 * \begin{align*}
 * f(\alpha, \beta) &= | (P1 + \alpha d_1) - (P2 + \beta d_2) |^2 \\
 * &= | (P1 - P2) + (\alpha d_1 - \beta d_2) |^2 ,
 * \end{align*}
 * \f]
 * where \f$ d_1 = Q1 - P1 \f$ and \f$ d_2 = Q2 - P2 \f$.
 *
 * We can expand \f$ f \f$ to obtain
 * \f[
 * \begin{align*}
 * f(\alpha,\beta) &= |P1-P2|^2 + 2 (P1-P2)^T (\alpha d_1 - \beta d_2) + | (\alpha d_1 - \beta d_2)
 * |^2 \\
 * &= |P1-P2|^2 + 2 (P1-P2)^T (\alpha d_1 - \beta d_2) + \alpha^2 |d_1|^2 - 2 \alpha \beta d_1^T d_2
 * + \beta^2 |d_2|^2 .
 * \end{align*}
 * \f]
 *
 * The gradients are thus
 * \f[
 * \begin{align*}
 * \frac{\partial f}{\partial \alpha} &= 2 (P1-P2)^T d_1 + 2 \alpha |d_1|^2 - 2 \beta d_1^T d_2 \\
 * \frac{\partial f}{\partial \beta} &= -2 (P1-P2)^T d_2 - 2 \alpha d_1^T d_2 + 2 \beta |d_2|^2
 * \end{align*}
 * \f]
 *
 * Setting the gradients to zero, we obtain the following equations
 * \f[
 * \begin{align*}
 * \beta &= \frac{(P1-P2)^T d_2 + \alpha d_1^T d_2}{|d_2|^2} \\
 * &= \frac{(P1-P2+\alpha d_1)^T d_2}{|d_2|^2} \\
 * \alpha &= \frac{\beta d_1^T d_2 - (P1-P2)^T d_1}{|d_1|^2} .
 * \end{align*}
 * \f]
 *
 * Injecting the first equation into the second, we obtain
 * \f[
 * \begin{align*}
 * \alpha &= \frac{-(P1-P2)^T d_1}{|d_1|^2} + \frac{(P1-P2)^T d_2 + \alpha d_1^T d_2}{|d_1|^2
 * |d_2|^2} d_1^T d_2 \\
 * \alpha &= \frac{-(P1-P2)^T d_1}{|d_1|^2} + \frac{(P1-P2)^T d_2}{|d_1|^2 |d_2|^2} d_1^T d_2 +
 * \alpha \frac{(d_1^T d_2)^2}{|d_1|^2 |d_2|^2} \\
 * (1 - \frac{(d_1^T d_2)^2}{|d_1|^2 |d_2|^2}) \alpha &= \frac{-(P1-P2)^T d_1}{|d_1|^2} +
 * \frac{(P1-P2)^T d_2}{|d_1|^2 |d_2|^2} d_1^T d_2 \\
 * (1 - \frac{(d_1^T d_2)^2}{|d_1|^2 |d_2|^2}) \alpha &= \frac{(P1-P2)^T (d_2 d_1^T d_2 - |d_2|^2
 * d_1)}{|d_1|^2 |d_2|^2} \\
 * (|d_1|^2 |d_2|^2 - (d_1^T d_2)^2) \alpha &= (P1-P2)^T (d_2 d_1^T d_2 - |d_2|^2 d_1) .
 * \end{align*}
 * \f]
 * which allows solving for \f$ \alpha \f$ first, then \f$ \beta \f$.
 *
 * Note that if the lines are parallel, then by definition,
 * \f[
 * \begin{align*}
 * d_1^T d_2 &= |d_1| |d_2| \cos(\theta) \\
 * &= |d_1| |d_2| \\
 * (d_1^T d_2)^2 &= |d_1|^2 |d_2|^2 \\
 * |d_1|^2 |d_2|^2 - (d_1^T d_2)^2 &= 0 .
 * \end{align*}
 * \f]
 *
 * In that case, we choose \f$ \alpha = 0 \f$ and return the corresponding \f$ \beta \f$.
 * If any line \f$ d_i = 0 \f$, it is degenerated into a point. In that case, we choose the
 * closest point on the other line.
 * If both lines are degenerated into points, the solution is \f$ \alpha, \beta = 0 \f$.
 *
 * @tparam TMatrixP1 Type of the input matrix P1
 * @tparam TMatrixQ1 Type of the input matrix Q1
 * @tparam TMatrixP2 Type of the input matrix P2
 * @tparam TMatrixQ2 Type of the input matrix Q2
 * @tparam TScalar Type of the scalar
 * @param P1 Point 1 on line 1
 * @param Q1 Point 2 on line 1
 * @param P2 Point 1 on line 2
 * @param Q2 Point 2 on line 2
 * @param eps Numerical error tolerance for zero checks
 * @return 2-vector \f$ (\alpha, \beta) \f$ such that the closest points are \f$ P1 + \alpha * (Q1 -
 * P1) \f$ and \f$ P2 + \beta * (Q2 - P2) \f$
 * @pre `eps >= 0`
 */
template <
    mini::CMatrix TMatrixP1,
    mini::CMatrix TMatrixQ1,
    mini::CMatrix TMatrixP2,
    mini::CMatrix TMatrixQ2,
    class TScalar = typename TMatrixP1::ScalarType>
PBAT_HOST_DEVICE auto Lines(
    TMatrixP1 const& P1,
    TMatrixQ1 const& Q1,
    TMatrixP2 const& P2,
    TMatrixQ2 const& Q2,
    TScalar eps = std::numeric_limits<TScalar>::min()) -> mini::SVector<TScalar, 2>;

template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto PointOnPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
{
    using namespace std;
    using ScalarType = typename TMatrixX::ScalarType;
#ifndef NDEBUG
    bool const bIsNormalUnit = abs(SquaredNorm(n) - ScalarType(1)) <= ScalarType(1e-15);
    assert(bIsNormalUnit);
#endif
    /**
     * See \cite ericson2004real section 5.11
     */
    ScalarType const t                                = Dot(n, X - P);
    mini::SVector<ScalarType, TMatrixX::kRows> Xplane = X - t * n;
    return Xplane;
}

template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ>
PBAT_HOST_DEVICE auto PointOnLineSegment(TMatrixX const& X, TMatrixP const& P, TMatrixQ const& Q)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
{
    using ScalarType = typename TMatrixX::ScalarType;
    using namespace std;
    /**
     * See \cite ericson2004real section 5.12
     */
    mini::SVector<ScalarType, TMatrixX::kRows> const PQ = Q - P;
    // Project X onto PQ, computing parameterized position R(t) = P + t*(Q � P)
    ScalarType t = Dot(X - P, PQ) / SquaredNorm(PQ);
    // If outside segment, clamp t (and therefore d) to the closest endpoint
    t = min(max(t, ScalarType(0)), ScalarType(1));
    // Compute projected position from the clamped t
    auto const Xpq = P + t * PQ;
    return Xpq;
}

template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE auto
PointOnAxisAlignedBoundingBox(TMatrixX const& P, TMatrixL const& L, TMatrixU const& U)
    -> mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
{
    using namespace std;
    /**
     * See \cite ericson2004real section 5.13
     */
    mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows> X = P;
    pbat::common::ForRange<0, TMatrixX::kRows>(
        [&]<auto i>() { X(i) = min(max(X(i), L(i)), U(i)); });
    return X;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
UvwPointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
    -> mini::SVector<typename TMatrixP::ScalarType, 3>
{
    using ScalarType = typename TMatrixP::ScalarType;
    /**
     * See \cite ericson2004real section 5.15
     */

    // Check if P in vertex region outside A
    auto constexpr kRows                      = TMatrixP::kRows;
    mini::SVector<ScalarType, kRows> const AB = B - A;
    mini::SVector<ScalarType, kRows> const AC = C - A;
    mini::SVector<ScalarType, kRows> const AP = P - A;
    ScalarType const d1                       = Dot(AB, AP);
    ScalarType const d2                       = Dot(AC, AP);
    if (d1 <= ScalarType(0) and d2 <= ScalarType(0))
    {
        return mini::Unit<ScalarType, 3>(0); // barycentric coordinates (1,0,0)
    }

    // Check if P in vertex region outside B
    mini::SVector<ScalarType, kRows> const BP = P - B;
    ScalarType const d3                       = Dot(AB, BP);
    ScalarType const d4                       = Dot(AC, BP);
    if (d3 >= ScalarType(0) and d4 <= d3)
    {
        return mini::Unit<ScalarType, 3>(1); // barycentric coordinates (0,1,0)
    }

    // Check if P in edge region of AB, if so return projection of P onto AB
    ScalarType const vc = d1 * d4 - d3 * d2;
    if (vc <= ScalarType(0) and d1 >= ScalarType(0) and d3 <= ScalarType(0))
    {
        ScalarType const v = d1 / (d1 - d3);
        return mini::SVector<ScalarType, 3>{
            ScalarType(1) - v,
            v,
            ScalarType(0)}; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    mini::SVector<ScalarType, kRows> const CP = P - C;
    ScalarType const d5                       = Dot(AB, CP);
    ScalarType const d6                       = Dot(AC, CP);
    if (d6 >= ScalarType(0) and d5 <= d6)
    {
        return mini::Unit<ScalarType, 3>(2); // barycentric coordinates (0,0,1)
    }

    // Check if P in edge region of AC, if so return projection of P onto AC
    ScalarType const vb = d5 * d2 - d1 * d6;
    if (vb <= ScalarType(0) and d2 >= ScalarType(0) and d6 <= ScalarType(0))
    {
        ScalarType const w = d2 / (d2 - d6);
        return mini::SVector<ScalarType, 3>{
            ScalarType(1) - w,
            ScalarType(0),
            w}; // barycentric coordinates (1-w,0,w)
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    ScalarType const va = d3 * d6 - d5 * d4;
    if (va <= ScalarType(0) and (d4 - d3) >= ScalarType(0) and (d5 - d6) >= ScalarType(0))
    {
        ScalarType const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return mini::SVector<ScalarType, 3>{
            ScalarType(0),
            ScalarType(1) - w,
            w}; // barycentric coordinates (0,1-w,w)
    }
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    ScalarType const denom = ScalarType(1) / (va + vb + vc);
    ScalarType const v     = vb * denom;
    ScalarType const w     = vc * denom;
    return mini::SVector<ScalarType, 3>{
        ScalarType(1) - v - w,
        v,
        w}; // = u*a + v*b + w*c, u = va * denom = ScalarType(1)0f-v-w
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
    -> mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>
{
    auto uvw = UvwPointInTriangle(P, A, B, C);
    return A * uvw(0) + B * uvw(1) + C * uvw(2);
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE auto PointInTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D) -> mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");

    /**
     * See \cite ericson2004real section 5.16
     */

    // Start out assuming point inside all halfspaces, so closest to itself
    mini::SVector<ScalarType, kDims> X = P;
    ScalarType d2min                   = std::numeric_limits<ScalarType>::max();

    auto const PointOutsidePlane = [](auto const& p, auto const& a, auto const& b, auto const& c) {
        ScalarType const d = Dot(p - a, Cross(b - a, c - a));
        return d > ScalarType(0);
    };
    auto const TestFace = [&](auto const& a, auto const& b, auto const& c) {
        // If point outside face abc then compute closest point on abc
        if (PointOutsidePlane(P, a, b, c))
        {
            auto const Q        = PointInTriangle(P, a, b, c);
            ScalarType const d2 = SquaredNorm(Q - P);
            // Update best closest point if (squared) distance is less than current best
            if (d2 < d2min)
            {
                d2min = d2;
                X     = Q;
            }
        }
    };
    TestFace(A, B, D);
    TestFace(B, C, D);
    TestFace(C, A, D);
    TestFace(A, C, B);
    return X;
}

template <
    mini::CMatrix TMatrixP1,
    mini::CMatrix TMatrixQ1,
    mini::CMatrix TMatrixP2,
    mini::CMatrix TMatrixQ2,
    class TScalar>
PBAT_HOST_DEVICE auto Lines(
    TMatrixP1 const& P1,
    TMatrixQ1 const& Q1,
    TMatrixP2 const& P2,
    TMatrixQ2 const& Q2,
    TScalar eps) -> mini::SVector<TScalar, 2>
{
    auto constexpr kDims                     = TMatrixP1::kRows;
    mini::SVector<TScalar, kDims> const d1   = Q1 - P1;
    mini::SVector<TScalar, kDims> const d2   = Q2 - P2;
    mini::SVector<TScalar, kDims> const P2P1 = P1 - P2;
    TScalar const d1sq                       = SquaredNorm(d1);
    TScalar const d2sq                       = SquaredNorm(d2);
    TScalar const d1d2                       = Dot(d1, d2);
    TScalar const similarity                 = d1sq * d2sq - d1d2 * d1d2;
    bool const bIsLine1Degenerate            = d1sq <= eps;
    bool const bIsLine2Degenerate            = d2sq <= eps;
    bool const bAreParallel                  = (-eps <= similarity) and (similarity <= eps);
    TScalar alpha{0}, beta{0};
    if (not(bIsLine1Degenerate and bIsLine2Degenerate))
    {
        if (bIsLine1Degenerate)
        {
            beta = Dot(P2P1, d2) / d2sq;
        }
        else if (bIsLine2Degenerate)
        {
            alpha = -Dot(P2P1, d1) / d1sq;
        }
        else
        {
            if (not bAreParallel)
                alpha = Dot(P2P1, d2 * d1d2 - d2sq * d1) / similarity;
            beta = Dot(P2P1 + alpha * d1, d2) / d2sq;
        }
    }
    return mini::SVector<TScalar, 2>{alpha, beta};
}

} // namespace ClosestPointQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_CLOSESTPOINTQUERIES_H
