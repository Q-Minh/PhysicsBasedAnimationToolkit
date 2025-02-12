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
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)1
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
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)2
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
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)3
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
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)5
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
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)6
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

} // namespace ClosestPointQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_CLOSESTPOINTQUERIES_H
