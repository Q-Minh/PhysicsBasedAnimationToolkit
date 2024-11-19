#ifndef PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H
#define PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H

#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>
#include <cassert>

namespace pbat {
namespace geometry {
namespace ClosestPointQueries {

namespace mini = math::linalg::mini;

/**
 * @brief Obtain the point on the plane (P,n) closest to the point X.
 * @param X
 * @param P
 * @param n
 * @return
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n);

/**
 * @brief Obtain the point on the line segment PQ closest to the point X.
 * @param X
 * @param P
 * @param Q
 * @return
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnLineSegment(TMatrixX const& X, TMatrixP const& P, TMatrixQ const& Q);

/**
 * @brief Obtain the point on the axis-aligned bounding box (AABB) defined by the lower
 * and upper corners closest to the point P.
 * @param P
 * @param L
 * @param U
 * @return
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnAxisAlignedBoundingBox(TMatrixX const& X, TMatrixL const& L, TMatrixU const& U);

/**
 * @brief Obtain the point on the triangle ABC closest to the point P in barycentric coordinates.
 * @param P
 * @param A
 * @param B
 * @param C
 * @return
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, 3>
UvwPointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C);

/**
 * @brief Obtain the point on the triangle ABC closest to the point P.
 * @param P
 * @param A
 * @param B
 * @param C
 * @return
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>
PointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C);

/**
 * @brief Obtain the point in the tetrahedron ABCD closest to the point P. The order of ABCD
 * must be such that all faces ABC, ACD, ADB and BDC are oriented with outwards pointing normals
 * when viewed from outside the tetrahedron.
 * @param P
 * @param A
 * @param B
 * @param C
 * @param D
 * @return
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows> PointInTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D);

template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n)
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
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnLineSegment(TMatrixX const& X, TMatrixP const& P, TMatrixQ const& Q)
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
PBAT_HOST_DEVICE mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows>
PointOnAxisAlignedBoundingBox(TMatrixX const& P, TMatrixL const& L, TMatrixU const& U)
{
    using namespace std;
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.ScalarType(1)3
     */
    mini::SVector<typename TMatrixX::ScalarType, TMatrixX::kRows> X = P;
    for (auto i = 0; i < P.Rows(); ++i)
        X(i) = min(max(X(i), L(i)), U(i));
    return X;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, 3>
UvwPointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
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
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>
PointInTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
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
PBAT_HOST_DEVICE mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows> PointInTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D)
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

#endif // PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H