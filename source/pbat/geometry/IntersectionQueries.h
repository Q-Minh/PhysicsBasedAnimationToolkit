/**
 * @file IntersectionQueries.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to answer intersection queries.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_INTERSECTIONQUERIES_H
#define PBAT_GEOMETRY_INTERSECTIONQUERIES_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <array>
#include <cmath>
#include <optional>

namespace pbat {
namespace geometry {
/**
 * @brief This namespace contains functions to answer intersection queries.
 */
namespace IntersectionQueries {
namespace mini = math::linalg::mini;

/**
 * @brief Computes the barycentric coordinates of point P with respect to the triangle spanned by
 * vertices A, B, and C.
 *
 * @tparam TMatrixAP Matrix type of the vector from A to P
 * @tparam TMatrixAB Matrix type of the vector from A to B
 * @tparam TMatrixAC Matrix type of the vector from A to C
 * @param AP The vector from A to P
 * @param AB The vector from A to B
 * @param AC The vector from A to C
 * @return The barycentric coordinates of P with respect to the triangle ABC
 * @pre AP, AB, and AC are coplanar
 */
template <mini::CMatrix TMatrixAP, mini::CMatrix TMatrixAB, mini::CMatrix TMatrixAC>
PBAT_HOST_DEVICE auto
TriangleBarycentricCoordinates(
    TMatrixAP const& AP,
    TMatrixAB const& AB,
    TMatrixAC const& AC)
    -> mini::SMatrix<typename TMatrixAP::ScalarType, 3, 1>
{
    using ScalarType = typename TMatrixAP::ScalarType;
    using namespace mini;
    ScalarType const d00   = Dot(AB, AB);
    ScalarType const d01   = Dot(AB, AC);
    ScalarType const d11   = Dot(AC, AC);
    ScalarType const d20   = Dot(AP, AB);
    ScalarType const d21   = Dot(AP, AC);
    ScalarType const denom = d00 * d11 - d01 * d01;
    ScalarType const v     = (d11 * d20 - d01 * d21) / denom;
    ScalarType const w     = (d00 * d21 - d01 * d20) / denom;
    ScalarType const u     = ScalarType(1) - v - w;
    SMatrix<ScalarType, 3, 1> uvw{};
    uvw(0, 0) = u;
    uvw(1, 0) = v;
    uvw(2, 0) = w;
    return uvw;
};

/**
 * @brief Computes the barycentric coordinates of point P with respect to the triangle spanned by
 * vertices A, B, and C.
 *
 * @tparam TMatrixP Matrix type of the point P
 * @tparam TMatrixA Matrix type of the vertex A
 * @tparam TMatrixB Matrix type of the vertex B
 * @tparam TMatrixC Matrix type of the vertex C
 * @param P Point from which to compute barycentric coordinates.
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return The barycentric coordinates of P with respect to the triangle ABC
 * @pre P is in the plane spanned by triangle ABC.
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto TriangleBarycentricCoordinates(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> mini::SVector<typename TMatrixP::ScalarType, 3>
{
    using ScalarType                          = typename TMatrixP::ScalarType;
    auto constexpr kRows                      = TMatrixP::kRows;
    mini::SVector<ScalarType, kRows> const AP = P - A;
    mini::SVector<ScalarType, kRows> const AB = B - A;
    mini::SVector<ScalarType, kRows> const AC = C - A;
    return TriangleBarycentricCoordinates(AP, AB, AC);
};

/**
 * @brief Computes the intersection volume between 2 axis aligned bounding boxes
 * @tparam TMatrixL1 Matrix type of the lower bound of AABB 1
 * @tparam TMatrixU1 Matrix type of the upper bound of AABB 1
 * @tparam TMatrixL2 Matrix type of the lower bound of AABB 2
 * @tparam TMatrixU2 Matrix type of the upper bound of AABB 2
 * @param L1 The lower bound of AABB 1
 * @param U1 The upper bound of AABB 1
 * @param L2 The lower bound of AABB 2
 * @param U2 The upper bound of AABB 2
 * @return The intersection volume between the 2 AABBs
 */
template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE auto AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2) -> mini::SMatrix<typename TMatrixL1::ScalarType, TMatrixL1::Rows, 2>;

/**
 * @brief Computes the intersection point, if any, between a line segment PQ and a sphere (C,r).
 *
 * If there are 2 intersection points, returns the one closest to P along PQ.
 *
 * @param P The start point of the line segment
 * @param Q The end point of the line segment
 * @param C The center of the sphere
 * @param R The radius of the sphere
 * @return The intersection point, if any
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto LineSegmentSphere(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixC const& C,
    typename TMatrixC::ScalarType R)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>;

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and the
 * plane spanned by triangle ABC, in 3D.
 *
 * @tparam TMatrixP Matrix type of the point P
 * @tparam TMatrixQ Matrix type of the point Q
 * @tparam TMatrixA Matrix type of the vertex A
 * @tparam TMatrixB Matrix type of the vertex B
 * @tparam TMatrixC Matrix type of the vertex C
 * @param P The start point of the line
 * @param Q The end point of the line
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return The intersection point, if any
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto LineSegmentPlane3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>;

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and the
 * plane (n,d), in 3D.
 *
 * @tparam TMatrixP Matrix type of the point P
 * @tparam TMatrixQ Matrix type of the point Q
 * @tparam TMatrixN Matrix type of the normal vector of the plane
 * @param P The start point of the line
 * @param Q The end point of the line
 * @param n The normal vector of the plane
 * @param d The distance from the origin to the plane
 * @return The intersection point, if any
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto LineSegmentPlane3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixN const& n,
    typename TMatrixN::ScalarType d)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>;

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and a
 * triangle ABC, in 3D.
 *
 * @tparam TMatrixP Matrix type of the point P
 * @tparam TMatrixQ Matrix type of the point Q
 * @tparam TMatrixA Matrix type of the vertex A
 * @tparam TMatrixB Matrix type of the vertex B
 * @tparam TMatrixC Matrix type of the vertex C
 * @param P The start point of the line
 * @param Q The end point of the line
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return The intersection point, if any
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto UvwLineTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> std::optional<mini::SVector<typename TMatrixP::ScalarType, 3>>;

/**
 * @brief Computes the intersection point, if any, between a line segment delimited by points
 * P,Q and a triangle ABC, in 3D.
 *
 * @tparam TMatrixP Matrix type of the point P
 * @tparam TMatrixQ Matrix type of the point Q
 * @tparam TMatrixA Matrix type of the vertex A
 * @tparam TMatrixB Matrix type of the vertex B
 * @tparam TMatrixC Matrix type of the vertex C
 * @param P The start point of the line segment
 * @param Q The end point of the line segment
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return The intersection point as a 4-vector, if any, where the last 3 elements are the
 * barycentric coordinates of the intersection point with respect to the triangle ABC, and the first
 * element is the barycentric coordinate of the intersection point with respect to line segment PQ
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto UvwLineSegmentTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> std::optional<mini::SVector<typename TMatrixP::ScalarType, 4>>;

/**
 * @brief Computes intersection points between 3 edges (as line segments) of triangle A1B1C1 and
 * triangle A2B2C2, and intersection points between 3 edges (as line segments) of triangle
 * A2B2C2 and triangle A1B1C1, in 3D.
 *
 * @tparam TMatrixA1 Matrix type of the vertex A1 of triangle 1
 * @tparam TMatrixB1 Matrix type of the vertex B1 of triangle 1
 * @tparam TMatrixC1 Matrix type of the vertex C1 of triangle 1
 * @tparam TMatrixA2 Matrix type of the vertex A2 of triangle 2
 * @tparam TMatrixB2 Matrix type of the vertex B2 of triangle 2
 * @tparam TMatrixC2 Matrix type of the vertex C2 of triangle 2
 * @param A1 Vertex A of triangle 1
 * @param B1 Vertex B of triangle 1
 * @param C1 Vertex C of triangle 1
 * @param A2 Vertex A of triangle 2
 * @param B2 Vertex B of triangle 2
 * @param C2 Vertex C of triangle 2
 * @return The intersection points, if any
 */
template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE auto UvwTriangles3D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2)
    -> std::array<std::optional<mini::SVector<typename TMatrixA1::ScalarType, 3>>, 6u>;

template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE auto AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2) -> mini::SMatrix<typename TMatrixL1::ScalarType, TMatrixL1::Rows, 2>
{
    using ScalarType = typename TMatrixL1::ScalarType;
    mini::SMatrix<ScalarType, TMatrixL1::Rows, 2> LU;
    LU.Col(0) = Max(L1, L2);
    LU.Col(1) = Min(U1, U2);
    return LU;
}

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto LineSegmentSphere(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixC const& C,
    typename TMatrixC::ScalarType R)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>
{
    using ScalarType                          = typename TMatrixP::ScalarType;
    auto constexpr kRows                      = TMatrixP::kRows;
    mini::SVector<ScalarType, kRows> const PQ = (Q - P);
    ScalarType const len                      = Norm(PQ);
    mini::SVector<ScalarType, kRows> const d  = PQ / len;
    mini::SVector<ScalarType, kRows> const m  = P - C;
    ScalarType const b                        = Dot(m, d);
    ScalarType const c                        = Dot(m, m) - (R * R);
    // Exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > ScalarType(0) and b > ScalarType(0))
        return {};
    ScalarType const discr = b * b - c;
    // A negative discriminant corresponds to ray missing sphere
    if (discr < ScalarType(0))
        return {};
    // Ray now found to intersect sphere, compute smallest t value of intersection
    using namespace std;
    ScalarType t = -b - sqrt(discr);
    // If t is negative, ray started inside sphere so clamp t to zero
    if (t < ScalarType(0))
        t = ScalarType(0);
    // If the intersection point lies beyond the segment PQ, then return nullopt
    if (t > len)
        return {};

    mini::SVector<ScalarType, kRows> I = P + t * d;
    return I;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto LineSegmentPlane3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");
    // Intersect segment ab against plane of triangle def. If intersecting,
    // return t value and position q of intersection
    mini::SVector<ScalarType, kRows> const n = Cross(B - A, C - A);
    ScalarType const d                       = Dot(n, A);
    return LineSegmentPlane3D(P, Q, n, d);
}

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto LineSegmentPlane3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixN const& n,
    typename TMatrixN::ScalarType d)
    -> std::optional<mini::SVector<typename TMatrixP::ScalarType, TMatrixP::kRows>>
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;
    // Compute the t value for the directed line ab intersecting the plane
    mini::SVector<ScalarType, kRows> const PQ = Q - P;
    ScalarType const t                        = (d - Dot(n, P)) / Dot(n, PQ);
    // If t in [0..1] compute and return intersection point
    if (t >= ScalarType(0) and t <= ScalarType(1))
    {
        auto I = P + t * PQ;
        return I;
    }
    // Else no intersection
    return {};
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto UvwLineTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> std::optional<mini::SVector<typename TMatrixP::ScalarType, 3>>
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");
    /**
     * See \cite ericson2004real section 5.3.4
     */
    mini::SVector<ScalarType, kDims> const PQ = Q - P;
    mini::SVector<ScalarType, kDims> const PA = A - P;
    mini::SVector<ScalarType, kDims> const PB = B - P;
    mini::SVector<ScalarType, kDims> const PC = C - P;
    // Test if pq is inside the edges bc, ca and ab. Done by testing
    // that the signed tetrahedral volumes, computed using ScalarType triple
    // products, are all non-zero
    mini::SVector<ScalarType, kDims> const m = Cross(PQ, PC);
    mini::SVector<ScalarType, kDims> uvw     = mini::Zeros<ScalarType, kDims, 1>();

    ScalarType& u = uvw(0);
    ScalarType& v = uvw(1);
    ScalarType& w = uvw(2);
    u             = Dot(PB, m);
    v             = -Dot(PA, m);
    using namespace std;
    auto const SameSign = [](ScalarType a, ScalarType b) -> bool {
        return signbit(a) == signbit(b);
    };
    if (not SameSign(u, v))
        return {};
    w = Dot(PQ, Cross(PB, PA));
    if (not SameSign(u, w))
        return {};

    ScalarType constexpr eps            = 1e-15;
    ScalarType const uvwSum             = u + v + w;
    bool const bIsLineInPlaneOfTriangle = abs(uvwSum) < eps;
    if (bIsLineInPlaneOfTriangle)
    {
        // WARNING:
        // Technically, if the line is in the plane of the triangle, it is intersecting
        // the triangle in a line segment, hence there are an infinite number of solutions.
        // However, I don't want to spend too much compute power to return one of those
        // solutions. If we ever need this feature in the future, we'll implement it at
        // that point in time.
        return {};
    }

    // Compute the barycentric coordinates (u, v, w) determining the
    // intersection point r, r = u*a + v*b + w*c
    ScalarType const denom = ScalarType(1) / (uvwSum);
    u *= denom;
    v *= denom;
    w *= denom; // w = ScalarType(1) - u - v;
    return uvw;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto UvwLineSegmentTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> std::optional<mini::SVector<typename TMatrixP::ScalarType, 4>>
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");
    mini::SVector<ScalarType, kDims> const AB = B - A;
    mini::SVector<ScalarType, kDims> const AC = C - A;
    mini::SVector<ScalarType, kDims> const PQ = Q - P;
    mini::SVector<ScalarType, kDims> const n  = Cross(AB, AC);
    // Compute denominator d. If d == 0, segment is parallel to triangle, so exit early
    ScalarType constexpr eps = static_cast<ScalarType>(1e-15);
    ScalarType const d       = Dot(PQ, n);
    using namespace std;
    bool const bIsSegmentParallelToTriangle = abs(d) < eps;
    if (bIsSegmentParallelToTriangle)
        return {};
    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= ScalarType(1).
    ScalarType const t = Dot(n, A - P) / d;
    if (t < ScalarType(0) or t > ScalarType(1))
        return {};
    // Compute barycentric coordinate components and test if within bounds
    mini::SVector<ScalarType, kDims> const I = P + t * PQ;
    mini::SVector<ScalarType, 4> uvwt;
    auto uvw                     = uvwt.template Slice<3, 1>(1, 0);
    uvw                          = TriangleBarycentricCoordinates(I, A, B, C);
    bool const bIsInsideTriangle = All((uvw >= ScalarType(0)) and (uvw <= ScalarType(1)));
    if (not bIsInsideTriangle)
        return {};
    uvwt(0) = t;
    return uvwt;
}

template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE auto UvwTriangles3D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2)
    -> std::array<std::optional<mini::SVector<typename TMatrixA1::ScalarType, 3>>, 6u>
{
    using ScalarType     = typename TMatrixA1::ScalarType;
    auto constexpr kRows = TMatrixA1::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");

    using namespace std;
    mini::SVector<ScalarType, kDims> const n1 = Normalized(Cross(B1 - A1, C1 - A1));
    mini::SVector<ScalarType, kDims> const n2 = Normalized(Cross(B2 - A2, C2 - A2));
    ScalarType constexpr eps                  = 1e-15;
    bool const bAreTrianglesCoplanar          = (ScalarType(1) - abs(Dot(n1, n2))) < eps;
    if (bAreTrianglesCoplanar)
    {
        // NOTE: If triangles are coplanar and vertex of one triangle is in the plane of the other
        // triangle, then they are intersecting. We do not handle this case for now.
        return {};
    }
    // NOTE: Maybe handle degenerate triangles? Or maybe we should require the user to check
    // that.
    // - For a colinear triangle, we can perform line segment against triangle test.
    // - For 2 colinear triangles, we can perform line segment against line segment test.
    // - For a triangle collapsed to a point, we perform point in triangle test.
    // - For 2 triangles collapsed to a point, we check if both points are numerically the same.

    // Test 3 edges of each triangle against the other triangle
    std::array<std::optional<mini::SVector<ScalarType, 3>>, 6u> intersections;
    auto uvwt = UvwLineSegmentTriangle3D(A1, B1, A2, B2, C2);
    #if defined(CUDART_VERSION)
    #pragma nv_diag_suppress 174
    #endif
    if (uvwt)
        intersections[0] = uvwt->template Slice<3, 1>(1, 0);
    uvwt = UvwLineSegmentTriangle3D(B1, C1, A2, B2, C2);
    if (uvwt)
        intersections[1] = uvwt->template Slice<3, 1>(1, 0);
    uvwt = UvwLineSegmentTriangle3D(C1, A1, A2, B2, C2);
    if (uvwt)
        intersections[2] = uvwt->template Slice<3, 1>(1, 0);
    uvwt = UvwLineSegmentTriangle3D(A2, B2, A1, B1, C1);
    if (uvwt)
        intersections[3] = uvwt->template Slice<3, 1>(1, 0);
    uvwt = UvwLineSegmentTriangle3D(B2, C2, A1, B1, C1);
    if (uvwt)
        intersections[4] = uvwt->template Slice<3, 1>(1, 0);
    uvwt = UvwLineSegmentTriangle3D(C2, A2, A1, B1, C1);
    if (uvwt)
        intersections[5] = uvwt->template Slice<3, 1>(1, 0);
    #if defined(CUDART_VERSION)
    #pragma nv_diag_default 174
    #endif
    return intersections;
}
} // namespace IntersectionQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_INTERSECTIONQUERIES_H