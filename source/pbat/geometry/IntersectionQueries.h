#ifndef PBAT_GEOMETRY_INTERSECTION_QUERIES_H
#define PBAT_GEOMETRY_INTERSECTION_QUERIES_H

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <optional>
#include <pbat/Aliases.h>

namespace pbat {
namespace geometry {
namespace IntersectionQueries {

/**
 * @brief Computes the intersection volume between 2 axis aligned bounding boxes
 * @param aabb1
 * @param aabb2
 * @return
 */
template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
Matrix<TDerivedL1::Rows, 2> AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2);

/**
 * @brief Computes the intersection point, if any, between a line segment PQ and a sphere (C,r).
 * If there are 2 intersection points, returns the one closest to P along PQ.
 * @param P
 * @param Q
 * @param C
 * @param R
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedC>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentSphere(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R);

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and the
 * plane spanned by triangle ABC, in 3D.
 * @tparam TDerivedQ
 * @tparam TDerivedA
 * @tparam TDerivedB
 * @tparam TDerivedC
 * @tparam TDerivedP
 * @param P
 * @param Q
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentPlane3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and the
 * plane (n,d), in 3D.
 * @tparam TDerivedP
 * @tparam TDerivedQ
 * @tparam TDerivedn
 * @param P
 * @param Q
 * @param n
 * @param d
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedn>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentPlane3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedn> const& n,
    Scalar d);

/**
 * @brief Computes the intersection point, if any, between a line including points P,Q and a
 * triangle ABC, in 3D.
 * @param P
 * @param Q
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<3>> UvwLineTriangle3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Computes the intersection point, if any, between a line segment delimited by points
 * P,Q and a triangle ABC, in 3D.
 * @param P
 * @param Q
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<3>> UvwLineSegmentTriangle3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Computes intersection points between 3 edges (as line segments) of triangle A1B1C1 and
 * triangle A2B2C2, and intersection points between 3 edges (as line segments) of triangle
 * A2B2C2 and triangle A1B1C1, in 3D.
 * @param A1
 * @param B1
 * @param C1
 * @param A2
 * @param B2
 * @param C2
 * @return
 */
template <
    class TDerivedA1,
    class TDerivedB1,
    class TDerivedC1,
    class TDerivedA2,
    class TDerivedB2,
    class TDerivedC2>
std::array<std::optional<Vector<3>>, 6u> UvwTriangles3D(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2);

template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
Matrix<TDerivedL1::Rows, 2> AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2)
{
    Matrix<TDerivedL1::Rows, 2> LU;
    LU.col(0) = L1.cwiseMax(L2);
    LU.col(1) = U1.cwiseMin(U2);
    return LU;
}

template <class TDerivedP, class TDerivedQ, class TDerivedC>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentSphere(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R)
{
    auto constexpr Rows   = TDerivedP::RowsAtCompileTime;
    Vector<Rows> const PQ = (Q - P);
    Scalar const len      = PQ.norm();
    Vector<Rows> const d  = PQ / len;
    Vector<Rows> const m  = P - C;
    Scalar const b        = m.dot(d);
    Scalar const c        = m.dot(m) - (R * R);
    // Exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > 0. and b > 0.)
        return {};
    Scalar const discr = b * b - c;
    // A negative discriminant corresponds to ray missing sphere
    if (discr < 0.)
        return {};
    // Ray now found to intersect sphere, compute smallest t value of intersection
    Scalar t = -b - std::sqrt(discr);
    // If t is negative, ray started inside sphere so clamp t to zero
    if (t < 0.)
        t = 0.;
    // If the intersection point lies beyond the segment PQ, then return nullopt
    if (t > len)
        return {};

    Vector<Rows> const I = P + t * d;
    return I;
}

template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentPlane3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    auto constexpr kDims = 3;
    static_assert(Rows == kDims, "This overlap test is specialized for 3D");
    // Intersect segment ab against plane of triangle def. If intersecting,
    // return t value and position q of intersection
    Vector<Rows> const n = (B - A).cross(C - A);
    Scalar const d       = n.dot(A);
    return LineSegmentPlane3D(P, Q, n, d);
}

template <class TDerivedP, class TDerivedQ, class TDerivedn>
std::optional<Vector<TDerivedP::RowsAtCompileTime>> LineSegmentPlane3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedn> const& n,
    Scalar d)
{
    auto constexpr Rows = TDerivedP::RowsAtCompileTime;
    // Compute the t value for the directed line ab intersecting the plane
    Vector<Rows> const PQ = Q - P;
    Scalar const t        = (d - n.dot(P)) / n.dot(PQ);
    // If t in [0..1] compute and return intersection point
    if (t >= 0. and t <= 1.)
    {
        auto const I = P + t * PQ;
        return I;
    }
    // Else no intersection
    return {};
}

template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<3>> UvwLineTriangle3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    auto constexpr kDims = 3;
    static_assert(Rows == kDims, "This overlap test is specialized for 3D");
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.3.4
     */
    Vector<kDims> const PQ = Q - P;
    Vector<kDims> const PA = A - P;
    Vector<kDims> const PB = B - P;
    Vector<kDims> const PC = C - P;
    // Test if pq is inside the edges bc, ca and ab. Done by testing
    // that the signed tetrahedral volumes, computed using scalar triple
    // products, are all non-zero
    Vector<kDims> const m = PQ.cross(PC);
    Vector<kDims> uvw     = Vector<kDims>::Zero();

    Scalar& u           = uvw(0);
    Scalar& v           = uvw(1);
    Scalar& w           = uvw(2);
    u                   = PB.dot(m);
    v                   = -PA.dot(m);
    auto const SameSign = [](Scalar a, Scalar b) -> bool {
        return std::signbit(a) == std::signbit(b);
    };
    if (not SameSign(u, v))
        return {};
    w = PQ.dot(PB.cross(PA));
    if (not SameSign(u, w))
        return {};

    auto constexpr eps                  = 1e-15;
    Scalar const uvwSum                 = u + v + w;
    bool const bIsLineInPlaneOfTriangle = std::abs(uvwSum) < eps;
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
    Scalar const denom = 1. / (uvwSum);
    u *= denom;
    v *= denom;
    w *= denom; // w = 1. - u - v;
    return uvw;
}

template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
std::optional<Vector<3>> UvwLineSegmentTriangle3D(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    auto constexpr kDims = 3;
    static_assert(Rows == kDims, "This overlap test is specialized for 3D");
    Vector<kDims> const AB = B - A;
    Vector<kDims> const AC = C - A;
    Vector<kDims> const PQ = Q - P;
    Vector<kDims> const n  = AB.cross(AC);
    // Compute denominator d. If d == 0, segment is parallel to triangle, so exit early
    auto constexpr eps                      = 1e-15;
    Scalar const d                          = PQ.dot(n);
    bool const bIsSegmentParallelToTriangle = std::abs(d) < eps;
    if (bIsSegmentParallelToTriangle)
        return {};
    // Compute intersection t value of pq with plane of triangle. A ray
    // intersects iff 0 <= t. Segment intersects iff 0 <= t <= 1. Delay
    // dividing by d until intersection has been found to pierce triangle
    Scalar const t = n.dot(A - P) / d;
    if (t < 0. or t > 1.)
        return {};
    // Compute barycentric coordinate components and test if within bounds
    auto const BarycentricCoordinatesOf = [&](auto const& P) {
        Vector<kDims> const v0 = B - A;
        Vector<kDims> const v1 = C - A;
        Vector<kDims> const v2 = P - A;
        Scalar const d00       = v0.dot(v0);
        Scalar const d01       = v0.dot(v1);
        Scalar const d11       = v1.dot(v1);
        Scalar const d20       = v2.dot(v0);
        Scalar const d21       = v2.dot(v1);
        Scalar const denom     = d00 * d11 - d01 * d01;
        Scalar const v         = (d11 * d20 - d01 * d21) / denom;
        Scalar const w         = (d00 * d21 - d01 * d20) / denom;
        Scalar const u         = 1. - v - w;
        return Vector<3>{u, v, w};
    };
    Vector<kDims> const I        = P + t * PQ;
    Vector<3> const uvw          = BarycentricCoordinatesOf(I);
    bool const bIsInsideTriangle = (uvw.array() >= 0.).all() and (uvw.array() <= 1.).all();
    if (!bIsInsideTriangle)
        return {};

    return uvw;
}

template <
    class TDerivedA1,
    class TDerivedB1,
    class TDerivedC1,
    class TDerivedA2,
    class TDerivedB2,
    class TDerivedC2>
std::array<std::optional<Vector<3>>, 6u> UvwTriangles3D(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2)
{
    auto constexpr Rows  = TDerivedA1::RowsAtCompileTime;
    auto constexpr kDims = 3;
    static_assert(Rows == kDims, "This overlap test is specialized for 3D");

    Vector<kDims> const n1           = (B1 - A1).cross(C1 - A1).normalized();
    Vector<kDims> const n2           = (B2 - A2).cross(C2 - A2).normalized();
    auto constexpr eps               = 1e-15;
    bool const bAreTrianglesCoplanar = (1. - std::abs(n1.dot(n2))) < eps();
    if (bAreTrianglesCoplanar)
    {
        // NOTE: If triangles are coplanar and vertex of one triangle is in the plane of the other
        // triangle, then they are intersecting. We do not handle this case for now.
        return {};
    }
    // NOTE: Maybe handle degenerate triangles? Or maybe we should require the user to check
    // that.
    // - For a colinear triangle, we can perform line segment against triangle test.
    // - For 2 colinear triangles, we can perform line segment against line segment.
    // - For a triangle collapsed to a point, we perform point in triangle test.
    // - FOr 2 triangles collapsed to a point, we check if both points are numerically the same.

    // Test 3 edges of each triangle against the other triangle
    std::array<std::optional<Vector<3>>, 6u> intersections{
        UvwLineSegmentTriangle3D(A1, B1, A2, B2, C2),
        UvwLineSegmentTriangle3D(B1, C1, A2, B2, C2),
        UvwLineSegmentTriangle3D(C1, A1, A2, B2, C2),
        UvwLineSegmentTriangle3D(A2, B2, A1, B1, C1),
        UvwLineSegmentTriangle3D(B2, C2, A1, B1, C1),
        UvwLineSegmentTriangle3D(C2, A2, A1, B1, C1)};
    return intersections;
}

} // namespace IntersectionQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_INTERSECTION_QUERIES_H
