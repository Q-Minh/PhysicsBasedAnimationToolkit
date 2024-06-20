#ifndef PBAT_GEOMETRY_OVERLAP_QUERIES_H
#define PBAT_GEOMETRY_OVERLAP_QUERIES_H

#include "ClosestPointQueries.h"
#include "IntersectionQueries.h"

#include <Eigen/Geometry>
#include <cmath>
#include <pbat/Aliases.h>

namespace pbat {
namespace geometry {
namespace OverlapQueries {

/**
 * @brief Checks if point P is contained in tetrahedron ABCD, in at least 3D.
 * @param P
 * @param A
 * @param B
 * @param C
 * @param D
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
bool PointTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D);

/**
 * @brief Tests for overlap between sphere (C1,R1) and sphere (C2,R2).
 * @param c1
 * @param r1
 * @param c2
 * @param r2
 * @return
 */
template <class TDerivedC1, class TDerivedC2>
bool Spheres(
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Scalar R1,
    Eigen::MatrixBase<TDerivedC2> const& C2,
    Scalar R2);

/**
 * @brief Tests for overlap between axis-aligned bounding box (L1,U1) and axis-aligned
 * bounding box (L2,U2)
 * @param L1
 * @param U1
 * @param L2
 * @param U2
 * @return
 */
template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
bool AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2);

/**
 * @brief Tests for overlap between sphere (c,r) and axis-aligned bounding box (low,up)
 * @param C
 * @param R
 * @param L
 * @param U
 * @return
 */
template <class TDerivedC, class TDerivedL, class TDerivedU>
bool SphereAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief
 * @param P
 * @param Q
 * @param C
 * @param R
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedC>
bool LineSegmentSphere(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R);

/**
 * @brief
 * @param P
 * @param Q
 * @param L
 * @param U
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedL, class TDerivedU>
bool LineSegmentAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief Detects if the line segment PQ passes through the triangle ABC, in 3D.
 * @param P
 * @param Q
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
bool LineSegmentTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Tests for overlap between plane (P,n) and axis-aligned bounding box (low,up)
 * @param P
 * @param n
 * @param L
 * @param U
 * @return
 */
template <class TDerivedP, class TDerivedn, class TDerivedL, class TDerivedU>
bool PlaneAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedn> const& n,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief Tests for overlap between triangle ABC and axis-aligned bounding box (low,up)
 * @param A
 * @param B
 * @param C
 * @param L
 * @param U
 * @return
 */
template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedL, class TDerivedU>
bool TriangleAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief Tests for overlap between tetrahedron ABCD and axis-aligned bounding box (L,U), in at
 * least 3D.
 * @param A
 * @param B
 * @param C
 * @param D
 * @param L
 * @param U
 * @return
 */
template <
    class TDerivedA,
    class TDerivedB,
    class TDerivedC,
    class TDerivedD,
    class TDerivedL,
    class TDerivedU>
bool TetrahedronAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief Tests for overlap between triangle A1B1C1 and triangle A2B2C2, in 3D.
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
bool UvwTriangles(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2);

/**
 * @brief Tests for overlap between triangle ABC and tetrahedron IJKL, in at least 3D.
 * @param A
 * @param B
 * @param C
 * @param I
 * @param J
 * @param K
 * @param L
 * @return
 */
template <
    class TDerivedA,
    class TDerivedB,
    class TDerivedC,
    class TDerivedI,
    class TDerivedJ,
    class TDerivedK,
    class TDerivedL>
bool TriangleTetrahedron(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedI> const& I,
    Eigen::MatrixBase<TDerivedJ> const& J,
    Eigen::MatrixBase<TDerivedK> const& K,
    Eigen::MatrixBase<TDerivedL> const& L);

/**
 * @brief Tests for overlap between tetrahedron A1B1C1D1 and tetrahedron A2B2C2D2, in at least 3D.
 * @param A1
 * @param B1
 * @param C1
 * @param D1
 * @param A2
 * @param B2
 * @param C2
 * @param D2
 * @return
 */
template <
    class TDerivedA1,
    class TDerivedB1,
    class TDerivedC1,
    class TDerivedD1,
    class TDerivedA2,
    class TDerivedB2,
    class TDerivedC2,
    class TDerivedD2>
bool Tetrahedra(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedD1> const& D1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2,
    Eigen::MatrixBase<TDerivedD2> const& D2);

/**
 * @brief Tests for overlap between a triangle ABC and a sphere with center C of radius R
 * @param A
 * @param B
 * @param C
 * @param c
 * @param r
 * @return
 */
template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedSC>
bool TriangleSphere(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedSC> const& SC,
    Scalar R);

/**
 * @brief Tests for overlap between a tetrahedron ABCD and a sphere with center C of radius R
 * @param A
 * @param B
 * @param C
 * @param D
 * @param c
 * @param r
 * @return
 */
template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD, class TDerivedSC>
bool TetrahedronSphere(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D,
    Eigen::MatrixBase<TDerivedSC> const& SC,
    Scalar R);

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
bool PointTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D)
{    
    auto const PointOutsidePlane = [](auto const& p, auto const& a, auto const& b, auto const& c) {
        Scalar const d = (p - a).dot((b - a).cross(c - a));
        return d > 0.;
    };
    if (PointOutsidePlane(P, A, B, D))
        return false;
    if (PointOutsidePlane(P, B, C, D))
        return false;
    if (PointOutsidePlane(P, C, A, D))
        return false;
    if (PointOutsidePlane(P, A, C, B))
        return false;
    return true;
}

template <class TDerivedC1, class TDerivedC2>
bool Spheres(
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Scalar R1,
    Eigen::MatrixBase<TDerivedC2> const& C2,
    Scalar R2)
{
    Scalar const upper  = R1 + R2;
    Scalar const upper2 = upper * upper;
    Scalar const d2     = (C1 - C2).squaredNorm();
    return d2 <= upper2;
}

template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
bool AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2)
{
    return (L1.array() <= U2.array()).all() && (L2.array() <= U1.array()).all();
}

template <class TDerivedC, class TDerivedL, class TDerivedU>
bool SphereAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    auto const Xaabb = ClosestPointQueries::PointOnAxisAlignedBoundingBox(C, L, U);
    auto const d2    = (C - Xaabb).squaredNorm();
    auto const r2    = R * R;
    return d2 < r2;
}

template <class TDerivedP, class TDerivedQ, class TDerivedC>
bool LineSegmentSphere(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedC> const& C,
    Scalar R)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    Vector<Rows> const d = Q - P;
    Vector<Rows> const m = P - C;
    Scalar const b       = m.dot(d);
    Scalar const c       = m.dot(m) - R * R;
    // Exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > 0. && b > 0.)
        return false;
    Scalar const discr = b * b - c;
    // A negative discriminant corresponds to ray missing sphere
    if (discr < 0.)
        return false;
    // Ray now found to intersect sphere, compute smallest t value of intersection
    Scalar t = -b - std::sqrt(discr);
    if (t > 1.)
        return false;
    return true;
}

template <class TDerivedP, class TDerivedQ, class TDerivedL, class TDerivedU>
bool LineSegmentAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    Vector<Rows> const c = 0.5 * (L + U);
    Vector<Rows> const e = U - L;
    Vector<Rows> const d = Q - P;
    Vector<Rows> m       = P + Q - L - U;
    m                    = m - c; // Translate box and segment to origin

    // Try world coordinate axes as separating axes
    auto const dims = c.rows();
    Vector<Rows> ad{};
    // Should not happen, hopefully...
    if constexpr (Rows == Eigen::Dynamic)
        ad.resize(dims);

    for (auto dim = 0; dim < dims; ++dim)
    {
        ad(dim) = std::abs(d(dim));
        if (std::abs(m(dim)) > e(dim) + ad(dim))
            return false;
    }
    // Add in an epsilon term to counteract arithmetic errors when segment is
    // (near) parallel to a coordinate axis (see text for detail)
    Scalar constexpr eps = 1e-15;
    for (auto dim = 0; dim < dims; ++dim)
    {
        ad(dim) += eps;
        auto i = (dim + 1) % dims;
        auto j = (dim + 2) % dims;
        // Try cross products of segment direction vector with coordinate axes
        if (std::abs(m(i) * d(i) - m(i) * d(i)) > e(i) * ad(j) + e(j) * ad(i))
            return false;
    }
    // No separating axis found; segment must be overlapping AABB
    return true;
}

template <class TDerivedP, class TDerivedQ, class TDerivedA, class TDerivedB, class TDerivedC>
bool LineSegmentTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    return IntersectionQueries::UvwLineSegmentTriangle(P, Q, A, B, C).has_value();
}

template <class TDerivedP, class TDerivedn, class TDerivedL, class TDerivedU>
bool PlaneAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedn> const& n,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    auto constexpr Rows  = TDerivedP::RowsAtCompileTime;
    Vector<Rows> const C = 0.5 * (L + U); // Compute AABB center
    Vector<Rows> const e = U - C;         // Compute positive extents
    // Compute the projection interval radius of b onto L(t) = C + t * n
    Scalar const r = (e.array() * n.array().abs()).sum();
    // Compute distance of box center from plane
    Scalar const s = n.dot(C - P);
    // Intersection occurs when distance s falls within [-r,+r] interval
    return std::abs(s) <= r;
}

template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedL, class TDerivedU>
bool TriangleAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.2.9
     */

    auto constexpr Rows = TDerivedL::RowsAtCompileTime;
    // Transform triangle into reference space of AABB
    Vector<Rows> const O  = 0.5 * (L + U);
    Vector<Rows> const e  = U - O;
    Vector<Rows> const AO = A - O;
    Vector<Rows> const BO = B - O;
    Vector<Rows> const CO = C - O;

    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of triangle edges and 3 perpendicular AABB edges
     * - Face normals of AABB
     * - Face normal of triangle
     */

    auto const ProjectTriangle = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<3> const p{AO.dot(a), BO.dot(a), CO.dot(a)};
        return std::make_pair(std::min({p(0), p(1), p(2)}), std::max({p(0), p(1), p(2)}));
    };
    auto const ProjectAabb = [&](auto const& a) -> Scalar {
        return (e.array() * a.array().abs()).sum();
    };
    auto const AreDisjoint = [](Scalar ABCprojlow, Scalar ABCprojup, Scalar AABBproj) {
        return (AABBproj < ABCprojlow) || (ABCprojup < -AABBproj);
    };
    auto const TestAxis = [&ProjectTriangle, &ProjectAabb, &AreDisjoint](auto const& axis) {
        auto const [ABCmin, ABCmax] = ProjectTriangle(axis);
        Scalar const r              = ProjectAabb(axis);
        return AreDisjoint(ABCmin, ABCmax, r);
    };

    // 1. Test edge pairs
    auto constexpr eps = 1e-15;
    auto const IsEdgePairIntersecting =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            auto const ab = b - a;
            auto axis     = ab.cross(d - c).normalized();
            if (!axis.isZero(eps))
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = ab.cross(c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis = ab.cross(n).normalized();
                if (!axis.isZero(eps))
                    return TestAxis(axis);
                // ab and ac parallel too, so edges ab and cd are colinear and will not be a
                // separating axis
            }
            return false;
        };
    /**
     * Our implementation is super inefficient for AABBs, because of all the known zeros that we
     * are not exploiting. Because AABBs are axis aligned, their edges have many zeros, and
     * cross product and dot product operations with these edges could save many floating point
     * operations, but we don't do it... Fortunately, this implementation is valid for OBBs.
     */
    auto const dims = L.rows();
    Vector<Rows> zero;
    if constexpr (Rows == Eigen::Dynamic)
        zero.setZero(dims);
    else
        zero.setZero();

    for (auto dim = 0; dim < dims; ++dim)
    {
        // Construct natural unit vector in axis dim
        Vector<Rows> u;
        if constexpr (Rows == Eigen::Dynamic)
            u.setZero(dims);
        else
            u.setZero();
        u(dim) = 1.;

        if (IsEdgePairIntersecting(AO, BO, zero, u))
            return false;
        if (IsEdgePairIntersecting(BO, CO, zero, u))
            return false;
        if (IsEdgePairIntersecting(CO, AO, zero, u))
            return false;
    }

    // 2. Test AABB face normals
    for (auto dim = 0; dim < dims; ++dim)
    {
        Scalar const max = std::max({AO(dim), BO(dim), CO(dim)});
        Scalar const min = std::min({AO(dim), BO(dim), CO(dim)});
        if (max < -e(dim) || min > e(dim))
            return false;
    }

    // 3. Test triangle face normal
    Vector<Rows> const n = (B - A).cross(C - A).normalized();
    return PlaneAxisAlignedBoundingBox(A, n, L, U);
}

template <
    class TDerivedA,
    class TDerivedB,
    class TDerivedC,
    class TDerivedD,
    class TDerivedL,
    class TDerivedU>
bool TetrahedronAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    auto constexpr Rows = TDerivedL::RowsAtCompileTime;
    // Transform tetrahedron into reference space of AABB
    Vector<Rows> const O  = 0.5 * (L + U);
    Vector<Rows> const e  = U - O;
    Vector<Rows> const AO = A - O;
    Vector<Rows> const BO = B - O;
    Vector<Rows> const CO = C - O;
    Vector<Rows> const DO = D - O;

    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 6 tetrahedron edges and 3 perpendicular AABB edges (18
     * tests)
     * - Face normals of AABB (3 tests)
     * - Face normals of tetrahedron (4 tests)
     */

    auto const ProjectTetrahedron = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<4> const p{AO.dot(a), BO.dot(a), CO.dot(a), DO.dot(a)};
        return std::make_pair(
            std::min({p(0), p(1), p(2), p(3)}),
            std::max({p(0), p(1), p(2), p(3)}));
    };
    auto const ProjectAabb = [&](auto const& a) -> Scalar {
        return (e.array() * a.array().abs()).sum();
    };
    auto const AreDisjoint = [](Scalar low, Scalar up, Scalar r) {
        return (up < -r) || (r < low);
    };
    auto const TestAxis = [&ProjectTetrahedron, &ProjectAabb, &AreDisjoint](auto const& axis) {
        auto const [low, up] = ProjectTetrahedron(axis);
        Scalar const r       = ProjectAabb(axis);
        return AreDisjoint(low, up, r);
    };

    // 1. Test edge pairs
    auto constexpr eps = 1e-15;
    auto const IsEdgePairIntersecting =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            auto const ab = b - a;
            auto axis     = ab.cross(d - c).normalized();
            if (!axis.isZero(eps))
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = ab.cross(c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis = ab.cross(n).normalized();
                if (!axis.isZero(eps))
                    return TestAxis(axis);
                // ab and ac parallel too, so edges ab and cd are colinear and will not be a
                // separating axis
            }
            return false;
        };
    /**
     * Our implementation is super inefficient for AABBs, because of all the known zeros that we
     * are not exploiting. Because AABBs are axis aligned, their edges have many zeros, and
     * cross product and dot product operations with these edges could save many floating point
     * operations, but we don't do it... Fortunately, this implementation is valid for OBBs.
     */
    auto const dims = L.rows();
    Vector<Rows> zero;
    if constexpr (Rows == Eigen::Dynamic)
        zero.setZero(dims);
    else
        zero.setZero();

    for (auto dim = 0; dim < dims; ++dim)
    {
        // Construct natural unit vector in axis dim
        Vector<Rows> u;
        if constexpr (Rows == Eigen::Dynamic)
            u.setZero(dims);
        else
            u.setZero();
        u(dim) = 1.;

        // Edges of tetrahedron are: AB, BC, CA, AD, BD, CD
        if (IsEdgePairIntersecting(A, B, zero, u))
            return false;

        if (IsEdgePairIntersecting(B, C, zero, u))
            return false;

        if (IsEdgePairIntersecting(C, A, zero, u))
            return false;

        if (IsEdgePairIntersecting(A, D, zero, u))
            return false;

        if (IsEdgePairIntersecting(B, D, zero, u))
            return false;

        if (IsEdgePairIntersecting(C, D, zero, u))
            return false;
    }

    // 2. Test AABB face normals
    for (auto dim = 0; dim < dims; ++dim)
    {
        Scalar const max = std::max({AO(dim), BO(dim), CO(dim), DO(dim)});
        Scalar const min = std::min({AO(dim), BO(dim), CO(dim), DO(dim)});
        if (max < -e(dim) || min > e(dim))
            return false;
    }

    // 3. Test tetrahedron face normals
    // Tetrahedron faces are: ABD, BCD, CAD, ACB
    Vector<Rows> n = (B - A).cross(D - A).normalized();
    if (!PlaneAxisAlignedBoundingBox(A, n, L, U))
        return false;
    n = (C - B).cross(D - B).normalized();
    if (!PlaneAxisAlignedBoundingBox(B, n, L, U))
        return false;
    n = (A - C).cross(D - C).normalized();
    if (!PlaneAxisAlignedBoundingBox(C, n, L, U))
        return false;
    n = (C - A).cross(B - A).normalized();
    return PlaneAxisAlignedBoundingBox(A, n, L, U);
}

template <
    class TDerivedA1,
    class TDerivedB1,
    class TDerivedC1,
    class TDerivedA2,
    class TDerivedB2,
    class TDerivedC2>
bool UvwTriangles(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2)
{
    auto const intersections = IntersectionQueries::UvwTriangles(A1, B1, C1, A2, B2, C2);
    for (auto const& intersection : intersections)
        if (intersection.has_value())
            return true;
    return false;
}

template <
    class TDerivedA,
    class TDerivedB,
    class TDerivedC,
    class TDerivedI,
    class TDerivedJ,
    class TDerivedK,
    class TDerivedL>
bool TriangleTetrahedron(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedI> const& I,
    Eigen::MatrixBase<TDerivedJ> const& J,
    Eigen::MatrixBase<TDerivedK> const& K,
    Eigen::MatrixBase<TDerivedL> const& L)
{
    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 3 triangle edges and 6 tetrahedron edges (18
     * tests)
     * - Face normals of tetrahedron (4 tests)
     * - Face normal of triangle (1 test)
     */
    auto constexpr Rows = TDerivedA::RowsAtCompileTime;

    // 1. Test edge pairs
    auto const ProjectTriangle = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<3> const p{A.dot(a), B.dot(a), C.dot(a)};
        return std::make_pair(std::min({p(0), p(1), p(2)}), std::max({p(0), p(1), p(2)}));
    };
    auto const ProjectTetrahedron = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<4> const p{I.dot(a), J.dot(a), K.dot(a), L.dot(a)};
        return std::make_pair(
            std::min({p(0), p(1), p(2), p(3)}),
            std::max({p(0), p(1), p(2), p(3)}));
    };
    auto const AreDisjoint = [](Scalar low1, Scalar up1, Scalar low2, Scalar up2) {
        return (up1 < low2) || (up2 < low1);
    };
    auto const TestAxis = [&ProjectTriangle, &ProjectTetrahedron, &AreDisjoint](auto const& a) {
        auto const [low1, up1] = ProjectTriangle(a);
        auto const [low2, up2] = ProjectTetrahedron(a);
        return AreDisjoint(low1, up1, low2, up2);
    };
    auto constexpr eps = 1e-15;
    auto const IsEdgePairSeparating =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            auto const ab     = b - a;
            Vector<Rows> axis = ab.cross(d - c).normalized();
            if (!axis.isZero(eps))
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = ab.cross(c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis = ab.cross(n).normalized();
                if (!axis.isZero(eps))
                    return TestAxis(axis);
                // ab and ac parallel too, so edges ab and cd are colinear and will not be a
                // separating axis
            }
            return false;
        };

    // Tetrahedron edges are: IJ, JK, KI, IL, JL, KL
    // Triangle edges are: AB, BC, CA
    if (IsEdgePairSeparating(A, B, I, J))
        return false;
    if (IsEdgePairSeparating(B, C, I, J))
        return false;
    if (IsEdgePairSeparating(C, A, I, J))
        return false;

    if (IsEdgePairSeparating(A, B, J, K))
        return false;
    if (IsEdgePairSeparating(B, C, J, K))
        return false;
    if (IsEdgePairSeparating(C, A, J, K))
        return false;

    if (IsEdgePairSeparating(A, B, K, I))
        return false;
    if (IsEdgePairSeparating(B, C, K, I))
        return false;
    if (IsEdgePairSeparating(C, A, K, I))
        return false;

    if (IsEdgePairSeparating(A, B, I, L))
        return false;
    if (IsEdgePairSeparating(B, C, I, L))
        return false;
    if (IsEdgePairSeparating(C, A, I, L))
        return false;

    if (IsEdgePairSeparating(A, B, J, L))
        return false;
    if (IsEdgePairSeparating(B, C, J, L))
        return false;
    if (IsEdgePairSeparating(C, A, J, L))
        return false;

    if (IsEdgePairSeparating(A, B, K, L))
        return false;
    if (IsEdgePairSeparating(B, C, K, L))
        return false;
    if (IsEdgePairSeparating(C, A, K, L))
        return false;

    // 2. Test tetrahedron face normals
    Vector<Rows> const IJ = J - I;
    Vector<Rows> const JK = K - J;
    Vector<Rows> const KI = I - K;
    Vector<Rows> const IL = L - I;
    Vector<Rows> const JL = L - J;
    Vector<Rows> const KL = L - K;
    Vector<Rows> n        = IJ.cross(IL).normalized();
    if (TestAxis(n))
        return false;
    n = JK.cross(JL).normalized();
    if (TestAxis(n))
        return false;
    n = KI.cross(KL).normalized();
    if (TestAxis(n))
        return false;
    Vector<Rows> const IK = K - I;
    n                     = IK.cross(IJ).normalized();
    if (TestAxis(n))
        return false;

    // 3. Test triangle face normal
    n = (B - A).cross(C - A).normalized();
    return !TestAxis(n);
}

template <
    class TDerivedA1,
    class TDerivedB1,
    class TDerivedC1,
    class TDerivedD1,
    class TDerivedA2,
    class TDerivedB2,
    class TDerivedC2,
    class TDerivedD2>
bool Tetrahedra(
    Eigen::MatrixBase<TDerivedA1> const& A1,
    Eigen::MatrixBase<TDerivedB1> const& B1,
    Eigen::MatrixBase<TDerivedC1> const& C1,
    Eigen::MatrixBase<TDerivedD1> const& D1,
    Eigen::MatrixBase<TDerivedA2> const& A2,
    Eigen::MatrixBase<TDerivedB2> const& B2,
    Eigen::MatrixBase<TDerivedC2> const& C2,
    Eigen::MatrixBase<TDerivedD2> const& D2)
{
    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 6 tetrahedron A1B1C1D1 edges and 6 tetrahedron A2B2C2D2
     * edges (36 tests)
     * - Face normals of tetrahedron (4+4=8 tests)
     */
    auto constexpr Rows = TDerivedA1::RowsAtCompileTime;

    auto const ProjectTetrahedron1 = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<4> const p{A1.dot(a), B1.dot(a), C1.dot(a), D1.dot(a)};
        return std::make_pair(
            std::min({p(0), p(1), p(2), p(3)}),
            std::max({p(0), p(1), p(2), p(3)}));
    };
    auto const ProjectTetrahedron2 = [&](auto const& a) -> std::pair<Scalar, Scalar> {
        Vector<4> const p{A2.dot(a), B2.dot(a), C2.dot(a), D2.dot(a)};
        return std::make_pair(
            std::min({p(0), p(1), p(2), p(3)}),
            std::max({p(0), p(1), p(2), p(3)}));
    };
    auto const AreDisjoint = [](Scalar low1, Scalar up1, Scalar low2, Scalar up2) {
        return (up1 < low2) || (up2 < low1);
    };
    auto const TestAxis =
        [&ProjectTetrahedron1, &ProjectTetrahedron2, &AreDisjoint](auto const& a) {
            auto const [low1, up1] = ProjectTetrahedron1(a);
            auto const [low2, up2] = ProjectTetrahedron2(a);
            return AreDisjoint(low1, up1, low2, up2);
        };

    // 1. Test edge pairs
    auto constexpr eps = 1e-15;
    auto const IsEdgePairSeparating =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            auto const ab     = b - a;
            Vector<Rows> axis = ab.cross(d - c).normalized();
            if (!axis.isZero(eps))
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = ab.cross(c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis = ab.cross(n).normalized();
                if (!axis.isZero(eps))
                {
                    return TestAxis(axis);
                }
                // ab and ac parallel too, so edges ab and cd are colinear and will not be a
                // separating axis
            }
            return false;
        };

    // Tetrahedron 1 edges are: A1B1, B1C1, C1A1, A1D1, B1D1, C1D1
    // Tetrahedron 2 edges are: A2B2, B2C2, C2A2, A2D2, B2D2, C2D2
    if (IsEdgePairSeparating(A1, B1, A2, B2))
        return false;
    if (IsEdgePairSeparating(B1, C1, A2, B2))
        return false;
    if (IsEdgePairSeparating(C1, A1, A2, B2))
        return false;
    if (IsEdgePairSeparating(A1, D1, A2, B2))
        return false;
    if (IsEdgePairSeparating(B1, D1, A2, B2))
        return false;
    if (IsEdgePairSeparating(C1, D1, A2, B2))
        return false;

    if (IsEdgePairSeparating(A1, B1, B2, C2))
        return false;
    if (IsEdgePairSeparating(B1, C1, B2, C2))
        return false;
    if (IsEdgePairSeparating(C1, A1, B2, C2))
        return false;
    if (IsEdgePairSeparating(A1, D1, B2, C2))
        return false;
    if (IsEdgePairSeparating(B1, D1, B2, C2))
        return false;
    if (IsEdgePairSeparating(C1, D1, B2, C2))
        return false;

    if (IsEdgePairSeparating(A1, B1, C2, A2))
        return false;
    if (IsEdgePairSeparating(B1, C1, C2, A2))
        return false;
    if (IsEdgePairSeparating(C1, A1, C2, A2))
        return false;
    if (IsEdgePairSeparating(A1, D1, C2, A2))
        return false;
    if (IsEdgePairSeparating(B1, D1, C2, A2))
        return false;
    if (IsEdgePairSeparating(C1, D1, C2, A2))
        return false;

    if (IsEdgePairSeparating(A1, B1, A2, D2))
        return false;
    if (IsEdgePairSeparating(B1, C1, A2, D2))
        return false;
    if (IsEdgePairSeparating(C1, A1, A2, D2))
        return false;
    if (IsEdgePairSeparating(A1, D1, A2, D2))
        return false;
    if (IsEdgePairSeparating(B1, D1, A2, D2))
        return false;
    if (IsEdgePairSeparating(C1, D1, A2, D2))
        return false;

    if (IsEdgePairSeparating(A1, B1, B2, D2))
        return false;
    if (IsEdgePairSeparating(B1, C1, B2, D2))
        return false;
    if (IsEdgePairSeparating(C1, A1, B2, D2))
        return false;
    if (IsEdgePairSeparating(A1, D1, B2, D2))
        return false;
    if (IsEdgePairSeparating(B1, D1, B2, D2))
        return false;
    if (IsEdgePairSeparating(C1, D1, B2, D2))
        return false;

    if (IsEdgePairSeparating(A1, B1, C2, D2))
        return false;
    if (IsEdgePairSeparating(B1, C1, C2, D2))
        return false;
    if (IsEdgePairSeparating(C1, A1, C2, D2))
        return false;
    if (IsEdgePairSeparating(A1, D1, C2, D2))
        return false;
    if (IsEdgePairSeparating(B1, D1, C2, D2))
        return false;
    if (IsEdgePairSeparating(C1, D1, C2, D2))
        return false;

    // 2. Test face normals:
    // Tetrahedron 1 faces are: A1B1D1, B1C1D1, C1A1D1, A1C1B1
    // Tetrahedron 2 faces are: A2B2D2, B2C2D2, C2A2D2, A2C2B2
    Vector<Rows> n = (B1 - A1).cross(D1 - A1).normalized();
    if (TestAxis(n))
        return false;
    n = (C1 - B1).cross(D1 - B1).normalized();
    if (TestAxis(n))
        return false;
    n = (A1 - C1).cross(D1 - C1).normalized();
    if (TestAxis(n))
        return false;
    n = (C1 - A1).cross(B1 - A1).normalized();
    if (TestAxis(n))
        return false;

    n = (B2 - A2).cross(D2 - A2).normalized();
    if (TestAxis(n))
        return false;
    n = (C2 - B2).cross(D2 - B2).normalized();
    if (TestAxis(n))
        return false;
    n = (A2 - C2).cross(D2 - C2).normalized();
    if (TestAxis(n))
        return false;
    n = (C2 - A2).cross(B2 - A2).normalized();
    return !TestAxis(n);
}

template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedSC>
bool TriangleSphere(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedSC> const& SC,
    Scalar R)
{
    auto const X    = ClosestPointQueries::PointInTriangle(SC, A, B, C);
    Scalar const d2 = (X - SC).squaredNorm();
    Scalar const r2 = R * R;
    return d2 < r2;
}

template <class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD, class TDerivedSC>
bool TetrahedronSphere(
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D,
    Eigen::MatrixBase<TDerivedSC> const& SC,
    Scalar R)
{
    auto const X    = ClosestPointQueries::PointInTetrahedron(SC, A, B, C, D);
    Scalar const d2 = (X - SC).squaredNorm();
    Scalar const r2 = R * R;
    return d2 < r2;
}

} // namespace OverlapQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_OVERLAP_QUERIES_H
