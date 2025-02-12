/**
 * @file OverlapQueries.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to answer overlap queries.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_OVERLAPQUERIES_H
#define PBAT_GEOMETRY_OVERLAPQUERIES_H

#include "ClosestPointQueries.h"
#include "IntersectionQueries.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <cmath>

namespace pbat {
namespace geometry {
/**
 * @brief This namespace contains functions to answer overlap queries.
 */
namespace OverlapQueries {

namespace mini = math::linalg::mini;

/**
 * @brief Tests for overlap between point P and axis-aligned bounding box (L,U)
 * @tparam TMatrixP Point matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param P Point
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if P is inside the box, false otherwise
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U);

/**
 * @brief Tests for overlap between point P and triangle ABC
 * @tparam TMatrixP Point matrix type
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @param P Point
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return True if P is inside the triangle, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C);

/**
 * @brief Checks if point P is contained in tetrahedron ABCD, in at least 3D.
 * @tparam TMatrixP Point matrix type
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @tparam TMatrixD Vertex D matrix type
 * @param P Point
 * @param A Vertex A of the tetrahedron
 * @param B Vertex B of the tetrahedron
 * @param C Vertex C of the tetrahedron
 * @param D Vertex D of the tetrahedron
 * @return True if P is inside the tetrahedron, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE bool PointTetrahedron3D(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D);

/**
 * @brief Tests for overlap between sphere (C1,R1) and sphere (C2,R2).
 * @tparam TMatrixC1 Center matrix type of the first sphere
 * @tparam TMatrixC2 Center matrix type of the second sphere
 * @param c1 Center of the first sphere
 * @param r1 Radius of the first sphere
 * @param c2 Center of the second sphere
 * @param r2 Radius of the second sphere
 * @return True if the spheres overlap, false otherwise
 */
template <mini::CMatrix TMatrixC1, mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Spheres(
    TMatrixC1 const& C1,
    typename TMatrixC1::ScalarType R1,
    TMatrixC2 const& C2,
    typename TMatrixC2::ScalarType R2);

/**
 * @brief Tests for overlap between axis-aligned bounding box (L1,U1) and axis-aligned
 * bounding box (L2,U2)
 * @tparam TMatrixL1 Matrix type of the lower bound of AABB 1
 * @tparam TMatrixU1 Matrix type of the upper bound of AABB 1
 * @tparam TMatrixL2 Matrix type of the lower bound of AABB 2
 * @tparam TMatrixU2 Matrix type of the upper bound of AABB 2
 * @param L1 The lower bound of AABB 1
 * @param U1 The upper bound of AABB 1
 * @param L2 The lower bound of AABB 2
 * @param U2 The upper bound of AABB 2
 * @return True if the AABBs overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE bool AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2);

/**
 * @brief Tests for overlap between sphere (c,r) and axis-aligned bounding box (low,up)
 * @tparam TMatrixC Center matrix type of the sphere
 * @tparam TMatrixL Lower corner matrix type of the box
 * @tparam TMatrixU Upper corner matrix type of the box
 * @param C Center of the sphere
 * @param R Radius of the sphere
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if the sphere and box overlap, false otherwise
 */
template <mini::CMatrix TMatrixC, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool SphereAxisAlignedBoundingBox(
    TMatrixC const& C,
    typename TMatrixC::ScalarType R,
    TMatrixL const& L,
    TMatrixU const& U);

/**
 * @brief Tests for overlap between line segment PQ and sphere (C,R)
 * @tparam TMatrixP Start point matrix type
 * @tparam TMatrixQ End point matrix type
 * @tparam TMatrixC Center matrix type
 * @param P Start point of the line segment
 * @param Q End point of the line segment
 * @param C Center of the sphere
 * @param R Radius of the sphere
 * @return True if the line segment and sphere overlap, false otherwise
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool LineSegmentSphere(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixC const& C,
    typename TMatrixC::ScalarType R);

/**
 * @brief Tests for overlap between line segment PQ and axis-aligned bounding box (L,U)
 * @tparam TMatrixP Start point matrix type
 * @tparam TMatrixQ End point matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param P Start point of the line segment
 * @param Q End point of the line segment
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if the line segment and box overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool LineSegmentAxisAlignedBoundingBox(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixL const& L,
    TMatrixU const& U);

/**
 * @brief Detects if the line segment PQ passes through the triangle ABC, in 3D.
 * @tparam TMatrixP Matrix type of the start point of the line segment
 * @tparam TMatrixQ Matrix type of the end point of the line segment
 * @tparam TMatrixA Matrix type of the vertex A of the triangle
 * @tparam TMatrixB Matrix type of the vertex B of the triangle
 * @tparam TMatrixC Matrix type of the vertex C of the triangle
 * @param P Start point of the line segment
 * @param Q End point of the line segment
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return True if the line segment passes through the triangle, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool LineSegmentTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C);

/**
 * @brief Tests overlap between line segment PQ and the swept volume spanned by linear interpolation
 * of A1B1C1 to A2B2C2
 *
 * @warning This function does not handle degeneracies yet. I don't suggest using it for now.
 *
 * @tparam TMatrixP Start point matrix type
 * @tparam TMatrixQ End point matrix type
 * @tparam TMatrixA1 Matrix type of the vertex A1 of triangle 1
 * @tparam TMatrixB1 Matrix type of the vertex B1 of triangle 1
 * @tparam TMatrixC1 Matrix type of the vertex C1 of triangle 1
 * @tparam TMatrixA2 Matrix type of the vertex A2 of triangle 2
 * @tparam TMatrixB2 Matrix type of the vertex B2 of triangle 2
 * @tparam TMatrixC2 Matrix type of the vertex C2 of triangle 2
 * @param P Start point of the line segment
 * @param Q End point of the line segment
 * @param A1 Vertex A of triangle 1
 * @param B1 Vertex B of triangle 1
 * @param C1 Vertex C of triangle 1
 * @param A2 Vertex A of triangle 2
 * @param B2 Vertex B of triangle 2
 * @param C2 Vertex C of triangle 2
 * @return True if the line segment passes through the swept volume, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool LineSegmentSweptTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2);

/**
 * @brief Tests for overlap between plane (P,n) and axis-aligned bounding box (low,up)
 * @tparam TMatrixP Point on the plane matrix type
 * @tparam TMatrixN Normal of the plane matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param P Point on the plane
 * @param n Normal of the plane
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if the plane and box overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixN,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool PlaneAxisAlignedBoundingBox(
    TMatrixP const& P,
    TMatrixN const& n,
    TMatrixL const& L,
    TMatrixU const& U);

/**
 * @brief Tests for overlap between triangle ABC and axis-aligned bounding box (low,up)
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if the triangle and box overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool TriangleAxisAlignedBoundingBox(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixL const& L,
    TMatrixU const& U);

/**
 * @brief Tests for overlap between tetrahedron ABCD and axis-aligned bounding box (L,U), in at
 * least 3D.
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @tparam TMatrixD Vertex D matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param A Vertex A of the tetrahedron
 * @param B Vertex B of the tetrahedron
 * @param C Vertex C of the tetrahedron
 * @param D Vertex D of the tetrahedron
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return True if the tetrahedron and box overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool TetrahedronAxisAlignedBoundingBox(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D,
    TMatrixL const& L,
    TMatrixU const& U);

/**
 * @brief Tests for overlap between triangle A1B1C1 and triangle A2B2C2, in 2D.
 * @tparam TMatrixA1 Matrix type of vertex A1 of triangle 1
 * @tparam TMatrixB1 Matrix type of vertex B1 of triangle 1
 * @tparam TMatrixC1 Matrix type of vertex C1 of triangle 1
 * @tparam TMatrixA2 Matrix type of vertex A2 of triangle 2
 * @tparam TMatrixB2 Matrix type of vertex B2 of triangle 2
 * @tparam TMatrixC2 Matrix type of vertex C2 of triangle 2
 * @param A1 Vertex A of triangle 1
 * @param B1 Vertex B of triangle 1
 * @param C1 Vertex C of triangle 1
 * @param A2 Vertex A of triangle 2
 * @param B2 Vertex B of triangle 2
 * @param C2 Vertex C of triangle 2
 * @return True if the triangles overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Triangles2D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2);

/**
 * @brief Tests for overlap between triangle A1B1C1 and triangle A2B2C2, in 3D.
 * @tparam TMatrixA1 Matrix type of vertex A1 of triangle 1
 * @tparam TMatrixB1 Matrix type of vertex B1 of triangle 1
 * @tparam TMatrixC1 Matrix type of vertex C1 of triangle 1
 * @tparam TMatrixA2 Matrix type of vertex A2 of triangle 2
 * @tparam TMatrixB2 Matrix type of vertex B2 of triangle 2
 * @tparam TMatrixC2 Matrix type of vertex C2 of triangle 2
 * @param A1 Vertex A of triangle 1
 * @param B1 Vertex B of triangle 1
 * @param C1 Vertex C of triangle 1
 * @param A2 Vertex A of triangle 2
 * @param B2 Vertex B of triangle 2
 * @param C2 Vertex C of triangle 2
 * @return True if the triangles overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Triangles3D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2);

/**
 * @brief Tests for overlap between triangle ABC and tetrahedron IJKL, in at least 3D.
 * @tparam TMatrixA Matrix type of vertex A of the triangle
 * @tparam TMatrixB Matrix type of vertex B of the triangle
 * @tparam TMatrixC Matrix type of vertex C of the triangle
 * @tparam TMatrixI Matrix type of vertex I of the tetrahedron
 * @tparam TMatrixJ Matrix type of vertex J of the tetrahedron
 * @tparam TMatrixK Matrix type of vertex K of the tetrahedron
 * @tparam TMatrixL Matrix type of vertex L of the tetrahedron
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @param I Vertex I of the tetrahedron
 * @param J Vertex J of the tetrahedron
 * @param K Vertex K of the tetrahedron
 * @param L Vertex L of the tetrahedron
 * @return True if the triangle and tetrahedron overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixI,
    mini::CMatrix TMatrixJ,
    mini::CMatrix TMatrixK,
    mini::CMatrix TMatrixL>
PBAT_HOST_DEVICE bool TriangleTetrahedron(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixI const& I,
    TMatrixJ const& J,
    TMatrixK const& K,
    TMatrixL const& L);

/**
 * @brief Tests for overlap between tetrahedron A1B1C1D1 and tetrahedron A2B2C2D2, in at least 3D.
 * @tparam TMatrixA1 Matrix type of vertex A1 of tetrahedron 1
 * @tparam TMatrixB1 Matrix type of vertex B1 of tetrahedron 1
 * @tparam TMatrixC1 Matrix type of vertex C1 of tetrahedron 1
 * @tparam TMatrixD1 Matrix type of vertex D1 of tetrahedron 1
 * @tparam TMatrixA2 Matrix type of vertex A2 of tetrahedron 2
 * @tparam TMatrixB2 Matrix type of vertex B2 of tetrahedron 2
 * @tparam TMatrixC2 Matrix type of vertex C2 of tetrahedron 2
 * @tparam TMatrixD2 Matrix type of vertex D2 of tetrahedron 2
 * @param A1 Vertex A of tetrahedron 1
 * @param B1 Vertex B of tetrahedron 1
 * @param C1 Vertex C of tetrahedron 1
 * @param D1 Vertex D of tetrahedron 1
 * @param A2 Vertex A of tetrahedron 2
 * @param B2 Vertex B of tetrahedron 2
 * @param C2 Vertex C of tetrahedron 2
 * @param D2 Vertex D of tetrahedron 2
 * @return True if the tetrahedra overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixD1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2,
    mini::CMatrix TMatrixD2>
PBAT_HOST_DEVICE bool Tetrahedra(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixD1 const& D1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2,
    TMatrixD2 const& D2);

/**
 * @brief Tests for overlap between a triangle ABC and a sphere with center c of radius R
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @tparam TMatrixSC Center matrix type of the sphere
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @param c Center of the sphere
 * @param r Radius of the sphere
 * @return True if the triangle and sphere overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixSC>
PBAT_HOST_DEVICE bool TriangleSphere(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixSC const& SC,
    typename TMatrixSC::ScalarType R);

/**
 * @brief Tests for overlap between a tetrahedron ABCD and a sphere with center c of radius R
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @tparam TMatrixD Vertex D matrix type
 * @tparam TMatrixSC Center matrix type of the sphere
 * @param A Vertex A of the tetrahedron
 * @param B Vertex B of the tetrahedron
 * @param C Vertex C of the tetrahedron
 * @param D Vertex D of the tetrahedron
 * @param c Center of the sphere
 * @param r Radius of the sphere
 * @return True if the tetrahedron and sphere overlap, false otherwise
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD,
    mini::CMatrix TMatrixSC>
PBAT_HOST_DEVICE bool TetrahedronSphere(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D,
    TMatrixSC const& SC,
    typename TMatrixSC::ScalarType R);

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U)
{
    bool bIsOutsideBox = Any((P < L) or (P > U));
    return not bIsOutsideBox;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
{
    auto uvw               = IntersectionQueries::TriangleBarycentricCoordinates(P, A, B, C);
    using ScalarType       = typename TMatrixP::ScalarType;
    bool bIsInsideTriangle = All((uvw >= ScalarType(0)) and (uvw <= ScalarType(1)));
    return bIsInsideTriangle;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE bool PointTetrahedron3D(
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

    auto const PointOutsidePlane = [](auto const& p, auto const& a, auto const& b, auto const& c) {
        ScalarType const d = Dot(p - a, Cross(b - a, c - a));
        return d > ScalarType(0);
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

template <mini::CMatrix TMatrixC1, mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Spheres(
    TMatrixC1 const& C1,
    typename TMatrixC1::ScalarType R1,
    TMatrixC2 const& C2,
    typename TMatrixC2::ScalarType R2)
{
    using ScalarType        = typename TMatrixC1::ScalarType;
    ScalarType const upper  = R1 + R2;
    ScalarType const upper2 = upper * upper;
    ScalarType const d2     = SquaredNorm(C1 - C2);
    return d2 <= upper2;
}

template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE bool AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2)
{
    bool bOverlap = All((L1 <= U2) and (L2 <= U1));
    return bOverlap;
}

template <mini::CMatrix TMatrixC, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool SphereAxisAlignedBoundingBox(
    TMatrixC const& C,
    typename TMatrixC::ScalarType R,
    TMatrixL const& L,
    TMatrixU const& U)
{
    auto const Xaabb = ClosestPointQueries::PointOnAxisAlignedBoundingBox(C, L, U);
    auto const d2    = SquaredNorm(C - Xaabb);
    auto const r2    = R * R;
    return d2 < r2;
}

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixQ, mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool LineSegmentSphere(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixC const& C,
    typename TMatrixC::ScalarType R)
{
    using ScalarType                         = typename TMatrixC::ScalarType;
    auto constexpr kRows                     = TMatrixP::kRows;
    mini::SVector<ScalarType, kRows> const d = Q - P;
    mini::SVector<ScalarType, kRows> const m = P - C;
    ScalarType const b                       = Dot(m, d);
    ScalarType const c                       = Dot(m, m) - R * R;
    // Exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > ScalarType(0) and b > ScalarType(0))
        return false;
    ScalarType const discr = b * b - c;
    // A negative discriminant corresponds to ray missing sphere
    if (discr < ScalarType(0))
        return false;
    // Ray now found to intersect sphere, compute smallest t value of intersection
    using namespace std;
    ScalarType t = -b - sqrt(discr);
    if (t > ScalarType(1))
        return false;
    return true;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool LineSegmentAxisAlignedBoundingBox(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixL const& L,
    TMatrixU const& U)
{
    using ScalarType     = typename TMatrixP::ScalarType;
    auto constexpr kRows = TMatrixP::kRows;

    mini::SVector<ScalarType, kRows> const c = ScalarType(0.5) * (L + U);
    mini::SVector<ScalarType, kRows> const e = U - L;
    mini::SVector<ScalarType, kRows> const d = Q - P;
    mini::SVector<ScalarType, kRows> m       = P + Q - L - U;
    m                                        = m - c; // Translate box and segment to origin

    // Try world coordinate axes as separating axes
    using namespace std;
    auto constexpr kDims = c.Rows();
    bool bAxesSeparating = Any(Abs(m) > (e + Abs(d)));
    if (bAxesSeparating)
        return false;
    // Add in an epsilon term to counteract arithmetic errors when segment is
    // (near) parallel to a coordinate axis (see text for detail)
    common::ForRange<0, kDims>([&]<auto kDim>() {
        ScalarType constexpr eps{1e-15};
        ad(kDim) += eps;
        auto i = (kDim + 1) % kDims;
        auto j = (kDim + 2) % kDims;
        // Try cross products of segment direction vector with coordinate axes
        bAxesSeparating &= abs(m(i) * d(i) - m(i) * d(i)) > e(i) * ad(j) + e(j) * ad(i);
    });
    // No separating axis found; segment must be overlapping AABB
    return not bAxesSeparating;
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE bool LineSegmentTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C)
{
    return IntersectionQueries::UvwLineSegmentTriangle3D(P, Q, A, B, C).has_value();
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixQ,
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool LineSegmentSweptTriangle3D(
    TMatrixP const& P,
    TMatrixQ const& Q,
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2)
{
    // We can construct the swept volume by intersecting the following half-planes:
    // 1. Plane spanned by A1B1C1
    // 2. Planed spanned by A2C2B2
    // 3. Plane spanned by A1A2B2B1 -> 2 triangles A1A2B2 and A1B2B1
    // 4. Plane spanned by B1B2C2C1 -> 2 triangles B1B2C2 and B1C2C1
    // 5. Plane spanned by C1C2A2A1 -> 2 triangles C1C2A2 and C1A2A1

    using ScalarType            = typename TMatrixP::ScalarType;
    static auto constexpr kDims = 3;

    auto const fSignWrtPlane = [=](auto const& X, auto const& A, auto const& B, auto const& C) {
        mini::SVector<ScalarType, kDims> const n = Cross(B - A, C - A);
        auto const d                             = Dot(X - A, n);
        return d;
    };

    // Check if triangles are coplanar
    bool const bCoplanar = fSignWrtPlane(A1, A2, B2, C2) == ScalarType(0) and
                           fSignWrtPlane(B1, A2, B2, C2) == ScalarType(0) and
                           fSignWrtPlane(C1, A2, B2, C2) == ScalarType(0);
    if (bCoplanar)
    {
        // WARNING:
        // We are not handling this correctly... Ideally, we would form a polygon on the plane from
        // the swept triangles, and intersect our line segment with that polygon. For now, let's
        // just check for intersection between the line segment and both triangles.
        return LineSegmentTriangle3D(P, Q, A1, B1, C1) or LineSegmentTriangle3D(P, Q, A2, C2, B2);
    }

    // Linearly swept triangle is a convex shape, check for PQ inside the volume.
    // Because the triangle orientation can be inverted, we need to check for 2 cases:
    // 1. All negative signs
    // 2. All positive signs
    // We do this by simply counting the number of negative signs and checking the count.
    // clang-format off
    mini::SVector<ScalarType, 10> sgn{};
    sgn(0) = fSignWrtPlane(P, A1, B1, C1);
    sgn(1) = fSignWrtPlane(P, A2, C2, B2);
    sgn(2) = fSignWrtPlane(P, A1, A2, B2);
    sgn(3) = fSignWrtPlane(P, B1, B2, C2);
    sgn(4) = fSignWrtPlane(P, C1, C2, A2);
    sgn(5) = fSignWrtPlane(Q, A1, B1, C1);
    sgn(6) = fSignWrtPlane(Q, A2, C2, B2);
    sgn(7) = fSignWrtPlane(Q, A1, A2, B2);
    sgn(8) = fSignWrtPlane(Q, B1, B2, C2);
    sgn(9) = fSignWrtPlane(Q, C1, C2, A2);
    bool const bIsPqInside = All(sgn <= ScalarType(0)) or All(sgn >= ScalarType(0));
    // clang-format on
    if (bIsPqInside)
        return true;

    // Check for intersection of PQ with each triangle on the boundary of the swept volume.
    // clang-format off
    return LineSegmentTriangle3D(P, Q, A1, B1, C1) or 
           LineSegmentTriangle3D(P, Q, A2, C2, B2) or 
           LineSegmentTriangle3D(P, Q, A1, A2, B2) or 
           LineSegmentTriangle3D(P, Q, A1, B2, B1) or 
           LineSegmentTriangle3D(P, Q, B1, B2, C2) or 
           LineSegmentTriangle3D(P, Q, B1, C2, C1) or 
           LineSegmentTriangle3D(P, Q, C1, C2, A2) or 
           LineSegmentTriangle3D(P, Q, C1, A2, A1);
    // clang-format on
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixN,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool PlaneAxisAlignedBoundingBox(
    TMatrixP const& P,
    TMatrixN const& n,
    TMatrixL const& L,
    TMatrixU const& U)
{
    using ScalarType                         = typename TMatrixP::ScalarType;
    auto constexpr kRows                     = TMatrixP::kRows;
    mini::SVector<ScalarType, kRows> const C = ScalarType(0.5) * (L + U); // Compute AABB center
    mini::SVector<ScalarType, kRows> const e = U - C; // Compute positive extents
    // Compute the projection interval radius of b onto L(t) = C + t * n
    ScalarType const r = Dot(e, Abs(n));
    // Compute distance of box center from plane
    ScalarType const s = Dot(n, C - P);
    // Intersection occurs when distance s falls within [-r,+r] interval
    using namespace std;
    return abs(s) <= r;
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool TriangleAxisAlignedBoundingBox(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixL const& L,
    TMatrixU const& U)
{
    /**
     * See \cite ericson2004real section 5.2.9
     */

    using ScalarType     = typename TMatrixA::ScalarType;
    auto constexpr kRows = TMatrixL::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");
    // Transform triangle into reference space of AABB
    mini::SVector<ScalarType, kDims> const O  = ScalarType(0.5) * (L + U);
    mini::SVector<ScalarType, kDims> const e  = U - O;
    mini::SVector<ScalarType, kDims> const AO = A - O;
    mini::SVector<ScalarType, kDims> const BO = B - O;
    mini::SVector<ScalarType, kDims> const CO = C - O;

    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of triangle edges and 3 perpendicular AABB edges
     * - Face normals of AABB
     * - Face normal of triangle
     */

    using namespace std;
    auto const ProjectTriangle = [&](auto const& a) -> std::pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 3> const p{Dot(AO, a), Dot(BO, a), Dot(CO, a)};
        return make_pair(Min(p), Max(p));
    };
    auto const ProjectAabb = [&](auto const& a) -> ScalarType {
        return Dot(e, Abs(a));
    };
    auto const AreDisjoint = [](ScalarType ABCprojlow, ScalarType ABCprojup, ScalarType AABBproj) {
        return (AABBproj < ABCprojlow) or (ABCprojup < -AABBproj);
    };
    auto const TestAxis = [&ProjectTriangle, &ProjectAabb, &AreDisjoint](auto const& axis) {
        auto const [ABCmin, ABCmax] = ProjectTriangle(axis);
        ScalarType const r          = ProjectAabb(axis);
        return AreDisjoint(ABCmin, ABCmax, r);
    };

    // ScalarType(1) Test edge pairs
    auto const IsEdgePairIntersecting = [&TestAxis](auto const& a, auto const& b, auto dim) {
        ScalarType constexpr eps = 1e-15;
        auto const ab            = b - a;
        // Construct natural unit vector in axis dim
        auto const u                          = mini::Unit<ScalarType, kDims>(dim);
        mini::SVector<ScalarType, kDims> axis = Normalized(Cross(ab, u /* - zero*/));
        bool bAxisIsZero                      = All(Abs(axis) <= eps);
        if (not bAxisIsZero)
        {
            return TestAxis(axis);
        }
        else
        {
            // Edges ab and cd are numerically parallel
            auto const n = Cross(ab, /*zero */ -a);
            // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
            axis        = Normalized(Cross(ab, n));
            bAxisIsZero = All(Abs(axis) <= eps);
            if (not bAxisIsZero)
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
    for (auto dim = 0; dim < kDims; ++dim)
    {
        if (IsEdgePairIntersecting(AO, BO, dim))
            return false;
        if (IsEdgePairIntersecting(BO, CO, dim))
            return false;
        if (IsEdgePairIntersecting(CO, AO, dim))
            return false;
    }

    // 2. Test AABB face normals
    for (auto dim = 0; dim < kDims; ++dim)
    {
        ScalarType const ma = max({AO(dim), BO(dim), CO(dim)});
        ScalarType const mi = min({AO(dim), BO(dim), CO(dim)});
        if (ma < -e(dim) or mi > e(dim))
            return false;
    }

    // 3. Test triangle face normal
    mini::SVector<ScalarType, kDims> const n = Normalized(Cross(B - A, C - A));
    return PlaneAxisAlignedBoundingBox(A, n, L, U);
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE bool TetrahedronAxisAlignedBoundingBox(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D,
    TMatrixL const& L,
    TMatrixU const& U)
{
    using ScalarType     = typename TMatrixA::ScalarType;
    auto constexpr kRows = TMatrixL::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");

    // Transform tetrahedron into reference space of AABB
    mini::SVector<ScalarType, kDims> const O  = ScalarType(0.5) * (L + U);
    mini::SVector<ScalarType, kDims> const e  = U - O;
    mini::SVector<ScalarType, kDims> const AO = A - O;
    mini::SVector<ScalarType, kDims> const BO = B - O;
    mini::SVector<ScalarType, kDims> const CO = C - O;
    mini::SVector<ScalarType, kDims> const DO = D - O;

    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 6 tetrahedron edges and 3 perpendicular AABB edges (18
     * tests)
     * - Face normals of AABB (3 tests)
     * - Face normals of tetrahedron (4 tests)
     */
    using namespace std;
    auto const ProjectTetrahedron = [&](auto const& a) -> std::pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 4> const p{Dot(AO, a), Dot(BO, a), Dot(CO, a), Dot(DO, a)};
        return make_pair(Min(p), Max(p));
    };
    auto const ProjectAabb = [&](auto const& a) -> ScalarType {
        return Dot(e, Abs(a));
    };
    auto const AreDisjoint = [](ScalarType low, ScalarType up, ScalarType r) {
        return (up < -r) or (r < low);
    };
    auto const TestAxis = [&ProjectTetrahedron, &ProjectAabb, &AreDisjoint](auto const& axis) {
        auto const [low, up] = ProjectTetrahedron(axis);
        ScalarType const r   = ProjectAabb(axis);
        return AreDisjoint(low, up, r);
    };

    // ScalarType(1) Test edge pairs
    auto const IsEdgePairIntersecting = [&TestAxis](auto const& a, auto const& b, auto dim) {
        ScalarType constexpr eps = 1e-15;
        auto const ab            = b - a;
        // Construct natural unit vector in axis dim
        auto const u                          = mini::Unit<ScalarType, kDims>(dim);
        mini::SVector<ScalarType, kDims> axis = Normalized(Cross(ab, u /* - zero*/));
        bool bAxisIsZero                      = All(Abs(axis) <= eps);
        if (not bAxisIsZero)
        {
            return TestAxis(axis);
        }
        else
        {
            // Edges ab and cd are numerically parallel
            auto const n = Cross(ab, /*zero */ -a);
            // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
            axis        = Normalized(Cross(ab, n));
            bAxisIsZero = All(Abs(axis) <= eps);
            if (not bAxisIsZero)
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
    for (auto dim = 0; dim < kDims; ++dim)
    {
        // Edges of tetrahedron are: AB, BC, CA, AD, BD, CD
        if (IsEdgePairIntersecting(A, B, dim))
            return false;

        if (IsEdgePairIntersecting(B, C, dim))
            return false;

        if (IsEdgePairIntersecting(C, A, dim))
            return false;

        if (IsEdgePairIntersecting(A, D, dim))
            return false;

        if (IsEdgePairIntersecting(B, D, dim))
            return false;

        if (IsEdgePairIntersecting(C, D, dim))
            return false;
    }

    // 2. Test AABB face normals
    for (auto dim = 0; dim < kDims; ++dim)
    {
        ScalarType const ma = max({AO(dim), BO(dim), CO(dim), DO(dim)});
        ScalarType const mi = min({AO(dim), BO(dim), CO(dim), DO(dim)});
        if (ma < -e(dim) or mi > e(dim))
            return false;
    }

    // 3. Test tetrahedron face normals
    // Tetrahedron faces are: ABD, BCD, CAD, ACB
    mini::SVector<ScalarType, kDims> n = Normalized(Cross(B - A, D - A));
    if (not PlaneAxisAlignedBoundingBox(A, n, L, U))
        return false;
    n = Normalized(Cross(C - B, D - B));
    if (not PlaneAxisAlignedBoundingBox(B, n, L, U))
        return false;
    n = Normalized(Cross(A - C, D - C));
    if (not PlaneAxisAlignedBoundingBox(C, n, L, U))
        return false;
    n = Normalized(Cross(C - A, B - A));
    return PlaneAxisAlignedBoundingBox(A, n, L, U);
}

template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Triangles2D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2)
{
    using ScalarType     = typename TMatrixA1::ScalarType;
    auto constexpr kRows = TMatrixA1::kRows;
    auto constexpr kDims = 2;
    static_assert(kRows == kDims, "This overlap test is specialized for 2D");

    using namespace std;
    // Separating axis' to test are all 6 triangle edges
    auto const ProjectTriangle = [&](auto const& a,
                                     auto const& b,
                                     auto const& c,
                                     auto const& axis) -> std::pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 3> const p{Dot(a, axis), Dot(b, axis), Dot(c, axis)};
        return make_pair(Min(p), Max(p));
    };
    auto const AreDisjoint = [](ScalarType low1, ScalarType up1, ScalarType low2, ScalarType up2) {
        return (up1 < low2) or (up2 < low1);
    };
    auto const TestAxis = [&](auto const& axis) {
        auto const [low1, up1] = ProjectTriangle(A1, B1, C1, axis);
        auto const [low2, up2] = ProjectTriangle(A2, B2, C2, axis);
        return AreDisjoint(low1, up1, low2, up2);
    };
    auto const EdgeNormal = [](auto e) {
        return Normalized(mini::SVector<ScalarType, kDims>{-e(1), e(0)});
    };
    if (TestAxis(EdgeNormal(B1 - A1)))
        return false;
    if (TestAxis(EdgeNormal(C1 - B1)))
        return false;
    if (TestAxis(EdgeNormal(A1 - C1)))
        return false;
    if (TestAxis(EdgeNormal(B2 - A2)))
        return false;
    if (TestAxis(EdgeNormal(C2 - B2)))
        return false;
    if (TestAxis(EdgeNormal(A2 - C2)))
        return false;
    return true;
}

template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2>
PBAT_HOST_DEVICE bool Triangles3D(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2)
{
    auto const intersections = IntersectionQueries::UvwTriangles3D(A1, B1, C1, A2, B2, C2);
    for (auto const& intersection : intersections)
        if (intersection.has_value())
            return true;
    return false;
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixI,
    mini::CMatrix TMatrixJ,
    mini::CMatrix TMatrixK,
    mini::CMatrix TMatrixL>
PBAT_HOST_DEVICE bool TriangleTetrahedron(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixI const& I,
    TMatrixJ const& J,
    TMatrixK const& K,
    TMatrixL const& L)
{
    using ScalarType = typename TMatrixA::ScalarType;
    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 3 triangle edges and 6 tetrahedron edges (18
     * tests)
     * - Face normals of tetrahedron (4 tests)
     * - Face normal of triangle (1 test)
     */
    auto constexpr kRows = TMatrixA::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");

    using namespace std;
    // ScalarType(1) Test edge pairs
    auto const ProjectTriangle = [&](auto const& a) -> pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 3> const p{Dot(A, a), Dot(B, a), Dot(C, a)};
        return make_pair(Min(p), Max(p));
    };
    auto const ProjectTetrahedron = [&](auto const& a) -> pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 4> const p{Dot(I, a), Dot(J, a), Dot(K, a), Dot(L, a)};
        return make_pair(min({p(0), p(1), p(2), p(3)}), max({p(0), p(1), p(2), p(3)}));
    };
    auto const AreDisjoint = [](ScalarType low1, ScalarType up1, ScalarType low2, ScalarType up2) {
        return (up1 < low2) or (up2 < low1);
    };
    auto const TestAxis = [&ProjectTriangle, &ProjectTetrahedron, &AreDisjoint](auto const& a) {
        auto const [low1, up1] = ProjectTriangle(a);
        auto const [low2, up2] = ProjectTetrahedron(a);
        return AreDisjoint(low1, up1, low2, up2);
    };
    auto const IsEdgePairSeparating =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            ScalarType constexpr eps              = 1e-15;
            auto const ab                         = b - a;
            mini::SVector<ScalarType, kDims> axis = Normalized(Cross(ab, d - c));
            bool bAxisIsZero                      = All(Abs(axis) <= eps);
            if (not bAxisIsZero)
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = Cross(ab, c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis        = Normalized(Cross(ab, n));
                bAxisIsZero = All(Abs(axis) <= eps);
                if (not bAxisIsZero)
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
    mini::SVector<ScalarType, kDims> const IJ = J - I;
    mini::SVector<ScalarType, kDims> const JK = K - J;
    mini::SVector<ScalarType, kDims> const KI = I - K;
    mini::SVector<ScalarType, kDims> const IL = L - I;
    mini::SVector<ScalarType, kDims> const JL = L - J;
    mini::SVector<ScalarType, kDims> const KL = L - K;
    mini::SVector<ScalarType, kDims> n        = Normalized(Cross(IJ, IL));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(JK, JL));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(KI, KL));
    if (TestAxis(n))
        return false;
    mini::SVector<ScalarType, kDims> const IK = K - I;
    n                                         = Normalized(Cross(IK, IJ));
    if (TestAxis(n))
        return false;

    // 3. Test triangle face normal
    n = Normalized(Cross(B - A, C - A));
    return not TestAxis(n);
}

template <
    mini::CMatrix TMatrixA1,
    mini::CMatrix TMatrixB1,
    mini::CMatrix TMatrixC1,
    mini::CMatrix TMatrixD1,
    mini::CMatrix TMatrixA2,
    mini::CMatrix TMatrixB2,
    mini::CMatrix TMatrixC2,
    mini::CMatrix TMatrixD2>
PBAT_HOST_DEVICE bool Tetrahedra(
    TMatrixA1 const& A1,
    TMatrixB1 const& B1,
    TMatrixC1 const& C1,
    TMatrixD1 const& D1,
    TMatrixA2 const& A2,
    TMatrixB2 const& B2,
    TMatrixC2 const& C2,
    TMatrixD2 const& D2)
{
    using ScalarType = typename TMatrixA1::ScalarType;
    /*
     * Separating axis' to test are:
     * - Perpendicular axis' of pairs of 6 tetrahedron A1B1C1D1 edges and 6 tetrahedron A2B2C2D2
     * edges (36 tests)
     * - Face normals of tetrahedron (4+4=8 tests)
     */
    auto constexpr kRows = TMatrixA1::kRows;
    auto constexpr kDims = 3;
    static_assert(kRows == kDims, "This overlap test is specialized for 3D");

    using namespace std;
    auto const ProjectTetrahedron1 = [&](auto const& a) -> pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 4> const p{Dot(A1, a), Dot(B1, a), Dot(C1, a), Dot(D1, a)};
        return make_pair(Min(p), Max(p));
    };
    auto const ProjectTetrahedron2 = [&](auto const& a) -> pair<ScalarType, ScalarType> {
        mini::SVector<ScalarType, 4> const p{Dot(A2, a), Dot(B2, a), Dot(C2, a), Dot(D2, a)};
        return make_pair(Min(p), Max(p));
    };
    auto const AreDisjoint = [](ScalarType low1, ScalarType up1, ScalarType low2, ScalarType up2) {
        return (up1 < low2) or (up2 < low1);
    };
    auto const TestAxis =
        [&ProjectTetrahedron1, &ProjectTetrahedron2, &AreDisjoint](auto const& a) {
            auto const [low1, up1] = ProjectTetrahedron1(a);
            auto const [low2, up2] = ProjectTetrahedron2(a);
            return AreDisjoint(low1, up1, low2, up2);
        };

    // ScalarType(1) Test edge pairs
    auto const IsEdgePairSeparating =
        [&TestAxis](auto const& a, auto const& b, auto const& c, auto const& d) {
            ScalarType constexpr eps              = 1e-15;
            auto const ab                         = b - a;
            mini::SVector<ScalarType, kDims> axis = Normalized(Cross(ab, d - c));
            bool bAxisIsZero                      = All(Abs(axis) <= eps);
            if (not bAxisIsZero)
            {
                return TestAxis(axis);
            }
            else
            {
                // Edges ab and cd are numerically parallel
                auto const n = Cross(ab, c - a);
                // Try a separating axis perpendicular to ab lying in the plane containing ab and cd
                axis        = Normalized(Cross(ab, n));
                bAxisIsZero = All(Abs(axis) <= eps);
                if (not bAxisIsZero)
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
    mini::SVector<ScalarType, kDims> n = Normalized(Cross(B1 - A1, D1 - A1));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(C1 - B1, D1 - B1));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(A1 - C1, D1 - C1));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(C1 - A1, B1 - A1));
    if (TestAxis(n))
        return false;

    n = Normalized(Cross(B2 - A2, D2 - A2));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(C2 - B2, D2 - B2));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(A2 - C2, D2 - C2));
    if (TestAxis(n))
        return false;
    n = Normalized(Cross(C2 - A2, B2 - A2));
    return not TestAxis(n);
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixSC>
PBAT_HOST_DEVICE bool TriangleSphere(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixSC const& SC,
    typename TMatrixSC::ScalarType R)
{
    using ScalarType    = typename TMatrixSC::ScalarType;
    auto const X        = ClosestPointQueries::PointInTriangle(SC, A, B, C);
    ScalarType const d2 = SquaredNorm(X - SC);
    ScalarType const r2 = R * R;
    return d2 < r2;
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD,
    mini::CMatrix TMatrixSC>
PBAT_HOST_DEVICE bool TetrahedronSphere(
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D,
    TMatrixSC const& SC,
    typename TMatrixSC::ScalarType R)
{
    using ScalarType    = typename TMatrixSC::ScalarType;
    auto const X        = ClosestPointQueries::PointInTetrahedron(SC, A, B, C, D);
    ScalarType const d2 = SquaredNorm(X - SC);
    ScalarType const r2 = R * R;
    return d2 < r2;
}

} // namespace OverlapQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_OVERLAPQUERIES_H
