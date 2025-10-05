/**
 * @file DistanceQueries.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to answer distance queries.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_DISTANCEQUERIES_H
#define PBAT_GEOMETRY_DISTANCEQUERIES_H

#include "ClosestPointQueries.h"
#include "OverlapQueries.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>

namespace pbat {
namespace geometry {
/**
 * @brief This namespace contains functions to answer distance queries.
 */
namespace DistanceQueries {

namespace mini = math::linalg::mini;

/**
 * @brief Obtain squared distance between 2 axis-aligned bounding boxes
 * @tparam TMatrixL1 1st AABB's lower corner matrix type
 * @tparam TMatrixU1 1st AABB's upper corner matrix type
 * @tparam TMatrixL2 2nd AABB's lower corner matrix type
 * @tparam TMatrixU2 2nd AABB's upper corner matrix type
 * @param L1 1st AABB's lower corner
 * @param U1 1st AABB's upper corner
 * @param L2 2nd AABB's lower corner
 * @param U2 2nd AABB's upper corner
 * @return Squared distance between the 2 AABBs
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
    TMatrixU2 const& U2) -> typename TMatrixL1::ScalarType;

/**
 * @brief Obtain squared distance between point P and axis-aligned box (L,U)
 * @tparam TMatrixP Point matrix type
 * @tparam TMatrixL Lower corner matrix type
 * @tparam TMatrixU Upper corner matrix type
 * @param P Point
 * @param L Lower corner of the box
 * @param U Upper corner of the box
 * @return Squared distance between point and box
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE auto
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U) ->
    typename TMatrixP::ScalarType;

/**
 * @brief Obtain squared distance between point P and triangle ABC
 * @tparam TMatrixP Point matrix type
 * @tparam TMatrixA Vertex A matrix type
 * @tparam TMatrixB Vertex B matrix type
 * @tparam TMatrixC Vertex C matrix type
 * @param P Point
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return Squared distance between point and triangle
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C) ->
    typename TMatrixP::ScalarType;

/**
 * @brief Obtain squared distance between point P and tetrahedron ABCD
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
 * @return Squared distance between point and tetrahedron
 */
template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE auto PointTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D) -> typename TMatrixP::ScalarType;

/**
 * @brief Obtains the signed distance of X w.r.t. plane (P,n)
 * @tparam TMatrixX Query point matrix type
 * @tparam TMatrixP Point on the plane matrix type
 * @tparam TMatrixN Normal of the plane matrix type
 * @param X Query point
 * @param P Point on the plane
 * @param n Normal of the plane
 * @return Signed distance of X w.r.t. plane (P,n)
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto PointPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n) ->
    typename TMatrixX::ScalarType;

/**
 * @brief Obtains the signed distance of X w.r.t. plane spanned by ABC
 * @tparam TMatrixX Query point matrix type
 * @tparam TMatrixA Vertex A of the triangle in plane matrix type
 * @tparam TMatrixB Vertex B of the triangle in plane matrix type
 * @tparam TMatrixC Vertex C of the triangle in plane matrix type
 * @param X Query point
 * @param A Vertex A of the triangle in plane
 * @param B Vertex B of the triangle in plane
 * @param C Vertex C of the triangle in plane
 * @return Signed distance of X w.r.t. plane spanned by ABC
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointPlane(TMatrixX const& X, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C) ->
    typename TMatrixX::ScalarType;

/**
 * @brief Obtains the squared distance between sphere (X,R) and triangle ABC.
 * @tparam TMatrixX Sphere center matrix type
 * @tparam TMatrixA Triangle vertex A matrix type
 * @tparam TMatrixB Triangle vertex B matrix type
 * @tparam TMatrixC Triangle vertex C matrix type
 * @param X Sphere center
 * @param R Sphere radius
 * @param A Vertex A of the triangle
 * @param B Vertex B of the triangle
 * @param C Vertex C of the triangle
 * @return Squared distance between sphere and triangle
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto SphereTriangle(
    TMatrixX const& X,
    typename TMatrixX::ScalarType R,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> typename TMatrixX::ScalarType;

template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE auto AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2) -> typename TMatrixL1::ScalarType
{
    using ScalarType                    = typename TMatrixL1::ScalarType;
    auto constexpr kDims                = TMatrixL1::kRows;
    mini::SVector<ScalarType, kDims> LI = Max(L1, L2);
    mini::SVector<ScalarType, kDims> UI = Min(U1, U2);
    auto LGU                            = LI > UI;
    mini::SVector<ScalarType, kDims> DI = LI - UI;
    ScalarType d2                       = Dot(Cast<ScalarType>(LGU), Squared(DI));
    return d2;
}

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE auto
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U) ->
    typename TMatrixP::ScalarType
{
    // If point is inside AABB, then distance is 0.
    bool const bIsInsideBox = OverlapQueries::PointAxisAlignedBoundingBox(P, L, U);
    if (bIsInsideBox)
        return 0.;
    // Otherwise compute distance to boundary
    auto const CP = ClosestPointQueries::PointOnAxisAlignedBoundingBox(P, L, U);
    return SquaredNorm(P - CP);
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C) ->
    typename TMatrixP::ScalarType
{
    auto const PP = ClosestPointQueries::PointInTriangle(P, A, B, C);
    return SquaredNorm(P - PP);
}

template <
    mini::CMatrix TMatrixP,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC,
    mini::CMatrix TMatrixD>
PBAT_HOST_DEVICE auto PointTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D) -> typename TMatrixP::ScalarType
{
    bool const bPointInTetrahedron = OverlapQueries::PointTetrahedron3D(P, A, B, C, D);
    if (bPointInTetrahedron)
        return 0.;

    using ScalarType = typename TMatrixP::ScalarType;
    mini::SVector<ScalarType, 4> sd{
        PointTriangle(P, A, B, D),
        PointTriangle(P, B, C, D),
        PointTriangle(P, C, A, D),
        PointTriangle(P, A, C, B)};
    ScalarType const min = Min(sd);
    return min;
}

template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE auto PointPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n) ->
    typename TMatrixX::ScalarType
{
    return Dot(X - P, n);
}

template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto
PointPlane(TMatrixX const& X, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C) ->
    typename TMatrixX::ScalarType
{
    auto const n = Cross(B - A, C - A);
    return PointPlane(X, A, n);
}

template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE auto SphereTriangle(
    TMatrixX const& X,
    typename TMatrixX::ScalarType R,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C) -> typename TMatrixX::ScalarType
{
    auto const sd2c = PointTriangle(X, A, B, C);
    return sd2c - R * R;
}

} // namespace DistanceQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_DISTANCEQUERIES_H
