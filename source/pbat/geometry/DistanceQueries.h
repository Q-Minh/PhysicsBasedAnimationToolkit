#ifndef PBAT_GEOMETRY_DISTANCE_QUERIES_H
#define PBAT_GEOMETRY_DISTANCE_QUERIES_H

#include "ClosestPointQueries.h"
#include "OverlapQueries.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>

namespace pbat {
namespace geometry {
namespace DistanceQueries {

namespace mini = math::linalg::mini;

/**
 * @brief Obtain squared distance between 2 axis-aligned bounding boxes
 * @param L1 1st AABB's lower corner
 * @param U1 1st AABB's upper corner
 * @param L2 2nd AABB's lower corner
 * @param U2 2nd AABB's upper corner
 * @return
 */
template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE typename TMatrixL1::ScalarType AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2);

/**
 * @brief Obtain squared distance between point P and axis-aligned box (L,U)
 * @tparam TMatrixP
 * @tparam TMatrixL
 * @tparam TMatrixU
 * @param P
 * @param L
 * @param U
 * @return
 */
template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE typename TMatrixP::ScalarType
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U);

/**
 * @brief Obtain squared distance between point P and triangle ABC
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
PBAT_HOST_DEVICE typename TMatrixP::ScalarType
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C);

/**
 * @brief Obtain squared distance between point P and tetrahedron ABCD
 * @tparam TMatrixP
 * @tparam TMatrixA
 * @tparam TMatrixB
 * @tparam TMatrixC
 * @tparam TMatrixD
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
PBAT_HOST_DEVICE typename TMatrixP::ScalarType PointTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D);

/**
 * @brief Obtains the signed distance of X w.r.t. plane (P,n)
 * @param X
 * @param P
 * @param n
 * @return
 */
template <mini::CMatrix TMatrixX, mini::CMatrix TMatrixP, mini::CMatrix TMatrixN>
PBAT_HOST_DEVICE typename TMatrixX::ScalarType
PointPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n);

/**
 * @brief Obtains the squared distance between sphere (X,R) and triangle ABC.
 * @param X
 * @param R
 * @param A
 * @param B
 * @param C
 * @return
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE typename TMatrixX::ScalarType SphereTriangle(
    TMatrixX const& X,
    typename TMatrixX::ScalarType R,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C);

template <
    mini::CMatrix TMatrixL1,
    mini::CMatrix TMatrixU1,
    mini::CMatrix TMatrixL2,
    mini::CMatrix TMatrixU2>
PBAT_HOST_DEVICE typename TMatrixL1::ScalarType AxisAlignedBoundingBoxes(
    TMatrixL1 const& L1,
    TMatrixU1 const& U1,
    TMatrixL2 const& L2,
    TMatrixU2 const& U2)
{
    using ScalarType                    = typename TMatrixL1::ScalarType;
    auto constexpr kDims                = TMatrixL1::kRows;
    mini::SVector<ScalarType, kDims> LI = Max(L1, L2);
    mini::SVector<ScalarType, kDims> UI = Min(U1, U2);
    auto LGU                            = LI > UI;
    mini::SVector<ScalarType, kDims> DI = LI - UI;
    ScalarType d2                       = Dot(LGU, Squared(DI));
    return d2;
}

template <mini::CMatrix TMatrixP, mini::CMatrix TMatrixL, mini::CMatrix TMatrixU>
PBAT_HOST_DEVICE typename TMatrixP::ScalarType
PointAxisAlignedBoundingBox(TMatrixP const& P, TMatrixL const& L, TMatrixU const& U)
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
PBAT_HOST_DEVICE typename TMatrixP::ScalarType
PointTriangle(TMatrixP const& P, TMatrixA const& A, TMatrixB const& B, TMatrixC const& C)
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
PBAT_HOST_DEVICE typename TMatrixP::ScalarType PointTetrahedron(
    TMatrixP const& P,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C,
    TMatrixD const& D)
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
PBAT_HOST_DEVICE typename TMatrixX::ScalarType
PointPlane(TMatrixX const& X, TMatrixP const& P, TMatrixN const& n)
{
    return Dot(X - P, n);
}

template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE typename TMatrixX::ScalarType SphereTriangle(
    TMatrixX const& X,
    typename TMatrixX::ScalarType R,
    TMatrixA const& A,
    TMatrixB const& B,
    TMatrixC const& C)
{
    auto const sd2c = PointTriangle(X, A, B, C);
    return sd2c - R * R;
}

} // namespace DistanceQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_DISTANCE_QUERIES_H
