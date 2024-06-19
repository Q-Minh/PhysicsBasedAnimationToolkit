#ifndef PBAT_GEOMETRY_DISTANCE_QUERIES_H
#define PBAT_GEOMETRY_DISTANCE_QUERIES_H

#include "ClosestPointQueries.h"
#include "OverlapQueries.h"

#include <pbat/Aliases.h>

namespace pbat {
namespace geometry {
namespace DistanceQueries {

/**
 * @brief Obtain squared distance between 2 axis-aligned bounding boxes
 * @param L1 1st AABB's lower corner
 * @param U1 1st AABB's upper corner
 * @param L2 2nd AABB's lower corner
 * @param U2 2nd AABB's upper corner
 * @return
 */
template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
Scalar AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2);

/**
 * @brief Obtain squared distance between point P and triangle ABC
 * @param P
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Scalar PointTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief
 * @tparam TDerivedP
 * @tparam TDerivedA
 * @tparam TDerivedB
 * @tparam TDerivedC
 * @tparam TDerivedD
 * @param P
 * @param A
 * @param B
 * @param C
 * @param D
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
Scalar PointTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedC> const& D);

/**
 * @brief Obtains the signed distance of X w.r.t. plane (P,n)
 * @param X
 * @param P
 * @param n
 * @return
 */
template <class TDerivedX, class TDerivedP, class TDerivedN>
Scalar PointPlane(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedN> const& n);

/**
 * @brief Obtains the distance between sphere (X,R) and triangle ABC.
 * @param X
 * @param R
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedX, class TDerivedA, class TDerivedB, class TDerivedC>
Scalar SphereTriangle(
    Eigen::MatrixBase<TDerivedX> const& X,
    Scalar R,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

template <class TDerivedL1, class TDerivedU1, class TDerivedL2, class TDerivedU2>
Scalar AxisAlignedBoundingBoxes(
    Eigen::MatrixBase<TDerivedL1> const& L1,
    Eigen::MatrixBase<TDerivedU1> const& U1,
    Eigen::MatrixBase<TDerivedL2> const& L2,
    Eigen::MatrixBase<TDerivedU2> const& U2)
{
    auto const dims                                           = L1.rows();
    Vector<TDerivedL1::RowsAtCompileTime> const Lintersection = L1.array().max(L2.array());
    Vector<TDerivedL1::RowsAtCompileTime> const Uintersection = U1.array().min(U2.array());
    Scalar d2{0.};
    for (auto i = 0; i < dims; ++i)
    {
        if (Lintersection(i) > Uintersection(i))
        {
            Scalar const di = Lintersection(i) - Uintersection(i);
            d2 += di * di;
        }
    }
    return d2;
}

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Scalar PointTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    Vector<TDerivedP::RowsAtCompileTime> const PP =
        ClosestPointQueries::PointInTriangle(P, A, B, C);
    return (P - PP).norm();
}

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
Scalar PointTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedC> const& D)
{
    bool const bPointInTetrahedron = OverlapQueries::PointTetrahedron(P, A, B, C, D);
    if (bPointInTetrahedron)
        return 0.;

    Vector<4> sd{
        PointTriangle(P, A, B, D),
        PointTriangle(P, B, C, D),
        PointTriangle(P, C, A, D),
        PointTriangle(P, A, C, B)};
    Scalar const min = sd.minCoeff();
    return min;
}

template <class TDerivedX, class TDerivedP, class TDerivedN>
Scalar PointPlane(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedN> const& n)
{
    return (X - P).dot(n);
}

template <class TDerivedX, class TDerivedA, class TDerivedB, class TDerivedC>
Scalar SphereTriangle(
    Eigen::MatrixBase<TDerivedX> const& X,
    Scalar R,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    Scalar const d2c = PointTriangle(X, A, B, C);
    return d2c - R;
}

} // namespace DistanceQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_DISTANCE_QUERIES_H
