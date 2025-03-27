/**
 * @file PointTriangleCcd.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Point-triangle continuous collision detection (CCD) implementation
 * @date 2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_POINTTRIANGLECCD_H
#define PBAT_GEOMETRY_POINTTRIANGLECCD_H

#include "IntersectionQueries.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Concepts.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Reductions.h"
#include "pbat/math/polynomial/Roots.h"

#include <array>
#include <cmath>

namespace pbat::geometry {

namespace detail {

/**
 * @brief Computes the univariate linearly swept point-triangle co-planarity polynomial
 *
 * \f[
 * \langle \mathbf{n}(t), \mathbf{q}(t) \rangle = 0 ,
 * \f]
 * where \f$ \mathbf{n}(t) = (\mathbf{b}(t) - \mathbf{a}) \times (\mathbf{c}(t) - \mathbf{a}) \f$
 * and \f$ \mathbf{q}(t) = \mathbf{x}(t) - \mathbf{a} \f$
 *
 * @note Code-generated from python/geometry/ccd.py
 *
 * @tparam TXT Type of the input matrix XT
 * @tparam TAT Type of the input matrix AT
 * @tparam TBT Type of the input matrix BT
 * @tparam TCT Type of the input matrix CT
 * @tparam TX Type of the input matrix X
 * @tparam TA Type of the input matrix A
 * @tparam TB Type of the input matrix B
 * @tparam TC Type of the input matrix C
 * @tparam TScalar Type of the scalar
 * @param XT Matrix of the initial positions of the point
 * @param AT Matrix of the initial positions of the triangle vertex A
 * @param BT Matrix of the initial positions of the triangle vertex B
 * @param CT Matrix of the initial positions of the triangle vertex C
 * @param X Matrix of the final positions of the point
 * @param A Matrix of the final positions of the triangle vertex A
 * @param B Matrix of the final positions of the triangle vertex B
 * @param C Matrix of the final positions of the triangle vertex C
 * @return 4-vector containing the coefficients of the polynomial in increasing order
 */
template <
    math::linalg::mini::CReadableVectorizedMatrix TXT,
    math::linalg::mini::CReadableVectorizedMatrix TAT,
    math::linalg::mini::CReadableVectorizedMatrix TBT,
    math::linalg::mini::CReadableVectorizedMatrix TCT,
    math::linalg::mini::CReadableVectorizedMatrix TX,
    math::linalg::mini::CReadableVectorizedMatrix TA,
    math::linalg::mini::CReadableVectorizedMatrix TB,
    math::linalg::mini::CReadableVectorizedMatrix TC,
    class TScalar = typename TX::Scalar>
PBAT_HOST_DEVICE std::array<TScalar, 4> PointTriangleCcdUnivariatePolynomial(
    TXT const& XT,
    TAT const& AT,
    TBT const& BT,
    TCT const& CT,
    TX const& X,
    TA const& A,
    TB const& B,
    TC const& C);

} // namespace detail

/**
 * @brief Computes the time of impact \f$ t^* \f$ and barycentric coordinates \f$ \mathbf{\beta} \f$
 * of the intersection point between a point and a triangle moving along a linear trajectory.
 *
 * Solves for roots (if any) in the range [0,1] of the polynomial
 * \f[
 * \langle \mathbf{n}(t), \mathbf{q}(t) \rangle = 0 ,
 * \f]
 * where \f$ \mathbf{n}(t) = (\mathbf{b}(t) - \mathbf{a}(t)) \times (\mathbf{c}(t) - \mathbf{a}(t))
 * \f$ and \f$ \mathbf{q}(t) = \mathbf{x}(t) - \mathbf{a}(t) \f$ using polynomial root finder from
 * @cite cem2022polyroot.
 *
 * See @cite provot1997collision and @cite ZachFerg2021CcdBenchmark for more details.
 *
 * @tparam TXT Type of the input matrix XT
 * @tparam TAT Type of the input matrix AT
 * @tparam TBT Type of the input matrix BT
 * @tparam TCT Type of the input matrix CT
 * @tparam TX Type of the input matrix X
 * @tparam TA Type of the input matrix A
 * @tparam TB Type of the input matrix B
 * @tparam TC Type of the input matrix C
 * @tparam TScalar Type of the scalar
 * @param XT Matrix of the initial positions of the point
 * @param AT Matrix of the initial positions of the triangle vertex A
 * @param BT Matrix of the initial positions of the triangle vertex B
 * @param CT Matrix of the initial positions of the triangle vertex C
 * @param X Matrix of the final positions of the point
 * @param A Matrix of the final positions of the triangle vertex A
 * @param B Matrix of the final positions of the triangle vertex B
 * @param C Matrix of the final positions of the triangle vertex C
 * @return A 4-vector `r` containing the time of impact and the barycentric coordinates.
 * @post If no intersection is found, `r[0] < 0`, otherwise
 * `r[0] >= 0` is the earliest time of impact (subject to floating point error).
 */
template <
    math::linalg::mini::CReadableVectorizedMatrix TXT,
    math::linalg::mini::CReadableVectorizedMatrix TAT,
    math::linalg::mini::CReadableVectorizedMatrix TBT,
    math::linalg::mini::CReadableVectorizedMatrix TCT,
    math::linalg::mini::CReadableVectorizedMatrix TX,
    math::linalg::mini::CReadableVectorizedMatrix TA,
    math::linalg::mini::CReadableVectorizedMatrix TB,
    math::linalg::mini::CReadableVectorizedMatrix TC,
    class TScalar = typename TX::ScalarType>
PBAT_HOST_DEVICE auto PointTriangleCcd(
    TXT const& XT,
    TAT const& AT,
    TBT const& BT,
    TCT const& CT,
    TX const& X,
    TA const& A,
    TB const& B,
    TC const& C) -> math::linalg::mini::SVector<TScalar, 4>
{
    auto constexpr kDims = TXT::kRows;
    // 1. Form co-planarity polynomial
    std::array<TScalar, 4> const coeffs =
        detail::PointTriangleCcdUnivariatePolynomial(XT, AT, BT, CT, X, A, B, C);
    // 2. Filter roots
    using namespace pbat::math::linalg::mini;
    SVector<TScalar, 4> r;
    bool const bIntersectionFound = math::polynomial::ForEachRoot(
        [&](TScalar t) {
            // 3. Compute barycentric coordinates of intersection point at earliest root
            auto uvw                  = r.template Slice<3, 1>(1, 0);
            SVector<TScalar, kDims> x = XT + t * (X - XT);
            SVector<TScalar, kDims> a = AT + t * (A - AT);
            SVector<TScalar, kDims> b = BT + t * (B - BT);
            SVector<TScalar, kDims> c = CT + t * (C - CT);
            uvw = IntersectionQueries::TriangleBarycentricCoordinates(x, a, b, c);
            bool const bIsInsideTriangle = All((uvw >= TScalar(0)) and (uvw <= TScalar(1)));
            // Point and triangle intersect at t, if X(t) is inside the triangle (A(t), B(t), C(t))
            r[0] = bIsInsideTriangle * t + (not bIsInsideTriangle) * r[0];
            // Exit as soon as an intersection is found, since we are traversing roots from earliest
            // to latest time of impact
            return bIsInsideTriangle;
        },
        coeffs,
        TScalar(0),
        TScalar(1));
    // Compute return value
    r[0] = bIntersectionFound * r[0] + (not bIntersectionFound) * TScalar(-1);
    return r;
}

namespace detail {

template <
    math::linalg::mini::CReadableVectorizedMatrix TXT,
    math::linalg::mini::CReadableVectorizedMatrix TAT,
    math::linalg::mini::CReadableVectorizedMatrix TBT,
    math::linalg::mini::CReadableVectorizedMatrix TCT,
    math::linalg::mini::CReadableVectorizedMatrix TX,
    math::linalg::mini::CReadableVectorizedMatrix TA,
    math::linalg::mini::CReadableVectorizedMatrix TB,
    math::linalg::mini::CReadableVectorizedMatrix TC,
    class TScalar>
PBAT_HOST_DEVICE std::array<TScalar, 4> PointTriangleCcdUnivariatePolynomial(
    TXT const& XT,
    TAT const& AT,
    TBT const& BT,
    TCT const& CT,
    TX const& X,
    TA const& A,
    TB const& B,
    TC const& C)
{
    std::array<TScalar, 4> sigma;
    TScalar const z0  = AT[0] * BT[1];
    TScalar const z1  = AT[0] * BT[2];
    TScalar const z2  = CT[1] * XT[2];
    TScalar const z3  = AT[1] * BT[0];
    TScalar const z4  = AT[1] * BT[2];
    TScalar const z5  = CT[2] * XT[0];
    TScalar const z6  = AT[2] * BT[0];
    TScalar const z7  = AT[2] * BT[1];
    TScalar const z8  = CT[0] * XT[1];
    TScalar const z9  = CT[2] * XT[1];
    TScalar const z10 = CT[0] * XT[2];
    TScalar const z11 = CT[1] * XT[0];
    TScalar const z12 = A[0] * BT[1];
    TScalar const z13 = A[0] * BT[2];
    TScalar const z14 = A[1] * BT[0];
    TScalar const z15 = A[1] * BT[2];
    TScalar const z16 = A[2] * BT[0];
    TScalar const z17 = A[2] * BT[1];
    TScalar const z18 = AT[0] * B[1];
    TScalar const z19 = AT[0] * B[2];
    TScalar const z20 = C[1] * XT[2];
    TScalar const z21 = CT[1] * X[2];
    TScalar const z22 = AT[1] * B[0];
    TScalar const z23 = AT[1] * B[2];
    TScalar const z24 = C[2] * XT[0];
    TScalar const z25 = CT[2] * X[0];
    TScalar const z26 = AT[2] * B[0];
    TScalar const z27 = AT[2] * B[1];
    TScalar const z28 = C[0] * XT[1];
    TScalar const z29 = CT[0] * X[1];
    TScalar const z30 = C[2] * XT[1];
    TScalar const z31 = CT[2] * X[1];
    TScalar const z32 = C[0] * XT[2];
    TScalar const z33 = CT[0] * X[2];
    TScalar const z34 = C[1] * XT[0];
    TScalar const z35 = CT[1] * X[0];
    TScalar const z36 = A[0] * B[1];
    TScalar const z37 = A[0] * B[2];
    TScalar const z38 = A[1] * B[0];
    TScalar const z39 = A[1] * B[2];
    TScalar const z40 = A[2] * B[0];
    TScalar const z41 = A[2] * B[1];
    TScalar const z42 = C[1] * X[2];
    TScalar const z43 = C[2] * X[0];
    TScalar const z44 = C[0] * X[1];
    TScalar const z45 = C[2] * X[1];
    TScalar const z46 = C[0] * X[2];
    TScalar const z47 = C[1] * X[0];
    sigma[0] = -z0 * CT[2] - z1 * XT[1] - z10 * BT[1] - z11 * BT[2] - z2 * AT[0] - z3 * XT[2] -
               z4 * CT[0] - z5 * AT[1] - z6 * CT[1] - z7 * XT[0] - z8 * AT[2] - z9 * BT[0] +
               AT[0] * BT[1] * XT[2] + AT[0] * BT[2] * CT[1] + AT[0] * CT[2] * XT[1] +
               AT[1] * BT[0] * CT[2] + AT[1] * BT[2] * XT[0] + AT[1] * CT[0] * XT[2] +
               AT[2] * BT[0] * XT[1] + AT[2] * BT[1] * CT[0] + AT[2] * CT[1] * XT[0] +
               BT[0] * CT[1] * XT[2] + BT[1] * CT[2] * XT[0] + BT[2] * CT[0] * XT[1];
    sigma[1] =
        -z0 * C[2] - z1 * X[1] - z10 * B[1] - z11 * B[2] - z12 * CT[2] - z13 * XT[1] - z14 * XT[2] -
        z15 * CT[0] - z16 * CT[1] - z17 * XT[0] - z18 * CT[2] - z19 * XT[1] - z2 * A[0] -
        z20 * AT[0] - z21 * AT[0] - z22 * XT[2] - z23 * CT[0] - z24 * AT[1] - z25 * AT[1] -
        z26 * CT[1] - z27 * XT[0] - z28 * AT[2] - z29 * AT[2] - z3 * X[2] - z30 * BT[0] -
        z31 * BT[0] - z32 * BT[1] - z33 * BT[1] - z34 * BT[2] - z35 * BT[2] - z4 * C[0] -
        z5 * A[1] - z6 * C[1] - z7 * X[0] - z8 * A[2] - z9 * B[0] + A[0] * BT[1] * XT[2] +
        A[0] * BT[2] * CT[1] + A[0] * CT[2] * XT[1] + A[1] * BT[0] * CT[2] + A[1] * BT[2] * XT[0] +
        A[1] * CT[0] * XT[2] + A[2] * BT[0] * XT[1] + A[2] * BT[1] * CT[0] + A[2] * CT[1] * XT[0] +
        AT[0] * B[1] * XT[2] + AT[0] * B[2] * CT[1] + AT[0] * BT[1] * X[2] + AT[0] * BT[2] * C[1] +
        AT[0] * C[2] * XT[1] + AT[0] * CT[2] * X[1] + AT[1] * B[0] * CT[2] + AT[1] * B[2] * XT[0] +
        AT[1] * BT[0] * C[2] + AT[1] * BT[2] * X[0] + AT[1] * C[0] * XT[2] + AT[1] * CT[0] * X[2] +
        AT[2] * B[0] * XT[1] + AT[2] * B[1] * CT[0] + AT[2] * BT[0] * X[1] + AT[2] * BT[1] * C[0] +
        AT[2] * C[1] * XT[0] + AT[2] * CT[1] * X[0] + B[0] * CT[1] * XT[2] + B[1] * CT[2] * XT[0] +
        B[2] * CT[0] * XT[1] + BT[0] * C[1] * XT[2] + BT[0] * CT[1] * X[2] + BT[1] * C[2] * XT[0] +
        BT[1] * CT[2] * X[0] + BT[2] * C[0] * XT[1] + BT[2] * CT[0] * X[1];
    sigma[2] =
        -z12 * C[2] - z13 * X[1] - z14 * X[2] - z15 * C[0] - z16 * C[1] - z17 * X[0] - z18 * C[2] -
        z19 * X[1] - z20 * A[0] - z21 * A[0] - z22 * X[2] - z23 * C[0] - z24 * A[1] - z25 * A[1] -
        z26 * C[1] - z27 * X[0] - z28 * A[2] - z29 * A[2] - z30 * B[0] - z31 * B[0] - z32 * B[1] -
        z33 * B[1] - z34 * B[2] - z35 * B[2] - z36 * CT[2] - z37 * XT[1] - z38 * XT[2] -
        z39 * CT[0] - z40 * CT[1] - z41 * XT[0] - z42 * AT[0] - z43 * AT[1] - z44 * AT[2] -
        z45 * BT[0] - z46 * BT[1] - z47 * BT[2] + A[0] * B[1] * XT[2] + A[0] * B[2] * CT[1] +
        A[0] * BT[1] * X[2] + A[0] * BT[2] * C[1] + A[0] * C[2] * XT[1] + A[0] * CT[2] * X[1] +
        A[1] * B[0] * CT[2] + A[1] * B[2] * XT[0] + A[1] * BT[0] * C[2] + A[1] * BT[2] * X[0] +
        A[1] * C[0] * XT[2] + A[1] * CT[0] * X[2] + A[2] * B[0] * XT[1] + A[2] * B[1] * CT[0] +
        A[2] * BT[0] * X[1] + A[2] * BT[1] * C[0] + A[2] * C[1] * XT[0] + A[2] * CT[1] * X[0] +
        AT[0] * B[1] * X[2] + AT[0] * B[2] * C[1] + AT[0] * C[2] * X[1] + AT[1] * B[0] * C[2] +
        AT[1] * B[2] * X[0] + AT[1] * C[0] * X[2] + AT[2] * B[0] * X[1] + AT[2] * B[1] * C[0] +
        AT[2] * C[1] * X[0] + B[0] * C[1] * XT[2] + B[0] * CT[1] * X[2] + B[1] * C[2] * XT[0] +
        B[1] * CT[2] * X[0] + B[2] * C[0] * XT[1] + B[2] * CT[0] * X[1] + BT[0] * C[1] * X[2] +
        BT[1] * C[2] * X[0] + BT[2] * C[0] * X[1];
    sigma[3] = -z36 * C[2] - z37 * X[1] - z38 * X[2] - z39 * C[0] - z40 * C[1] - z41 * X[0] -
               z42 * A[0] - z43 * A[1] - z44 * A[2] - z45 * B[0] - z46 * B[1] - z47 * B[2] +
               A[0] * B[1] * X[2] + A[0] * B[2] * C[1] + A[0] * C[2] * X[1] + A[1] * B[0] * C[2] +
               A[1] * B[2] * X[0] + A[1] * C[0] * X[2] + A[2] * B[0] * X[1] + A[2] * B[1] * C[0] +
               A[2] * C[1] * X[0] + B[0] * C[1] * X[2] + B[1] * C[2] * X[0] + B[2] * C[0] * X[1];
    return sigma;
}

} // namespace detail

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_POINTTRIANGLECCD_H
