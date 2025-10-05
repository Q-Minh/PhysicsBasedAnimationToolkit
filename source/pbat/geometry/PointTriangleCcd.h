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
    class TScalar = typename TX::ScalarType>
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
    class TScalar = typename TXT::ScalarType>
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
    bool const bIntersectionFound = math::polynomial::ForEachRoot<3>(
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
    TScalar const z1  = z0 * CT[2];
    TScalar const z2  = AT[0] * BT[2];
    TScalar const z3  = z2 * XT[1];
    TScalar const z4  = CT[1] * XT[2];
    TScalar const z5  = z4 * AT[0];
    TScalar const z6  = AT[1] * BT[0];
    TScalar const z7  = z6 * XT[2];
    TScalar const z8  = AT[1] * BT[2];
    TScalar const z9  = z8 * CT[0];
    TScalar const z10 = CT[2] * XT[0];
    TScalar const z11 = z10 * AT[1];
    TScalar const z12 = AT[2] * BT[0];
    TScalar const z13 = z12 * CT[1];
    TScalar const z14 = AT[2] * BT[1];
    TScalar const z15 = z14 * XT[0];
    TScalar const z16 = CT[0] * XT[1];
    TScalar const z17 = z16 * AT[2];
    TScalar const z18 = CT[2] * XT[1];
    TScalar const z19 = z18 * BT[0];
    TScalar const z20 = CT[0] * XT[2];
    TScalar const z21 = z20 * BT[1];
    TScalar const z22 = CT[1] * XT[0];
    TScalar const z23 = z22 * BT[2];
    TScalar const z24 = z0 * XT[2];
    TScalar const z25 = z2 * CT[1];
    TScalar const z26 = z18 * AT[0];
    TScalar const z27 = z6 * CT[2];
    TScalar const z28 = z8 * XT[0];
    TScalar const z29 = z20 * AT[1];
    TScalar const z30 = z12 * XT[1];
    TScalar const z31 = z14 * CT[0];
    TScalar const z32 = z22 * AT[2];
    TScalar const z33 = z4 * BT[0];
    TScalar const z34 = z10 * BT[1];
    TScalar const z35 = z16 * BT[2];
    TScalar const z36 = 3 * z1;
    TScalar const z37 = 3 * z3;
    TScalar const z38 = 3 * z5;
    TScalar const z39 = 3 * z7;
    TScalar const z40 = 3 * z9;
    TScalar const z41 = 3 * z11;
    TScalar const z42 = 3 * z13;
    TScalar const z43 = 3 * z15;
    TScalar const z44 = 3 * z17;
    TScalar const z45 = 3 * z19;
    TScalar const z46 = 3 * z21;
    TScalar const z47 = 3 * z23;
    TScalar const z48 = 3 * z24;
    TScalar const z49 = 3 * z25;
    TScalar const z50 = 3 * z26;
    TScalar const z51 = 3 * z27;
    TScalar const z52 = 3 * z28;
    TScalar const z53 = 3 * z29;
    TScalar const z54 = 3 * z30;
    TScalar const z55 = 3 * z31;
    TScalar const z56 = 3 * z32;
    TScalar const z57 = 3 * z33;
    TScalar const z58 = 3 * z34;
    TScalar const z59 = 3 * z35;
    TScalar const z60 = A[0] * BT[1];
    TScalar const z61 = A[0] * BT[2];
    TScalar const z62 = A[1] * BT[0];
    TScalar const z63 = A[1] * BT[2];
    TScalar const z64 = A[2] * BT[0];
    TScalar const z65 = A[2] * BT[1];
    TScalar const z66 = AT[0] * B[1];
    TScalar const z67 = AT[0] * B[2];
    TScalar const z68 = C[1] * XT[2];
    TScalar const z69 = CT[1] * X[2];
    TScalar const z70 = AT[1] * B[0];
    TScalar const z71 = AT[1] * B[2];
    TScalar const z72 = C[2] * XT[0];
    TScalar const z73 = CT[2] * X[0];
    TScalar const z74 = AT[2] * B[0];
    TScalar const z75 = AT[2] * B[1];
    TScalar const z76 = C[0] * XT[1];
    TScalar const z77 = CT[0] * X[1];
    TScalar const z78 = C[2] * XT[1];
    TScalar const z79 = CT[2] * X[1];
    TScalar const z80 = C[0] * XT[2];
    TScalar const z81 = CT[0] * X[2];
    TScalar const z82 = C[1] * XT[0];
    TScalar const z83 = CT[1] * X[0];
    TScalar const z84 =
        z0 * C[2] + z10 * A[1] + z12 * C[1] + z14 * X[0] + z16 * A[2] + z18 * B[0] + z2 * X[1] +
        z20 * B[1] + z22 * B[2] + z4 * A[0] + z6 * X[2] + z60 * CT[2] + z61 * XT[1] + z62 * XT[2] +
        z63 * CT[0] + z64 * CT[1] + z65 * XT[0] + z66 * CT[2] + z67 * XT[1] + z68 * AT[0] +
        z69 * AT[0] + z70 * XT[2] + z71 * CT[0] + z72 * AT[1] + z73 * AT[1] + z74 * CT[1] +
        z75 * XT[0] + z76 * AT[2] + z77 * AT[2] + z78 * BT[0] + z79 * BT[0] + z8 * C[0] +
        z80 * BT[1] + z81 * BT[1] + z82 * BT[2] + z83 * BT[2] - A[0] * BT[1] * XT[2] -
        A[0] * BT[2] * CT[1] - A[0] * CT[2] * XT[1] - A[1] * BT[0] * CT[2] - A[1] * BT[2] * XT[0] -
        A[1] * CT[0] * XT[2] - A[2] * BT[0] * XT[1] - A[2] * BT[1] * CT[0] - A[2] * CT[1] * XT[0] -
        AT[0] * B[1] * XT[2] - AT[0] * B[2] * CT[1] - AT[0] * BT[1] * X[2] - AT[0] * BT[2] * C[1] -
        AT[0] * C[2] * XT[1] - AT[0] * CT[2] * X[1] - AT[1] * B[0] * CT[2] - AT[1] * B[2] * XT[0] -
        AT[1] * BT[0] * C[2] - AT[1] * BT[2] * X[0] - AT[1] * C[0] * XT[2] - AT[1] * CT[0] * X[2] -
        AT[2] * B[0] * XT[1] - AT[2] * B[1] * CT[0] - AT[2] * BT[0] * X[1] - AT[2] * BT[1] * C[0] -
        AT[2] * C[1] * XT[0] - AT[2] * CT[1] * X[0] - B[0] * CT[1] * XT[2] - B[1] * CT[2] * XT[0] -
        B[2] * CT[0] * XT[1] - BT[0] * C[1] * XT[2] - BT[0] * CT[1] * X[2] - BT[1] * C[2] * XT[0] -
        BT[1] * CT[2] * X[0] - BT[2] * C[0] * XT[1] - BT[2] * CT[0] * X[1];
    TScalar const z85  = A[0] * B[1];
    TScalar const z86  = z85 * CT[2];
    TScalar const z87  = A[0] * B[2];
    TScalar const z88  = z87 * XT[1];
    TScalar const z89  = z60 * C[2];
    TScalar const z90  = z61 * X[1];
    TScalar const z91  = z68 * A[0];
    TScalar const z92  = z69 * A[0];
    TScalar const z93  = A[1] * B[0];
    TScalar const z94  = z93 * XT[2];
    TScalar const z95  = A[1] * B[2];
    TScalar const z96  = z95 * CT[0];
    TScalar const z97  = z62 * X[2];
    TScalar const z98  = z63 * C[0];
    TScalar const z99  = z72 * A[1];
    TScalar const z100 = z73 * A[1];
    TScalar const z101 = A[2] * B[0];
    TScalar const z102 = z101 * CT[1];
    TScalar const z103 = A[2] * B[1];
    TScalar const z104 = z103 * XT[0];
    TScalar const z105 = z64 * C[1];
    TScalar const z106 = z65 * X[0];
    TScalar const z107 = z76 * A[2];
    TScalar const z108 = z77 * A[2];
    TScalar const z109 = z66 * C[2];
    TScalar const z110 = z67 * X[1];
    TScalar const z111 = C[1] * X[2];
    TScalar const z112 = z111 * AT[0];
    TScalar const z113 = z70 * X[2];
    TScalar const z114 = z71 * C[0];
    TScalar const z115 = C[2] * X[0];
    TScalar const z116 = z115 * AT[1];
    TScalar const z117 = z74 * C[1];
    TScalar const z118 = z75 * X[0];
    TScalar const z119 = C[0] * X[1];
    TScalar const z120 = z119 * AT[2];
    TScalar const z121 = z78 * B[0];
    TScalar const z122 = z79 * B[0];
    TScalar const z123 = z80 * B[1];
    TScalar const z124 = z81 * B[1];
    TScalar const z125 = z82 * B[2];
    TScalar const z126 = z83 * B[2];
    TScalar const z127 = C[2] * X[1];
    TScalar const z128 = z127 * BT[0];
    TScalar const z129 = C[0] * X[2];
    TScalar const z130 = z129 * BT[1];
    TScalar const z131 = C[1] * X[0];
    TScalar const z132 = z131 * BT[2];
    TScalar const z133 = z85 * XT[2];
    TScalar const z134 = z87 * CT[1];
    TScalar const z135 = z60 * X[2];
    TScalar const z136 = z61 * C[1];
    TScalar const z137 = z78 * A[0];
    TScalar const z138 = z79 * A[0];
    TScalar const z139 = z93 * CT[2];
    TScalar const z140 = z95 * XT[0];
    TScalar const z141 = z62 * C[2];
    TScalar const z142 = z63 * X[0];
    TScalar const z143 = z80 * A[1];
    TScalar const z144 = z81 * A[1];
    TScalar const z145 = z101 * XT[1];
    TScalar const z146 = z103 * CT[0];
    TScalar const z147 = z64 * X[1];
    TScalar const z148 = z65 * C[0];
    TScalar const z149 = z82 * A[2];
    TScalar const z150 = z83 * A[2];
    TScalar const z151 = z66 * X[2];
    TScalar const z152 = z67 * C[1];
    TScalar const z153 = z127 * AT[0];
    TScalar const z154 = z70 * C[2];
    TScalar const z155 = z71 * X[0];
    TScalar const z156 = z129 * AT[1];
    TScalar const z157 = z74 * X[1];
    TScalar const z158 = z75 * C[0];
    TScalar const z159 = z131 * AT[2];
    TScalar const z160 = z68 * B[0];
    TScalar const z161 = z69 * B[0];
    TScalar const z162 = z72 * B[1];
    TScalar const z163 = z73 * B[1];
    TScalar const z164 = z76 * B[2];
    TScalar const z165 = z77 * B[2];
    TScalar const z166 = z111 * BT[0];
    TScalar const z167 = z115 * BT[1];
    TScalar const z168 = z119 * BT[2];
    sigma[0] = -z1 - z11 - z13 - z15 - z17 - z19 - z21 - z23 + z24 + z25 + z26 + z27 + z28 + z29 -
               z3 + z30 + z31 + z32 + z33 + z34 + z35 - z5 - z7 - z9;
    sigma[1] = z36 + z37 + z38 + z39 + z40 + z41 + z42 + z43 + z44 + z45 + z46 + z47 - z48 - z49 -
               z50 - z51 - z52 - z53 - z54 - z55 - z56 - z57 - z58 - z59 - z84;
    sigma[2] =
        -2 * z0 * X[2] - 2 * z10 * B[1] - z100 - z102 - z104 - z105 - z106 - z107 - z108 - z109 -
        z110 - z112 - z113 - z114 - z116 - z117 - z118 - 2 * z12 * X[1] - z120 - z121 - z122 -
        z123 - z124 - z125 - z126 - z128 - z130 - z132 + z133 + z134 + z135 + z136 + z137 + z138 +
        z139 - 2 * z14 * C[0] + z140 + z141 + z142 + z143 + z144 + z145 + z146 + z147 + z148 +
        z149 + z150 + z151 + z152 + z153 + z154 + z155 + z156 + z157 + z158 + z159 -
        2 * z16 * B[2] + z160 + z161 + z162 + z163 + z164 + z165 + z166 + z167 + z168 -
        2 * z18 * A[0] - 2 * z2 * C[1] - 2 * z20 * A[1] - 2 * z22 * A[2] - z36 - z37 - z38 - z39 -
        2 * z4 * B[0] - z40 - z41 - z42 - z43 - z44 - z45 - z46 - z47 + z48 + z49 + z50 + z51 +
        z52 + z53 + z54 + z55 + z56 + z57 + z58 + z59 - 2 * z6 * C[2] - 2 * z60 * XT[2] -
        2 * z61 * CT[1] - 2 * z62 * CT[2] - 2 * z63 * XT[0] - 2 * z64 * XT[1] - 2 * z65 * CT[0] -
        2 * z66 * XT[2] - 2 * z67 * CT[1] - 2 * z68 * BT[0] - 2 * z69 * BT[0] - 2 * z70 * CT[2] -
        2 * z71 * XT[0] - 2 * z72 * BT[1] - 2 * z73 * BT[1] - 2 * z74 * XT[1] - 2 * z75 * CT[0] -
        2 * z76 * BT[2] - 2 * z77 * BT[2] - 2 * z78 * AT[0] - 2 * z79 * AT[0] - 2 * z8 * X[0] -
        2 * z80 * AT[1] - 2 * z81 * AT[1] - 2 * z82 * AT[2] - 2 * z83 * AT[2] - z86 - z88 - z89 -
        z90 - z91 - z92 - z94 - z96 - z97 - z98 - z99 + 2 * A[0] * BT[1] * CT[2] +
        2 * A[0] * BT[2] * XT[1] + 2 * A[0] * CT[1] * XT[2] + 2 * A[1] * BT[0] * XT[2] +
        2 * A[1] * BT[2] * CT[0] + 2 * A[1] * CT[2] * XT[0] + 2 * A[2] * BT[0] * CT[1] +
        2 * A[2] * BT[1] * XT[0] + 2 * A[2] * CT[0] * XT[1] + 2 * AT[0] * B[1] * CT[2] +
        2 * AT[0] * B[2] * XT[1] + 2 * AT[0] * BT[1] * C[2] + 2 * AT[0] * BT[2] * X[1] +
        2 * AT[0] * C[1] * XT[2] + 2 * AT[0] * CT[1] * X[2] + 2 * AT[1] * B[0] * XT[2] +
        2 * AT[1] * B[2] * CT[0] + 2 * AT[1] * BT[0] * X[2] + 2 * AT[1] * BT[2] * C[0] +
        2 * AT[1] * C[2] * XT[0] + 2 * AT[1] * CT[2] * X[0] + 2 * AT[2] * B[0] * CT[1] +
        2 * AT[2] * B[1] * XT[0] + 2 * AT[2] * BT[0] * C[1] + 2 * AT[2] * BT[1] * X[0] +
        2 * AT[2] * C[0] * XT[1] + 2 * AT[2] * CT[0] * X[1] + 2 * B[0] * CT[2] * XT[1] +
        2 * B[1] * CT[0] * XT[2] + 2 * B[2] * CT[1] * XT[0] + 2 * BT[0] * C[2] * XT[1] +
        2 * BT[0] * CT[2] * X[1] + 2 * BT[1] * C[0] * XT[2] + 2 * BT[1] * CT[0] * X[2] +
        2 * BT[2] * C[1] * XT[0] + 2 * BT[2] * CT[1] * X[0];
    sigma[3] = z1 + z100 - z101 * C[1] + z102 - z103 * X[0] + z104 + z105 + z106 + z107 + z108 +
               z109 + z11 + z110 - z111 * A[0] + z112 + z113 + z114 - z115 * A[1] + z116 + z117 +
               z118 - z119 * A[2] + z120 + z121 + z122 + z123 + z124 + z125 + z126 - z127 * B[0] +
               z128 - z129 * B[1] + z13 + z130 - z131 * B[2] + z132 - z133 - z134 - z135 - z136 -
               z137 - z138 - z139 - z140 - z141 - z142 - z143 - z144 - z145 - z146 - z147 - z148 -
               z149 + z15 - z150 - z151 - z152 - z153 - z154 - z155 - z156 - z157 - z158 - z159 -
               z160 - z161 - z162 - z163 - z164 - z165 - z166 - z167 - z168 + z17 + z19 + z21 +
               z23 - z24 - z25 - z26 - z27 - z28 - z29 + z3 - z30 - z31 - z32 - z33 - z34 - z35 +
               z5 + z7 - z84 - z85 * C[2] + z86 - z87 * X[1] + z88 + z89 + z9 + z90 + z91 + z92 -
               z93 * X[2] + z94 - z95 * C[0] + z96 + z97 + z98 + z99 + A[0] * B[1] * X[2] +
               A[0] * B[2] * C[1] + A[0] * C[2] * X[1] + A[1] * B[0] * C[2] + A[1] * B[2] * X[0] +
               A[1] * C[0] * X[2] + A[2] * B[0] * X[1] + A[2] * B[1] * C[0] + A[2] * C[1] * X[0] +
               B[0] * C[1] * X[2] + B[1] * C[2] * X[0] + B[2] * C[0] * X[1];
    return sigma;
}

} // namespace detail

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_POINTTRIANGLECCD_H
