/**
 * @file EdgeEdgeCcd.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Edge-edge continuous collision detection (CCD) implementation
 * @date 2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_EDGEEDGECCD_H
#define PBAT_GEOMETRY_EDGEEDGECCD_H

#include "pbat/HostDevice.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/math/linalg/mini/Concepts.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Reductions.h"
#include "pbat/math/polynomial/Roots.h"

#include <array>
#include <cmath>

namespace pbat::geometry {

namespace detail {

/**
 * @brief Computes the univariate linearly swept edge-edge co-planarity polynomial
 *
 * \f[
 * \langle \mathbf{n}(t), \mathbf{q}(t) \rangle = 0 ,
 * \f]
 * where \f$ \mathbf{n}(t) = (\mathbf{q}_1(t) - \mathbf{p}_1(t)) \times (\mathbf{q}_2(t) -
 * \mathbf{p}_2(t)) \f$ and \f$ \mathbf{q}(t) = \mathbf{x}(t) - \mathbf{p}_1(t) \f$
 *
 * @note Code-generated from python/geometry/ccd.py
 *
 * @tparam TP1T Type of the input matrix P1T
 * @tparam TQ1T Type of the input matrix Q1T
 * @tparam TP2T Type of the input matrix P2T
 * @tparam TQ2T Type of the input matrix Q2T
 * @tparam TP1 Type of the input matrix P1
 * @tparam TQ1 Type of the input matrix Q1
 * @tparam TP2 Type of the input matrix P2
 * @tparam TQ2 Type of the input matrix Q2
 * @tparam TScalar Type of the scalar
 * @param P1T Matrix of the initial positions of the point P on edge 1
 * @param Q1T Matrix of the initial positions of the point Q on edge 1
 * @param P2T Matrix of the initial positions of the point P on edge 2
 * @param Q2T Matrix of the initial positions of the point Q on edge 2
 * @param P1 Matrix of the final positions of the point P on edge 1
 * @param Q1 Matrix of the final positions of the point Q on edge 1
 * @param P2 Matrix of the final positions of the point P on edge 2
 * @param Q2 Matrix of the final positions of the point Q on edge 2
 * @return 4-vector containing the coefficients of the polynomial in increasing order
 */
template <
    math::linalg::mini::CReadableVectorizedMatrix TP1T,
    math::linalg::mini::CReadableVectorizedMatrix TQ1T,
    math::linalg::mini::CReadableVectorizedMatrix TP2T,
    math::linalg::mini::CReadableVectorizedMatrix TQ2T,
    math::linalg::mini::CReadableVectorizedMatrix TP1,
    math::linalg::mini::CReadableVectorizedMatrix TQ1,
    math::linalg::mini::CReadableVectorizedMatrix TP2,
    math::linalg::mini::CReadableVectorizedMatrix TQ2,
    class TScalar = typename TP1T::ScalarType>
PBAT_HOST_DEVICE std::array<TScalar, 4> EdgeEdgeCcdUnivariatePolynomial(
    TP1T const& P1T,
    TQ1T const& Q1T,
    TP2T const& P2T,
    TQ2T const& Q2T,
    TP1 const& P1,
    TQ1 const& Q1,
    TP2 const& P2,
    TQ2 const& Q2);

} // namespace detail

/**
 * @brief Computes the time of impact \f$ t^* \f$ and barycentric coordinates \f$ \mathbf{\beta} \f$
 * of the intersection point between an edge \f$ (P1,Q1) \f$ and another edge \f$ (P2,Q2) \f$ moving
 * along a linear trajectory.
 *
 * Solves for inexact roots (if any) in the range [0,1] of the polynomial
 * \f[
 * \langle \mathbf{n}(t), \mathbf{q}(t) \rangle = 0 ,
 * \f]
 * where \f$ \mathbf{n}(t) = (\mathbf{q}_1(t) - \mathbf{p}_1(t)) \times (\mathbf{q}_2(t) -
 * \mathbf{p}_2(t)) \f$ and \f$ \mathbf{q}(t) = \mathbf{p}_2(t) - \mathbf{p}_1(t) \f$ using
 * polynomial root finder from @cite cem2022polyroot.
 *
 * See @cite provot1997collision and @cite ZachFerg2021CcdBenchmark for more details.
 *
 * @tparam TP1T Type of the input matrix P1T
 * @tparam TQ1T Type of the input matrix Q1T
 * @tparam TP2T Type of the input matrix P2T
 * @tparam TQ2T Type of the input matrix Q2T
 * @tparam TP1 Type of the input matrix P1
 * @tparam TQ1 Type of the input matrix Q1
 * @tparam TP2 Type of the input matrix P2
 * @tparam TQ2 Type of the input matrix Q2
 * @tparam TScalar Type of the scalar
 * @param P1T Matrix of the initial positions of the point P on edge 1
 * @param Q1T Matrix of the initial positions of the point Q on edge 1
 * @param P2T Matrix of the initial positions of the point P on edge 2
 * @param Q2T Matrix of the initial positions of the point Q on edge 2
 * @param P1 Matrix of the final positions of the point P on edge 1
 * @param Q1 Matrix of the final positions of the point Q on edge 1
 * @param P2 Matrix of the final positions of the point P on edge 2
 * @param Q2 Matrix of the final positions of the point Q on edge 2
 * @return 3-vector containing the time of impact and the barycentric coordinates
 * @post If no intersection is found, `r[0] < 0`, otherwise `r[0]` is the earliest time of impact
 * @post `r[1]` and `r[2]` are the barycentric coordinates of the intersection point along \f$
 * (P1,Q1) \f$ and \f$ (P2,Q2) \f$ respectively
 */
template <
    math::linalg::mini::CReadableVectorizedMatrix TP1T,
    math::linalg::mini::CReadableVectorizedMatrix TQ1T,
    math::linalg::mini::CReadableVectorizedMatrix TP2T,
    math::linalg::mini::CReadableVectorizedMatrix TQ2T,
    math::linalg::mini::CReadableVectorizedMatrix TP1,
    math::linalg::mini::CReadableVectorizedMatrix TQ1,
    math::linalg::mini::CReadableVectorizedMatrix TP2,
    math::linalg::mini::CReadableVectorizedMatrix TQ2,
    class TScalar = typename TP1T::ScalarType>
PBAT_HOST_DEVICE auto EdgeEdgeCcd(
    TP1T const& P1T,
    TQ1T const& Q1T,
    TP2T const& P2T,
    TQ2T const& Q2T,
    TP1 const& P1,
    TQ1 const& Q1,
    TP2 const& P2,
    TQ2 const& Q2) -> math::linalg::mini::SVector<TScalar, 3>
{
    auto constexpr kDims = TP1T::kRows;
    // 1. Form co-planarity polynomial
    std::array<TScalar, 4> const coeffs =
        detail::EdgeEdgeCcdUnivariatePolynomial(P1T, Q1T, P2T, Q2T, P1, Q1, P2, Q2);
    // 2. Filter roots
    using namespace pbat::math::linalg::mini;
    SVector<TScalar, 3> r;
    bool const bIntersectionFound = math::polynomial::ForEachRoot<3>(
        [&](TScalar t) {
            // 3. Compute barycentric coordinates of intersection point (if any) at root
            auto uv                          = r.template Slice<2, 1>(1, 0);
            SVector<TScalar, kDims> const p1 = P1T + t * (P1 - P1T);
            SVector<TScalar, kDims> const q1 = Q1T + t * (Q1 - Q1T);
            SVector<TScalar, kDims> const p2 = P2T + t * (P2 - P2T);
            SVector<TScalar, kDims> const q2 = Q2T + t * (Q2 - Q2T);
            auto constexpr zero              = std::numeric_limits<TScalar>::min();
            uv                               = ClosestPointQueries::Lines(p1, q1, p2, q2, zero);
            // 4. Check if there is any intersection point, i.e. closest points on lines must be in
            // the line segments, and the distance between the closest points must be zero.
            bool const bIsInsideEdges  = All((uv >= TScalar(0)) and (uv <= TScalar(1)));
            auto const cp1             = p1 + uv[0] * (q1 - p1);
            auto const cp2             = p2 + uv[1] * (q2 - p2);
            TScalar const d2           = SquaredNorm(cp1 - cp2);
            bool const bIsIntersection = (d2 <= zero) and bIsInsideEdges;
            r[0]                       = bIsIntersection * t + (not bIsIntersection) * r[0];
            // Exit as soon as an intersection is found, since we are traversing roots from earliest
            // to latest time of impact
            return bIsIntersection;
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
    math::linalg::mini::CReadableVectorizedMatrix TP1T,
    math::linalg::mini::CReadableVectorizedMatrix TQ1T,
    math::linalg::mini::CReadableVectorizedMatrix TP2T,
    math::linalg::mini::CReadableVectorizedMatrix TQ2T,
    math::linalg::mini::CReadableVectorizedMatrix TP1,
    math::linalg::mini::CReadableVectorizedMatrix TQ1,
    math::linalg::mini::CReadableVectorizedMatrix TP2,
    math::linalg::mini::CReadableVectorizedMatrix TQ2,
    class TScalar>
PBAT_HOST_DEVICE std::array<TScalar, 4> EdgeEdgeCcdUnivariatePolynomial(
    TP1T const& P1T,
    TQ1T const& Q1T,
    TP2T const& P2T,
    TQ2T const& Q2T,
    TP1 const& P1,
    TQ1 const& Q1,
    TP2 const& P2,
    TQ2 const& Q2)
{
    std::array<TScalar, 4> sigma;
    TScalar const z0  = P1T[0] * P2T[1];
    TScalar const z1  = P1T[0] * P2T[2];
    TScalar const z2  = Q1T[1] * Q2T[2];
    TScalar const z3  = P1T[1] * P2T[0];
    TScalar const z4  = P1T[1] * P2T[2];
    TScalar const z5  = Q1T[2] * Q2T[0];
    TScalar const z6  = P1T[2] * P2T[0];
    TScalar const z7  = P1T[2] * P2T[1];
    TScalar const z8  = Q1T[0] * Q2T[1];
    TScalar const z9  = Q1T[2] * Q2T[1];
    TScalar const z10 = Q1T[0] * Q2T[2];
    TScalar const z11 = Q1T[1] * Q2T[0];
    TScalar const z12 =
        z0 * Q1T[2] + z1 * Q2T[1] + z10 * P2T[1] + z11 * P2T[2] + z2 * P1T[0] + z3 * Q2T[2] +
        z4 * Q1T[0] + z5 * P1T[1] + z6 * Q1T[1] + z7 * Q2T[0] + z8 * P1T[2] + z9 * P2T[0] -
        P1T[0] * P2T[1] * Q2T[2] - P1T[0] * P2T[2] * Q1T[1] - P1T[0] * Q1T[2] * Q2T[1] -
        P1T[1] * P2T[0] * Q1T[2] - P1T[1] * P2T[2] * Q2T[0] - P1T[1] * Q1T[0] * Q2T[2] -
        P1T[2] * P2T[0] * Q2T[1] - P1T[2] * P2T[1] * Q1T[0] - P1T[2] * Q1T[1] * Q2T[0] -
        P2T[0] * Q1T[1] * Q2T[2] - P2T[1] * Q1T[2] * Q2T[0] - P2T[2] * Q1T[0] * Q2T[1];
    TScalar const z13 = P1[0] * P2T[1];
    TScalar const z14 = z13 * Q1T[2];
    TScalar const z15 = P1[0] * P2T[2];
    TScalar const z16 = z15 * Q2T[1];
    TScalar const z17 = z2 * P1[0];
    TScalar const z18 = P1[1] * P2T[0];
    TScalar const z19 = z18 * Q2T[2];
    TScalar const z20 = P1[1] * P2T[2];
    TScalar const z21 = z20 * Q1T[0];
    TScalar const z22 = z5 * P1[1];
    TScalar const z23 = P1[2] * P2T[0];
    TScalar const z24 = z23 * Q1T[1];
    TScalar const z25 = P1[2] * P2T[1];
    TScalar const z26 = z25 * Q2T[0];
    TScalar const z27 = z8 * P1[2];
    TScalar const z28 = P1T[0] * P2[2];
    TScalar const z29 = z28 * Q2T[1];
    TScalar const z30 = z0 * Q1[2];
    TScalar const z31 = Q1[1] * Q2T[2];
    TScalar const z32 = z31 * P1T[0];
    TScalar const z33 = P1T[1] * P2[0];
    TScalar const z34 = z33 * Q2T[2];
    TScalar const z35 = z4 * Q1[0];
    TScalar const z36 = Q1[2] * Q2T[0];
    TScalar const z37 = z36 * P1T[1];
    TScalar const z38 = P1T[2] * P2[1];
    TScalar const z39 = z38 * Q2T[0];
    TScalar const z40 = z6 * Q1[1];
    TScalar const z41 = Q1[0] * Q2T[1];
    TScalar const z42 = z41 * P1T[2];
    TScalar const z43 = z9 * P2[0];
    TScalar const z44 = z10 * P2[1];
    TScalar const z45 = z11 * P2[2];
    TScalar const z46 = Q1[2] * Q2T[1];
    TScalar const z47 = z46 * P2T[0];
    TScalar const z48 = Q1[0] * Q2T[2];
    TScalar const z49 = z48 * P2T[1];
    TScalar const z50 = Q1[1] * Q2T[0];
    TScalar const z51 = z50 * P2T[2];
    TScalar const z52 = z13 * Q2T[2];
    TScalar const z53 = z15 * Q1T[1];
    TScalar const z54 = z9 * P1[0];
    TScalar const z55 = z18 * Q1T[2];
    TScalar const z56 = z20 * Q2T[0];
    TScalar const z57 = z10 * P1[1];
    TScalar const z58 = z23 * Q2T[1];
    TScalar const z59 = z25 * Q1T[0];
    TScalar const z60 = z11 * P1[2];
    TScalar const z61 = P1T[0] * P2[1];
    TScalar const z62 = z61 * Q2T[2];
    TScalar const z63 = z1 * Q1[1];
    TScalar const z64 = z46 * P1T[0];
    TScalar const z65 = P1T[1] * P2[2];
    TScalar const z66 = z65 * Q2T[0];
    TScalar const z67 = z3 * Q1[2];
    TScalar const z68 = z48 * P1T[1];
    TScalar const z69 = P1T[2] * P2[0];
    TScalar const z70 = z69 * Q2T[1];
    TScalar const z71 = z7 * Q1[0];
    TScalar const z72 = z50 * P1T[2];
    TScalar const z73 = z2 * P2[0];
    TScalar const z74 = z5 * P2[1];
    TScalar const z75 = z8 * P2[2];
    TScalar const z76 = z31 * P2T[0];
    TScalar const z77 = z36 * P2T[1];
    TScalar const z78 = z41 * P2T[2];
    TScalar const z79 = Q1T[1] * Q2[2];
    TScalar const z80 = Q1T[2] * Q2[0];
    TScalar const z81 = Q1T[0] * Q2[1];
    TScalar const z82 = Q1T[2] * Q2[1];
    TScalar const z83 = Q1T[0] * Q2[2];
    TScalar const z84 = Q1T[1] * Q2[0];
    TScalar const z85 =
        z1 * Q2[1] + z3 * Q2[2] + z61 * Q1T[2] + z65 * Q1T[0] + z69 * Q1T[1] + z7 * Q2[0] +
        z79 * P1T[0] + z80 * P1T[1] + z81 * P1T[2] + z82 * P2T[0] + z83 * P2T[1] + z84 * P2T[2] -
        P1T[0] * P2[2] * Q1T[1] - P1T[0] * P2T[1] * Q2[2] - P1T[0] * Q1T[2] * Q2[1] -
        P1T[1] * P2[0] * Q1T[2] - P1T[1] * P2T[2] * Q2[0] - P1T[1] * Q1T[0] * Q2[2] -
        P1T[2] * P2[1] * Q1T[0] - P1T[2] * P2T[0] * Q2[1] - P1T[2] * Q1T[1] * Q2[0] -
        P2T[0] * Q1T[1] * Q2[2] - P2T[1] * Q1T[2] * Q2[0] - P2T[2] * Q1T[0] * Q2[1];
    TScalar const z86  = P1[0] * P2[1];
    TScalar const z87  = z86 * Q1T[2];
    TScalar const z88  = P1[0] * P2[2];
    TScalar const z89  = z15 * Q2[1];
    TScalar const z90  = z79 * P1[0];
    TScalar const z91  = P1[1] * P2[0];
    TScalar const z92  = P1[1] * P2[2];
    TScalar const z93  = z92 * Q1T[0];
    TScalar const z94  = z18 * Q2[2];
    TScalar const z95  = z80 * P1[1];
    TScalar const z96  = P1[2] * P2[0];
    TScalar const z97  = z96 * Q1T[1];
    TScalar const z98  = P1[2] * P2[1];
    TScalar const z99  = z25 * Q2[0];
    TScalar const z100 = z81 * P1[2];
    TScalar const z101 = z1 * P2[1];
    TScalar const z102 = z61 * Q1[2];
    TScalar const z103 = z28 * Q2[1];
    TScalar const z104 = Q1[1] * Q2[2];
    TScalar const z105 = z104 * P1T[0];
    TScalar const z106 = z33 * Q2[2];
    TScalar const z107 = z3 * P2[2];
    TScalar const z108 = z65 * Q1[0];
    TScalar const z109 = Q1[2] * Q2[0];
    TScalar const z110 = z109 * P1T[1];
    TScalar const z111 = z7 * P2[0];
    TScalar const z112 = z69 * Q1[1];
    TScalar const z113 = z38 * Q2[0];
    TScalar const z114 = Q1[0] * Q2[1];
    TScalar const z115 = z114 * P1T[2];
    TScalar const z116 = P2[0] * P2T[2];
    TScalar const z117 = z116 * Q1T[1];
    TScalar const z118 = z82 * P2[0];
    TScalar const z119 = P2[1] * P2T[0];
    TScalar const z120 = z119 * Q1T[2];
    TScalar const z121 = z83 * P2[1];
    TScalar const z122 = P2[2] * P2T[1];
    TScalar const z123 = z122 * Q1T[0];
    TScalar const z124 = z84 * P2[2];
    TScalar const z125 = Q1[2] * Q2[1];
    TScalar const z126 = z125 * P2T[0];
    TScalar const z127 = Q1[0] * Q2[2];
    TScalar const z128 = z127 * P2T[1];
    TScalar const z129 = Q1[1] * Q2[0];
    TScalar const z130 = z129 * P2T[2];
    TScalar const z131 = P2[2] * Q1T[1];
    TScalar const z132 = z131 * P1[0];
    TScalar const z133 = z13 * Q2[2];
    TScalar const z134 = z82 * P1[0];
    TScalar const z135 = P2[0] * Q1T[2];
    TScalar const z136 = z135 * P1[1];
    TScalar const z137 = z20 * Q2[0];
    TScalar const z138 = z83 * P1[1];
    TScalar const z139 = P2[1] * Q1T[0];
    TScalar const z140 = z139 * P1[2];
    TScalar const z141 = z23 * Q2[1];
    TScalar const z142 = z84 * P1[2];
    TScalar const z143 = z61 * Q2[2];
    TScalar const z144 = z0 * P2[2];
    TScalar const z145 = z28 * Q1[1];
    TScalar const z146 = z125 * P1T[0];
    TScalar const z147 = z4 * P2[0];
    TScalar const z148 = z33 * Q1[2];
    TScalar const z149 = z65 * Q2[0];
    TScalar const z150 = z127 * P1T[1];
    TScalar const z151 = z69 * Q2[1];
    TScalar const z152 = z6 * P2[1];
    TScalar const z153 = z38 * Q1[0];
    TScalar const z154 = z129 * P1T[2];
    TScalar const z155 = z135 * P2T[1];
    TScalar const z156 = z79 * P2[0];
    TScalar const z157 = z139 * P2T[2];
    TScalar const z158 = z80 * P2[1];
    TScalar const z159 = z131 * P2T[0];
    TScalar const z160 = z81 * P2[2];
    TScalar const z161 = z104 * P2T[0];
    TScalar const z162 = z109 * P2T[1];
    TScalar const z163 = z114 * P2T[2];
    sigma[0]           = -z12;
    sigma[1] =
        -2 * z0 * Q2T[2] - 2 * z1 * Q1T[1] - 2 * z10 * P1T[1] - 2 * z11 * P1T[2] - z14 - z16 - z17 -
        z19 - 2 * z2 * P2T[0] - z21 - z22 - z24 - z26 - z27 - z29 - 2 * z3 * Q1T[2] - z30 - z32 -
        z34 - z35 - z37 - z39 - 2 * z4 * Q2T[0] - z40 - z42 - z43 - z44 - z45 - z47 - z49 -
        2 * z5 * P2T[1] - z51 + z52 + z53 + z54 + z55 + z56 + z57 + z58 + z59 - 2 * z6 * Q2T[1] +
        z60 + z62 + z63 + z64 + z66 + z67 + z68 - 2 * z7 * Q1T[0] + z70 + z71 + z72 + z73 + z74 +
        z75 + z76 + z77 + z78 - 2 * z8 * P2T[2] - z85 - 2 * z9 * P1T[0] +
        2 * P1T[0] * P2T[1] * Q1T[2] + 2 * P1T[0] * P2T[2] * Q2T[1] + 2 * P1T[0] * Q1T[1] * Q2T[2] +
        2 * P1T[1] * P2T[0] * Q2T[2] + 2 * P1T[1] * P2T[2] * Q1T[0] + 2 * P1T[1] * Q1T[2] * Q2T[0] +
        2 * P1T[2] * P2T[0] * Q1T[1] + 2 * P1T[2] * P2T[1] * Q2T[0] + 2 * P1T[2] * Q1T[0] * Q2T[1] +
        2 * P2T[0] * Q1T[2] * Q2T[1] + 2 * P2T[1] * Q1T[0] * Q2T[2] + 2 * P2T[2] * Q1T[1] * Q2T[0];
    sigma[2] =
        -2 * z0 * Q2[2] - z100 - z101 - z102 - z103 - z105 - z106 - z107 - z108 - z110 - z111 -
        z112 - z113 - z115 - z117 - z118 - z12 - z120 - z121 - z123 - z124 - z126 - z128 -
        z13 * Q1[2] - z130 + z132 + z133 + z134 + z136 + z137 + z138 + z14 + z140 + z141 + z142 +
        z143 + z144 + z145 + z146 + z147 + z148 + z149 + z150 + z151 + z152 + z153 + z154 + z155 +
        z156 + z157 + z158 + z159 + z16 + z160 + z161 + z162 + z163 + z17 + z19 - z20 * Q1[0] +
        z21 + z22 - z23 * Q1[1] + z24 + z26 + z27 - 2 * z28 * Q1T[1] + z29 + z30 - z31 * P1[0] +
        z32 - 2 * z33 * Q1T[2] + z34 + z35 - z36 * P1[1] + z37 - 2 * z38 * Q1T[0] + z39 -
        2 * z4 * Q2[0] + z40 - z41 * P1[2] + z42 + z43 + z44 + z45 - z46 * P2[0] + z47 -
        z48 * P2[1] + z49 - z50 * P2[2] + z51 - z52 - z53 - z54 - z55 - z56 - z57 - z58 - z59 -
        2 * z6 * Q2[1] - z60 - z62 - z63 - z64 - z66 - z67 - z68 - z70 - z71 - z72 - z73 - z74 -
        z75 - z76 - z77 - z78 - 2 * z79 * P2T[0] - 2 * z80 * P2T[1] - 2 * z81 * P2T[2] -
        2 * z82 * P1T[0] - 2 * z83 * P1T[1] - 2 * z84 * P1T[2] - z87 - z88 * Q2T[1] - z89 - z90 -
        z91 * Q2T[2] - z93 - z94 - z95 - z97 - z98 * Q2T[0] - z99 + P1[0] * P2[1] * Q2T[2] +
        P1[0] * P2T[2] * Q1[1] + P1[0] * Q1[2] * Q2T[1] + P1[1] * P2[2] * Q2T[0] +
        P1[1] * P2T[0] * Q1[2] + P1[1] * Q1[0] * Q2T[2] + P1[2] * P2[0] * Q2T[1] +
        P1[2] * P2T[1] * Q1[0] + P1[2] * Q1[1] * Q2T[0] + 2 * P1T[0] * P2[1] * Q1T[2] +
        2 * P1T[0] * P2T[2] * Q2[1] + 2 * P1T[0] * Q1T[1] * Q2[2] + 2 * P1T[1] * P2[2] * Q1T[0] +
        2 * P1T[1] * P2T[0] * Q2[2] + 2 * P1T[1] * Q1T[2] * Q2[0] + 2 * P1T[2] * P2[0] * Q1T[1] +
        2 * P1T[2] * P2T[1] * Q2[0] + 2 * P1T[2] * Q1T[0] * Q2[1] + P2[0] * Q1[1] * Q2T[2] +
        P2[1] * Q1[2] * Q2T[0] + P2[2] * Q1[0] * Q2T[1] + 2 * P2T[0] * Q1T[2] * Q2[1] +
        2 * P2T[1] * Q1T[0] * Q2[2] + 2 * P2T[2] * Q1T[1] * Q2[0];
    sigma[3] = z100 + z101 + z102 + z103 - z104 * P1[0] + z105 + z106 + z107 + z108 - z109 * P1[1] +
               z110 + z111 + z112 + z113 - z114 * P1[2] + z115 - z116 * Q1[1] + z117 + z118 -
               z119 * Q1[2] + z120 + z121 - z122 * Q1[0] + z123 + z124 - z125 * P2[0] + z126 -
               z127 * P2[1] + z128 - z129 * P2[2] + z130 - z132 - z133 - z134 - z136 - z137 - z138 -
               z140 - z141 - z142 - z143 - z144 - z145 - z146 - z147 - z148 - z149 - z15 * P2[1] -
               z150 - z151 - z152 - z153 - z154 - z155 - z156 - z157 - z158 - z159 - z160 - z161 -
               z162 - z163 - z18 * P2[2] - z25 * P2[0] - z85 - z86 * Q1[2] + z87 - z88 * Q2[1] +
               z89 + z90 - z91 * Q2[2] - z92 * Q1[0] + z93 + z94 + z95 - z96 * Q1[1] + z97 -
               z98 * Q2[0] + z99 + P1[0] * P2[1] * Q2[2] + P1[0] * P2[2] * P2T[1] +
               P1[0] * P2[2] * Q1[1] + P1[0] * Q1[2] * Q2[1] + P1[1] * P2[0] * P2T[2] +
               P1[1] * P2[0] * Q1[2] + P1[1] * P2[2] * Q2[0] + P1[1] * Q1[0] * Q2[2] +
               P1[2] * P2[0] * Q2[1] + P1[2] * P2[1] * P2T[0] + P1[2] * P2[1] * Q1[0] +
               P1[2] * Q1[1] * Q2[0] + P2[0] * P2T[1] * Q1[2] + P2[0] * Q1[1] * Q2[2] +
               P2[1] * P2T[2] * Q1[0] + P2[1] * Q1[2] * Q2[0] + P2[2] * P2T[0] * Q1[1] +
               P2[2] * Q1[0] * Q2[1];
    return sigma;
}

} // namespace detail

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_EDGEEDGECCD_H
