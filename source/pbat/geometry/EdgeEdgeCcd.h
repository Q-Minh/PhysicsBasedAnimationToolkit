#ifndef PBAT_GEOMETRY_EDGEEDGECCD_H
#define PBAT_GEOMETRY_EDGEEDGECCD_H

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
 * where \f$ \mathbf{n}(t) = (\mathbf{b}(t) - \mathbf{a}) \times (\mathbf{c}(t) - \mathbf{a}) \f$
 * and \f$ \mathbf{q}(t) = \mathbf{x}(t) - \mathbf{a} \f$
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
    class TScalar = typename TP1T::Scalar>
std::array<TScalar, 4> EdgeEdgeCcdUnivariatePolynomial(
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
 * of the intersection point between an edge (P1,Q1) and another edge (P2,Q2) moving along a linear
 * trajectory.
 *
 * Solves for inexact roots (if any) in the range [0,1] of the polynomial
 * \f[
 * \langle \mathbf{n}(t), \mathbf{q}(t) \rangle = 0 ,
 * \f]
 * where \f$ \mathbf{n}(t) = (\mathbf{b}(t) - \mathbf{a}) \times (\mathbf{c}(t) - \mathbf{a}) \f$
 * and \f$ \mathbf{q}(t) = \mathbf{x}(t) - \mathbf{a} \f$ using polynomial root finder from
 * @cite cem2022polyroot.
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
 * @post `r[1]` and `r[2]` are the barycentric coordinates of the intersection point along (P1,Q1)
 * and (P2,Q2) respectively
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
    class TScalar = typename TP1T::Scalar>
auto EdgeEdgeCcd(
    TP1T const& P1T,
    TQ1T const& Q1T,
    TP2T const& P2T,
    TQ2T const& Q2T,
    TP1 const& P1,
    TQ1 const& Q1,
    TP2 const& P2,
    TQ2 const& Q2) -> math::linalg::mini::SVector<TScalar, 3>
{
    // 1. Form co-planarity polynomial
    std::array<TScalar, 4> const coeffs =
        detail::EdgeEdgeCcdUnivariatePolynomial(P1T, Q1T, P2T, Q2T, P1, Q1, P2, Q2);
    // 2. Filter roots
    using namespace pbat::math::linalg::mini;
    SVector<TScalar, 3> r;
    r[0] = std::numeric_limits<TScalar>::max();
    math::polynomial::ForEachRoot(
        [&](TScalar t) {
            if (std::isnan(t))
                return true;
            // 3. Compute barycentric coordinates of intersection point at earliest root
            auto uvw = r.template Slice<2, 1>(1, 0);
            // TODO: Compute barycentric coordinates!!
            // 4. Check if intersection point is inside both edges
            bool const bIsInsideEdge1 = uvw[0] >= TScalar(0) and uvw[0] <= TScalar(1);
            bool const bIsInsideEdge2 = uvw[1] >= TScalar(0) and uvw[1] <= TScalar(1);
            if (bIsInsideEdge1 and bIsInsideEdge2)
                r[0] = std::min(r[0], t);
            return false;
        },
        coeffs,
        TScalar(0),
        TScalar(1));
    // Compute return value
    if (r[0] == std::numeric_limits<TScalar>::max())
        r[0] = TScalar(-1);
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
    class TScalar = typename TP1T::Scalar>
std::array<TScalar, 4> EdgeEdgeCcdUnivariatePolynomial(
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
    TScalar const z1  = z0 * Q1T[2];
    TScalar const z2  = P1T[0] * P2T[2];
    TScalar const z3  = z2 * Q2T[1];
    TScalar const z4  = Q1T[1] * Q2T[2];
    TScalar const z5  = z4 * P1T[0];
    TScalar const z6  = P1T[1] * P2T[0];
    TScalar const z7  = z6 * Q2T[2];
    TScalar const z8  = P1T[1] * P2T[2];
    TScalar const z9  = z8 * Q1T[0];
    TScalar const z10 = Q1T[2] * Q2T[0];
    TScalar const z11 = z10 * P1T[1];
    TScalar const z12 = P1T[2] * P2T[0];
    TScalar const z13 = z12 * Q1T[1];
    TScalar const z14 = P1T[2] * P2T[1];
    TScalar const z15 = z14 * Q2T[0];
    TScalar const z16 = Q1T[0] * Q2T[1];
    TScalar const z17 = z16 * P1T[2];
    TScalar const z18 = Q1T[2] * Q2T[1];
    TScalar const z19 = z18 * P2T[0];
    TScalar const z20 = Q1T[0] * Q2T[2];
    TScalar const z21 = z20 * P2T[1];
    TScalar const z22 = Q1T[1] * Q2T[0];
    TScalar const z23 = z22 * P2T[2];
    TScalar const z24 = z0 * Q2T[2];
    TScalar const z25 = z2 * Q1T[1];
    TScalar const z26 = z18 * P1T[0];
    TScalar const z27 = z6 * Q1T[2];
    TScalar const z28 = z8 * Q2T[0];
    TScalar const z29 = z20 * P1T[1];
    TScalar const z30 = z12 * Q2T[1];
    TScalar const z31 = z14 * Q1T[0];
    TScalar const z32 = z22 * P1T[2];
    TScalar const z33 = z4 * P2T[0];
    TScalar const z34 = z10 * P2T[1];
    TScalar const z35 = z16 * P2T[2];
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
    TScalar const z60 = P1[0] * P2T[1];
    TScalar const z61 = P1[0] * P2T[2];
    TScalar const z62 = P1[1] * P2T[0];
    TScalar const z63 = P1[1] * P2T[2];
    TScalar const z64 = P1[2] * P2T[0];
    TScalar const z65 = P1[2] * P2T[1];
    TScalar const z66 = P1T[0] * P2[1];
    TScalar const z67 = P1T[0] * P2[2];
    TScalar const z68 = Q1[1] * Q2T[2];
    TScalar const z69 = Q1T[1] * Q2[2];
    TScalar const z70 = P1T[1] * P2[0];
    TScalar const z71 = P1T[1] * P2[2];
    TScalar const z72 = Q1[2] * Q2T[0];
    TScalar const z73 = Q1T[2] * Q2[0];
    TScalar const z74 = P1T[2] * P2[0];
    TScalar const z75 = P1T[2] * P2[1];
    TScalar const z76 = Q1[0] * Q2T[1];
    TScalar const z77 = Q1T[0] * Q2[1];
    TScalar const z78 = Q1[2] * Q2T[1];
    TScalar const z79 = Q1T[2] * Q2[1];
    TScalar const z80 = Q1[0] * Q2T[2];
    TScalar const z81 = Q1T[0] * Q2[2];
    TScalar const z82 = Q1[1] * Q2T[0];
    TScalar const z83 = Q1T[1] * Q2[0];
    TScalar const z84 =
        z0 * Q1[2] + z10 * P1[1] + z12 * Q1[1] + z14 * Q2[0] + z16 * P1[2] + z18 * P2[0] +
        z2 * Q2[1] + z20 * P2[1] + z22 * P2[2] + z4 * P1[0] + z6 * Q2[2] + z60 * Q1T[2] +
        z61 * Q2T[1] + z62 * Q2T[2] + z63 * Q1T[0] + z64 * Q1T[1] + z65 * Q2T[0] + z66 * Q1T[2] +
        z67 * Q2T[1] + z68 * P1T[0] + z69 * P1T[0] + z70 * Q2T[2] + z71 * Q1T[0] + z72 * P1T[1] +
        z73 * P1T[1] + z74 * Q1T[1] + z75 * Q2T[0] + z76 * P1T[2] + z77 * P1T[2] + z78 * P2T[0] +
        z79 * P2T[0] + z8 * Q1[0] + z80 * P2T[1] + z81 * P2T[1] + z82 * P2T[2] + z83 * P2T[2] -
        P1[0] * P2T[1] * Q2T[2] - P1[0] * P2T[2] * Q1T[1] - P1[0] * Q1T[2] * Q2T[1] -
        P1[1] * P2T[0] * Q1T[2] - P1[1] * P2T[2] * Q2T[0] - P1[1] * Q1T[0] * Q2T[2] -
        P1[2] * P2T[0] * Q2T[1] - P1[2] * P2T[1] * Q1T[0] - P1[2] * Q1T[1] * Q2T[0] -
        P1T[0] * P2[1] * Q2T[2] - P1T[0] * P2[2] * Q1T[1] - P1T[0] * P2T[1] * Q2[2] -
        P1T[0] * P2T[2] * Q1[1] - P1T[0] * Q1[2] * Q2T[1] - P1T[0] * Q1T[2] * Q2[1] -
        P1T[1] * P2[0] * Q1T[2] - P1T[1] * P2[2] * Q2T[0] - P1T[1] * P2T[0] * Q1[2] -
        P1T[1] * P2T[2] * Q2[0] - P1T[1] * Q1[0] * Q2T[2] - P1T[1] * Q1T[0] * Q2[2] -
        P1T[2] * P2[0] * Q2T[1] - P1T[2] * P2[1] * Q1T[0] - P1T[2] * P2T[0] * Q2[1] -
        P1T[2] * P2T[1] * Q1[0] - P1T[2] * Q1[1] * Q2T[0] - P1T[2] * Q1T[1] * Q2[0] -
        P2[0] * Q1T[1] * Q2T[2] - P2[1] * Q1T[2] * Q2T[0] - P2[2] * Q1T[0] * Q2T[1] -
        P2T[0] * Q1[1] * Q2T[2] - P2T[0] * Q1T[1] * Q2[2] - P2T[1] * Q1[2] * Q2T[0] -
        P2T[1] * Q1T[2] * Q2[0] - P2T[2] * Q1[0] * Q2T[1] - P2T[2] * Q1T[0] * Q2[1];
    TScalar const z85  = P1[0] * P2[1];
    TScalar const z86  = z85 * Q1T[2];
    TScalar const z87  = P1[0] * P2[2];
    TScalar const z88  = z87 * Q2T[1];
    TScalar const z89  = z60 * Q1[2];
    TScalar const z90  = z61 * Q2[1];
    TScalar const z91  = z68 * P1[0];
    TScalar const z92  = z69 * P1[0];
    TScalar const z93  = P1[1] * P2[0];
    TScalar const z94  = z93 * Q2T[2];
    TScalar const z95  = P1[1] * P2[2];
    TScalar const z96  = z95 * Q1T[0];
    TScalar const z97  = z62 * Q2[2];
    TScalar const z98  = z63 * Q1[0];
    TScalar const z99  = z72 * P1[1];
    TScalar const z100 = z73 * P1[1];
    TScalar const z101 = P1[2] * P2[0];
    TScalar const z102 = z101 * Q1T[1];
    TScalar const z103 = P1[2] * P2[1];
    TScalar const z104 = z103 * Q2T[0];
    TScalar const z105 = z64 * Q1[1];
    TScalar const z106 = z65 * Q2[0];
    TScalar const z107 = z76 * P1[2];
    TScalar const z108 = z77 * P1[2];
    TScalar const z109 = z66 * Q1[2];
    TScalar const z110 = z67 * Q2[1];
    TScalar const z111 = Q1[1] * Q2[2];
    TScalar const z112 = z111 * P1T[0];
    TScalar const z113 = z70 * Q2[2];
    TScalar const z114 = z71 * Q1[0];
    TScalar const z115 = Q1[2] * Q2[0];
    TScalar const z116 = z115 * P1T[1];
    TScalar const z117 = z74 * Q1[1];
    TScalar const z118 = z75 * Q2[0];
    TScalar const z119 = Q1[0] * Q2[1];
    TScalar const z120 = z119 * P1T[2];
    TScalar const z121 = z78 * P2[0];
    TScalar const z122 = z79 * P2[0];
    TScalar const z123 = z80 * P2[1];
    TScalar const z124 = z81 * P2[1];
    TScalar const z125 = z82 * P2[2];
    TScalar const z126 = z83 * P2[2];
    TScalar const z127 = Q1[2] * Q2[1];
    TScalar const z128 = z127 * P2T[0];
    TScalar const z129 = Q1[0] * Q2[2];
    TScalar const z130 = z129 * P2T[1];
    TScalar const z131 = Q1[1] * Q2[0];
    TScalar const z132 = z131 * P2T[2];
    TScalar const z133 = z85 * Q2T[2];
    TScalar const z134 = z87 * Q1T[1];
    TScalar const z135 = z60 * Q2[2];
    TScalar const z136 = z61 * Q1[1];
    TScalar const z137 = z78 * P1[0];
    TScalar const z138 = z79 * P1[0];
    TScalar const z139 = z93 * Q1T[2];
    TScalar const z140 = z95 * Q2T[0];
    TScalar const z141 = z62 * Q1[2];
    TScalar const z142 = z63 * Q2[0];
    TScalar const z143 = z80 * P1[1];
    TScalar const z144 = z81 * P1[1];
    TScalar const z145 = z101 * Q2T[1];
    TScalar const z146 = z103 * Q1T[0];
    TScalar const z147 = z64 * Q2[1];
    TScalar const z148 = z65 * Q1[0];
    TScalar const z149 = z82 * P1[2];
    TScalar const z150 = z83 * P1[2];
    TScalar const z151 = z66 * Q2[2];
    TScalar const z152 = z67 * Q1[1];
    TScalar const z153 = z127 * P1T[0];
    TScalar const z154 = z70 * Q1[2];
    TScalar const z155 = z71 * Q2[0];
    TScalar const z156 = z129 * P1T[1];
    TScalar const z157 = z74 * Q2[1];
    TScalar const z158 = z75 * Q1[0];
    TScalar const z159 = z131 * P1T[2];
    TScalar const z160 = z68 * P2[0];
    TScalar const z161 = z69 * P2[0];
    TScalar const z162 = z72 * P2[1];
    TScalar const z163 = z73 * P2[1];
    TScalar const z164 = z76 * P2[2];
    TScalar const z165 = z77 * P2[2];
    TScalar const z166 = z111 * P2T[0];
    TScalar const z167 = z115 * P2T[1];
    TScalar const z168 = z119 * P2T[2];
    sigma[0] = -z1 - z11 - z13 - z15 - z17 - z19 - z21 - z23 + z24 + z25 + z26 + z27 + z28 + z29 -
               z3 + z30 + z31 + z32 + z33 + z34 + z35 - z5 - z7 - z9;
    sigma[1] = z36 + z37 + z38 + z39 + z40 + z41 + z42 + z43 + z44 + z45 + z46 + z47 - z48 - z49 -
               z50 - z51 - z52 - z53 - z54 - z55 - z56 - z57 - z58 - z59 - z84;
    sigma[2] =
        -2 * z0 * Q2[2] - 2 * z10 * P2[1] - z100 - z102 - z104 - z105 - z106 - z107 - z108 - z109 -
        z110 - z112 - z113 - z114 - z116 - z117 - z118 - 2 * z12 * Q2[1] - z120 - z121 - z122 -
        z123 - z124 - z125 - z126 - z128 - z130 - z132 + z133 + z134 + z135 + z136 + z137 + z138 +
        z139 - 2 * z14 * Q1[0] + z140 + z141 + z142 + z143 + z144 + z145 + z146 + z147 + z148 +
        z149 + z150 + z151 + z152 + z153 + z154 + z155 + z156 + z157 + z158 + z159 -
        2 * z16 * P2[2] + z160 + z161 + z162 + z163 + z164 + z165 + z166 + z167 + z168 -
        2 * z18 * P1[0] - 2 * z2 * Q1[1] - 2 * z20 * P1[1] - 2 * z22 * P1[2] - z36 - z37 - z38 -
        z39 - 2 * z4 * P2[0] - z40 - z41 - z42 - z43 - z44 - z45 - z46 - z47 + z48 + z49 + z50 +
        z51 + z52 + z53 + z54 + z55 + z56 + z57 + z58 + z59 - 2 * z6 * Q1[2] - 2 * z60 * Q2T[2] -
        2 * z61 * Q1T[1] - 2 * z62 * Q1T[2] - 2 * z63 * Q2T[0] - 2 * z64 * Q2T[1] -
        2 * z65 * Q1T[0] - 2 * z66 * Q2T[2] - 2 * z67 * Q1T[1] - 2 * z68 * P2T[0] -
        2 * z69 * P2T[0] - 2 * z70 * Q1T[2] - 2 * z71 * Q2T[0] - 2 * z72 * P2T[1] -
        2 * z73 * P2T[1] - 2 * z74 * Q2T[1] - 2 * z75 * Q1T[0] - 2 * z76 * P2T[2] -
        2 * z77 * P2T[2] - 2 * z78 * P1T[0] - 2 * z79 * P1T[0] - 2 * z8 * Q2[0] - 2 * z80 * P1T[1] -
        2 * z81 * P1T[1] - 2 * z82 * P1T[2] - 2 * z83 * P1T[2] - z86 - z88 - z89 - z90 - z91 - z92 -
        z94 - z96 - z97 - z98 - z99 + 2 * P1[0] * P2T[1] * Q1T[2] + 2 * P1[0] * P2T[2] * Q2T[1] +
        2 * P1[0] * Q1T[1] * Q2T[2] + 2 * P1[1] * P2T[0] * Q2T[2] + 2 * P1[1] * P2T[2] * Q1T[0] +
        2 * P1[1] * Q1T[2] * Q2T[0] + 2 * P1[2] * P2T[0] * Q1T[1] + 2 * P1[2] * P2T[1] * Q2T[0] +
        2 * P1[2] * Q1T[0] * Q2T[1] + 2 * P1T[0] * P2[1] * Q1T[2] + 2 * P1T[0] * P2[2] * Q2T[1] +
        2 * P1T[0] * P2T[1] * Q1[2] + 2 * P1T[0] * P2T[2] * Q2[1] + 2 * P1T[0] * Q1[1] * Q2T[2] +
        2 * P1T[0] * Q1T[1] * Q2[2] + 2 * P1T[1] * P2[0] * Q2T[2] + 2 * P1T[1] * P2[2] * Q1T[0] +
        2 * P1T[1] * P2T[0] * Q2[2] + 2 * P1T[1] * P2T[2] * Q1[0] + 2 * P1T[1] * Q1[2] * Q2T[0] +
        2 * P1T[1] * Q1T[2] * Q2[0] + 2 * P1T[2] * P2[0] * Q1T[1] + 2 * P1T[2] * P2[1] * Q2T[0] +
        2 * P1T[2] * P2T[0] * Q1[1] + 2 * P1T[2] * P2T[1] * Q2[0] + 2 * P1T[2] * Q1[0] * Q2T[1] +
        2 * P1T[2] * Q1T[0] * Q2[1] + 2 * P2[0] * Q1T[2] * Q2T[1] + 2 * P2[1] * Q1T[0] * Q2T[2] +
        2 * P2[2] * Q1T[1] * Q2T[0] + 2 * P2T[0] * Q1[2] * Q2T[1] + 2 * P2T[0] * Q1T[2] * Q2[1] +
        2 * P2T[1] * Q1[0] * Q2T[2] + 2 * P2T[1] * Q1T[0] * Q2[2] + 2 * P2T[2] * Q1[1] * Q2T[0] +
        2 * P2T[2] * Q1T[1] * Q2[0];
    sigma[3] = z1 + z100 - z101 * Q1[1] + z102 - z103 * Q2[0] + z104 + z105 + z106 + z107 + z108 +
               z109 + z11 + z110 - z111 * P1[0] + z112 + z113 + z114 - z115 * P1[1] + z116 + z117 +
               z118 - z119 * P1[2] + z120 + z121 + z122 + z123 + z124 + z125 + z126 - z127 * P2[0] +
               z128 - z129 * P2[1] + z13 + z130 - z131 * P2[2] + z132 - z133 - z134 - z135 - z136 -
               z137 - z138 - z139 - z140 - z141 - z142 - z143 - z144 - z145 - z146 - z147 - z148 -
               z149 + z15 - z150 - z151 - z152 - z153 - z154 - z155 - z156 - z157 - z158 - z159 -
               z160 - z161 - z162 - z163 - z164 - z165 - z166 - z167 - z168 + z17 + z19 + z21 +
               z23 - z24 - z25 - z26 - z27 - z28 - z29 + z3 - z30 - z31 - z32 - z33 - z34 - z35 +
               z5 + z7 - z84 - z85 * Q1[2] + z86 - z87 * Q2[1] + z88 + z89 + z9 + z90 + z91 + z92 -
               z93 * Q2[2] + z94 - z95 * Q1[0] + z96 + z97 + z98 + z99 + P1[0] * P2[1] * Q2[2] +
               P1[0] * P2[2] * Q1[1] + P1[0] * Q1[2] * Q2[1] + P1[1] * P2[0] * Q1[2] +
               P1[1] * P2[2] * Q2[0] + P1[1] * Q1[0] * Q2[2] + P1[2] * P2[0] * Q2[1] +
               P1[2] * P2[1] * Q1[0] + P1[2] * Q1[1] * Q2[0] + P2[0] * Q1[1] * Q2[2] +
               P2[1] * Q1[2] * Q2[0] + P2[2] * Q1[0] * Q2[1];
    return sigma;
}

} // namespace detail

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_EDGEEDGECCD_H
