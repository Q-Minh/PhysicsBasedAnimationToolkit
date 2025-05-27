
/**
 * @file Tetrahedron.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Tetrahedron finite element
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef PBAT_FEM_TETRAHEDRON_H
#define PBAT_FEM_TETRAHEDRON_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "QuadratureRules.h"

#include <array>

namespace pbat {
namespace fem {

namespace detail {

template <int Order>
struct Tetrahedron;

} // namespace detail

/**
 * @brief Tetrahedron finite element
 * 
 * Satisfies concept CElement  
 *
 * @tparam Order Polynomial order
 */
template <int Order>
using Tetrahedron = typename detail::Tetrahedron<Order>;

namespace detail {

template <>
struct Tetrahedron<1>
{
    using AffineBaseType = Tetrahedron<1>;

    static int constexpr kOrder = 1;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 4;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,0,1,0,0,0,1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,1,2,3}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = true;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        Nm[0] = -X[0] - X[1] - X[2] + 1;
        Nm[1] = X[0];
        Nm[2] = X[1];
        Nm[3] = X[2];
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        GNp[0] = -1;
        GNp[1] = 1;
        GNp[2] = 0;
        GNp[3] = 0;
        GNp[4] = -1;
        GNp[5] = 0;
        GNp[6] = 1;
        GNp[7] = 0;
        GNp[8] = -1;
        GNp[9] = 0;
        GNp[10] = 0;
        GNp[11] = 1;
        return GNm;
    }
};

template <>
struct Tetrahedron<2>
{
    using AffineBaseType = Tetrahedron<1>;

    static int constexpr kOrder = 2;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 10;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,0,1,0,1,1,0,0,2,0,0,0,1,1,0,1,0,1,1,0,0,2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,2,5,9}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] + X[1] + X[2] - 1;
        auto const a1 = 2*X[1];
        auto const a2 = 2*X[2];
        auto const a3 = 2*X[0] - 1;
        auto const a4 = 4*a0;
        auto const a5 = 4*X[0];
        Nm[0] = a0*(a1 + a2 + a3);
        Nm[1] = -a4*X[0];
        Nm[2] = a3*X[0];
        Nm[3] = -a4*X[1];
        Nm[4] = a5*X[1];
        Nm[5] = (a1 - 1)*X[1];
        Nm[6] = -a4*X[2];
        Nm[7] = a5*X[2];
        Nm[8] = 4*X[1]*X[2];
        Nm[9] = (a2 - 1)*X[2];
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        Scalar const a0 = 4*X[0];
        Scalar const a1 = 4*X[1];
        Scalar const a2 = 4*X[2];
        Scalar const a3 = a1 + a2;
        Scalar const a4 = a0 + a3 - 3;
        Scalar const a5 = -a1;
        Scalar const a6 = -a2;
        Scalar const a7 = -a0;
        Scalar const a8 = a0 - 4;
        GNp[0] = a4;
        GNp[1] = -a3 - 8*X[0] + 4;
        GNp[2] = a0 - 1;
        GNp[3] = a5;
        GNp[4] = a1;
        GNp[5] = 0;
        GNp[6] = a6;
        GNp[7] = a2;
        GNp[8] = 0;
        GNp[9] = 0;
        GNp[10] = a4;
        GNp[11] = a7;
        GNp[12] = 0;
        GNp[13] = -a2 - a8 - 8*X[1];
        GNp[14] = a0;
        GNp[15] = a1 - 1;
        GNp[16] = a6;
        GNp[17] = 0;
        GNp[18] = a2;
        GNp[19] = 0;
        GNp[20] = a4;
        GNp[21] = a7;
        GNp[22] = 0;
        GNp[23] = a5;
        GNp[24] = 0;
        GNp[25] = 0;
        GNp[26] = -a1 - a8 - 8*X[2];
        GNp[27] = a0;
        GNp[28] = a1;
        GNp[29] = a2 - 1;
        return GNm;
    }
};

template <>
struct Tetrahedron<3>
{
    using AffineBaseType = Tetrahedron<1>;

    static int constexpr kOrder = 3;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 20;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,3,0,0,0,1,0,1,1,0,2,1,0,0,2,0,1,2,0,0,3,0,0,0,1,1,0,1,2,0,1,0,1,1,1,1,1,0,2,1,0,0,2,1,0,2,0,1,2,0,0,3}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,3,9,19}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = 3*X[0];
        auto const a1 = a0 - 1;
        auto const a2 = 3*X[1];
        auto const a3 = 3*X[2];
        auto const a4 = a2 + a3;
        auto const a5 = X[0] + X[1] + X[2] - 1;
        auto const a6 = a0 - 2;
        auto const a7 = a5*(a4 + a6);
        auto const a8 = (9.0/2.0)*X[0];
        auto const a9 = a1*a8;
        auto const a10 = (9.0/2.0)*X[1];
        auto const a11 = 27*a5*X[0];
        auto const a12 = a2 - 1;
        auto const a13 = a10*a12;
        auto const a14 = a12*X[1];
        auto const a15 = (9.0/2.0)*X[2];
        auto const a16 = 27*X[1]*X[2];
        auto const a17 = a3 - 1;
        auto const a18 = a17*X[2];
        Nm[0] = -1.0/2.0*a7*(a1 + a4);
        Nm[1] = a7*a8;
        Nm[2] = -a5*a9;
        Nm[3] = (1.0/2.0)*a1*a6*X[0];
        Nm[4] = a10*a7;
        Nm[5] = -a11*X[1];
        Nm[6] = a9*X[1];
        Nm[7] = -a13*a5;
        Nm[8] = a14*a8;
        Nm[9] = (1.0/2.0)*a14*(a2 - 2);
        Nm[10] = a15*a7;
        Nm[11] = -a11*X[2];
        Nm[12] = a9*X[2];
        Nm[13] = -a16*a5;
        Nm[14] = a16*X[0];
        Nm[15] = a13*X[2];
        Nm[16] = -a15*a17*a5;
        Nm[17] = a18*a8;
        Nm[18] = a10*a18;
        Nm[19] = (1.0/2.0)*a18*(a3 - 2);
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        Scalar const a0 = X[0] + X[1] + X[2] - 1;
        Scalar const a1 = (3.0/2.0)*X[0];
        Scalar const a2 = a1 - 1.0/2.0;
        Scalar const a3 = (3.0/2.0)*X[1];
        Scalar const a4 = (3.0/2.0)*X[2];
        Scalar const a5 = a3 + a4;
        Scalar const a6 = -a2 - a5;
        Scalar const a7 = 3*X[1];
        Scalar const a8 = 3*X[2];
        Scalar const a9 = 3*X[0] - 2;
        Scalar const a10 = a7 + a8 + a9;
        Scalar const a11 = 3*a0*a6 + a10*a6 + a10*(-a1 - a5 + 3.0/2.0);
        Scalar const a12 = (9.0/2.0)*X[0];
        Scalar const a13 = (9.0/2.0)*X[1];
        Scalar const a14 = (9.0/2.0)*X[2];
        Scalar const a15 = a10*(a12 + a13 + a14 - 9.0/2.0);
        Scalar const a16 = (27.0/2.0)*X[0];
        Scalar const a17 = (27.0/2.0)*X[1];
        Scalar const a18 = (27.0/2.0)*X[2];
        Scalar const a19 = a16 + a17 + a18;
        Scalar const a20 = a19 - 27.0/2.0;
        Scalar const a21 = a19 - 9;
        Scalar const a22 = a20*X[0] + a21*X[0];
        Scalar const a23 = a16 - 9.0/2.0;
        Scalar const a24 = -a23;
        Scalar const a25 = a24*X[0];
        Scalar const a26 = -a20;
        Scalar const a27 = a20*X[1] + a21*X[1];
        Scalar const a28 = 27*X[0];
        Scalar const a29 = a28*X[1];
        Scalar const a30 = -a29;
        Scalar const a31 = 27*X[1];
        Scalar const a32 = -a28 - a31 - 27*X[2] + 27;
        Scalar const a33 = a32*X[1];
        Scalar const a34 = a16*X[1];
        Scalar const a35 = a17 - 9.0/2.0;
        Scalar const a36 = -a35;
        Scalar const a37 = a36*X[1];
        Scalar const a38 = a35*X[1];
        Scalar const a39 = a20*X[2] + a21*X[2];
        Scalar const a40 = a28*X[2];
        Scalar const a41 = -a40;
        Scalar const a42 = a32*X[2];
        Scalar const a43 = a16*X[2];
        Scalar const a44 = a31*X[2];
        Scalar const a45 = -a44;
        Scalar const a46 = a18 - 9.0/2.0;
        Scalar const a47 = -a46;
        Scalar const a48 = a47*X[2];
        Scalar const a49 = a46*X[2];
        Scalar const a50 = a32*X[0];
        Scalar const a51 = a23*X[0];
        Scalar const a52 = a17*X[2];
        GNp[0] = a11;
        GNp[1] = a15 + a22;
        GNp[2] = a0*a24 + a25 + a26*X[0];
        GNp[3] = a2*a9 + (a12 - 3)*X[0] + (a12 - 3.0/2.0)*X[0];
        GNp[4] = a27;
        GNp[5] = a30 + a33;
        GNp[6] = a23*X[1] + a34;
        GNp[7] = a37;
        GNp[8] = a38;
        GNp[9] = 0;
        GNp[10] = a39;
        GNp[11] = a41 + a42;
        GNp[12] = a23*X[2] + a43;
        GNp[13] = a45;
        GNp[14] = a44;
        GNp[15] = 0;
        GNp[16] = a48;
        GNp[17] = a49;
        GNp[18] = 0;
        GNp[19] = 0;
        GNp[20] = a11;
        GNp[21] = a22;
        GNp[22] = a25;
        GNp[23] = 0;
        GNp[24] = a15 + a27;
        GNp[25] = a30 + a50;
        GNp[26] = a51;
        GNp[27] = a0*a36 + a26*X[1] + a37;
        GNp[28] = a34 + a35*X[0];
        GNp[29] = (a13 - 3)*X[1] + (a13 - 3.0/2.0)*X[1] + (a3 - 1.0/2.0)*(a7 - 2);
        GNp[30] = a39;
        GNp[31] = a41;
        GNp[32] = 0;
        GNp[33] = a42 + a45;
        GNp[34] = a40;
        GNp[35] = a35*X[2] + a52;
        GNp[36] = a48;
        GNp[37] = 0;
        GNp[38] = a49;
        GNp[39] = 0;
        GNp[40] = a11;
        GNp[41] = a22;
        GNp[42] = a25;
        GNp[43] = 0;
        GNp[44] = a27;
        GNp[45] = a30;
        GNp[46] = 0;
        GNp[47] = a37;
        GNp[48] = 0;
        GNp[49] = 0;
        GNp[50] = a15 + a39;
        GNp[51] = a41 + a50;
        GNp[52] = a51;
        GNp[53] = a33 + a45;
        GNp[54] = a29;
        GNp[55] = a38;
        GNp[56] = a0*a47 + a26*X[2] + a48;
        GNp[57] = a43 + a46*X[0];
        GNp[58] = a46*X[1] + a52;
        GNp[59] = (a14 - 3)*X[2] + (a14 - 3.0/2.0)*X[2] + (a4 - 1.0/2.0)*(a8 - 2);
        return GNm;
    }
};

} // namespace detail
} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_TETRAHEDRON_H
