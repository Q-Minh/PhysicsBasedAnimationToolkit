
/**
 * @file Hexahedron.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Hexahedron finite element
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef PBAT_FEM_HEXAHEDRON_H
#define PBAT_FEM_HEXAHEDRON_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "QuadratureRules.h"

#include <array>

namespace pbat {
namespace fem {

namespace detail {

template <int Order>
struct Hexahedron;

} // namespace detail

/**
 * @brief Hexahedron finite element
 * 
 * Satisfies concept CElement  
 *
 * @tparam Order Polynomial order
 */
template <int Order>
using Hexahedron = typename detail::Hexahedron<Order>;

namespace detail {

template <>
struct Hexahedron<1>
{
    using AffineBaseType = Hexahedron<1>;

    static int constexpr kOrder = 1;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 8;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,1,2,3,4,5,6,7}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const X = X_.reshaped();
        auto const a0 = X[0] - 1;
        auto const a1 = X[1] - 1;
        auto const a2 = X[2] - 1;
        auto const a3 = a1*a2;
        auto const a4 = a2*X[1];
        auto const a5 = a1*X[2];
        auto const a6 = X[1]*X[2];
        Nm[0] = -a0*a3;
        Nm[1] = a3*X[0];
        Nm[2] = a0*a4;
        Nm[3] = -a4*X[0];
        Nm[4] = a0*a5;
        Nm[5] = -a5*X[0];
        Nm[6] = -a0*a6;
        Nm[7] = a6*X[0];
        return Nm;
#include "pbat/warning/Pop.h"
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Matrix<TScalar, kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        [[maybe_unused]] auto const X = X_.reshaped();
        auto const a0 = X[2] - 1;
        auto const a1 = X[1] - 1;
        auto const a2 = -a1;
        auto const a3 = -a0;
        auto const a4 = X[1]*X[2];
        auto const a5 = X[0] - 1;
        auto const a6 = -a5;
        auto const a7 = X[0]*X[2];
        auto const a8 = X[0]*X[1];
        GNp[0] = a0*a2;
        GNp[1] = a0*a1;
        GNp[2] = a0*X[1];
        GNp[3] = a3*X[1];
        GNp[4] = a1*X[2];
        GNp[5] = a2*X[2];
        GNp[6] = -a4;
        GNp[7] = a4;
        GNp[8] = a0*a6;
        GNp[9] = a0*X[0];
        GNp[10] = a0*a5;
        GNp[11] = a3*X[0];
        GNp[12] = a5*X[2];
        GNp[13] = -a7;
        GNp[14] = a6*X[2];
        GNp[15] = a7;
        GNp[16] = a1*a6;
        GNp[17] = a1*X[0];
        GNp[18] = a5*X[1];
        GNp[19] = -a8;
        GNp[20] = a1*a5;
        GNp[21] = a2*X[0];
        GNp[22] = a6*X[1];
        GNp[23] = a8;
        return GNm;
#include "pbat/warning/Pop.h"
    }
};

template <>
struct Hexahedron<2>
{
    using AffineBaseType = Hexahedron<1>;

    static int constexpr kOrder = 2;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 27;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,0,1,0,1,1,0,2,1,0,0,2,0,1,2,0,2,2,0,0,0,1,1,0,1,2,0,1,0,1,1,1,1,1,2,1,1,0,2,1,1,2,1,2,2,1,0,0,2,1,0,2,2,0,2,0,1,2,1,1,2,2,1,2,0,2,2,1,2,2,2,2,2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,2,6,8,18,20,24,26}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const X = X_.reshaped();
        auto const a0 = 2*X[0] - 1;
        auto const a1 = 2*X[1] - 1;
        auto const a2 = 2*X[2] - 1;
        auto const a3 = a0*a1*a2;
        auto const a4 = X[0] - 1;
        auto const a5 = X[1] - 1;
        auto const a6 = X[2] - 1;
        auto const a7 = a4*a5*a6;
        auto const a8 = a2*a7;
        auto const a9 = 4*a1;
        auto const a10 = a3*a6;
        auto const a11 = a5*X[0];
        auto const a12 = a0*X[1];
        auto const a13 = 4*a12;
        auto const a14 = X[0]*X[1];
        auto const a15 = 16*a14;
        auto const a16 = a2*a6;
        auto const a17 = a11*a13;
        auto const a18 = a10*X[1];
        auto const a19 = a4*a9;
        auto const a20 = a14*a19;
        auto const a21 = a7*X[2];
        auto const a22 = a0*a9;
        auto const a23 = 16*a21;
        auto const a24 = a6*X[2];
        auto const a25 = a11*a24;
        auto const a26 = a3*X[2];
        auto const a27 = a26*a4;
        auto const a28 = a2*X[2];
        auto const a29 = a11*a28;
        Nm[0] = a3*a7;
        Nm[1] = -a8*a9*X[0];
        Nm[2] = a10*a11;
        Nm[3] = -a13*a8;
        Nm[4] = a15*a8;
        Nm[5] = -a16*a17;
        Nm[6] = a18*a4;
        Nm[7] = -a16*a20;
        Nm[8] = a18*X[0];
        Nm[9] = -a21*a22;
        Nm[10] = a1*a23*X[0];
        Nm[11] = -a22*a25;
        Nm[12] = a12*a23;
        Nm[13] = -64*a14*a21;
        Nm[14] = 16*a12*a25;
        Nm[15] = -a12*a19*a24;
        Nm[16] = a1*a15*a24*a4;
        Nm[17] = -a14*a22*a24;
        Nm[18] = a27*a5;
        Nm[19] = -a19*a29;
        Nm[20] = a11*a26;
        Nm[21] = -a13*a28*a4*a5;
        Nm[22] = 16*a29*a4*X[1];
        Nm[23] = -a17*a28;
        Nm[24] = a27*X[1];
        Nm[25] = -a20*a28;
        Nm[26] = a14*a26;
        return Nm;
#include "pbat/warning/Pop.h"
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Matrix<TScalar, kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        [[maybe_unused]] auto const X = X_.reshaped();
        auto const a0 = 4*X[1] - 2;
        auto const a1 = X[0] - 1;
        auto const a2 = 2*X[2];
        auto const a3 = a2 - 1;
        auto const a4 = X[1] - 1;
        auto const a5 = X[2] - 1;
        auto const a6 = a4*a5;
        auto const a7 = a3*a6;
        auto const a8 = a1*a7;
        auto const a9 = 2*X[0];
        auto const a10 = 2*X[1];
        auto const a11 = a10 - 1;
        auto const a12 = a11*(a9 - 1);
        auto const a13 = a12*a7;
        auto const a14 = 4 - 8*X[1];
        auto const a15 = a7*X[0];
        auto const a16 = a6*X[1];
        auto const a17 = 8 - 16*X[2];
        auto const a18 = a1*a17;
        auto const a19 = 4 - 8*X[0];
        auto const a20 = a19*X[1];
        auto const a21 = a20*a7;
        auto const a22 = 32*X[2] - 16;
        auto const a23 = a16*a22;
        auto const a24 = a17*X[0];
        auto const a25 = a5*X[1];
        auto const a26 = a25*a3;
        auto const a27 = a1*a26;
        auto const a28 = a12*a3;
        auto const a29 = a25*a28;
        auto const a30 = a26*X[0];
        auto const a31 = 8 - 16*X[1];
        auto const a32 = a6*X[2];
        auto const a33 = a1*a32;
        auto const a34 = a11*a19;
        auto const a35 = a32*a34;
        auto const a36 = 32*X[1];
        auto const a37 = a36 - 16;
        auto const a38 = a32*X[0];
        auto const a39 = 32*X[0];
        auto const a40 = a39 - 32;
        auto const a41 = a16*X[2];
        auto const a42 = a39 - 16;
        auto const a43 = a41*a42;
        auto const a44 = a25*X[2];
        auto const a45 = a44*X[0];
        auto const a46 = 64 - 64*X[0];
        auto const a47 = a1*a44;
        auto const a48 = a20*X[2];
        auto const a49 = a11*a5;
        auto const a50 = a48*a49;
        auto const a51 = a4*X[2];
        auto const a52 = a3*a51;
        auto const a53 = a1*a52;
        auto const a54 = a28*a51;
        auto const a55 = a52*X[0];
        auto const a56 = a51*X[1];
        auto const a57 = a3*a48;
        auto const a58 = a4*a57;
        auto const a59 = a22*a56;
        auto const a60 = X[1]*X[2];
        auto const a61 = a3*a60;
        auto const a62 = a1*a61;
        auto const a63 = a28*a60;
        auto const a64 = a61*X[0];
        auto const a65 = 4*X[0] - 2;
        auto const a66 = a1*a5;
        auto const a67 = a28*a66;
        auto const a68 = a6*X[0];
        auto const a69 = a3*a66;
        auto const a70 = a14*X[0];
        auto const a71 = a69*a70;
        auto const a72 = a5*X[0];
        auto const a73 = a28*a72;
        auto const a74 = a1*a22;
        auto const a75 = a25*X[0];
        auto const a76 = a20*a3;
        auto const a77 = 8 - 16*X[0];
        auto const a78 = a66*X[2];
        auto const a79 = a34*a78;
        auto const a80 = a37*X[0];
        auto const a81 = a78*a80;
        auto const a82 = a34*a72*X[2];
        auto const a83 = a1*X[2];
        auto const a84 = a28*a83;
        auto const a85 = a51*X[0];
        auto const a86 = a3*a70;
        auto const a87 = a83*a86;
        auto const a88 = a28*X[0];
        auto const a89 = a88*X[2];
        auto const a90 = a60*X[0];
        auto const a91 = a12*a6;
        auto const a92 = a1*a4;
        auto const a93 = a28*a92;
        auto const a94 = a1*a31;
        auto const a95 = a86*a92;
        auto const a96 = a4*a88;
        auto const a97 = a16*a77;
        auto const a98 = a76*a92;
        auto const a99 = a4*X[0];
        auto const a100 = a74*a99*X[1];
        auto const a101 = a16*X[0];
        auto const a102 = a76*a99;
        auto const a103 = a10*a12;
        auto const a104 = a1*X[1];
        auto const a105 = a104*a28;
        auto const a106 = a104*a86;
        auto const a107 = a12*a9;
        auto const a108 = a88*X[1];
        auto const a109 = a1*a34;
        auto const a110 = a1*a80;
        auto const a111 = a1*a37;
        auto const a112 = a1*a42;
        auto const a113 = a56*X[0];
        auto const a114 = a11*a48;
        auto const a115 = a20*a49;
        GNp[0] = a0*a8 + a13;
        GNp[1] = a14*a15 + a14*a8;
        GNp[2] = a0*a15 + a13;
        GNp[3] = a16*a18 + a21;
        GNp[4] = a1*a23 + a23*X[0];
        GNp[5] = a16*a24 + a21;
        GNp[6] = a0*a27 + a29;
        GNp[7] = a14*a27 + a14*a30;
        GNp[8] = a0*a30 + a29;
        GNp[9] = a31*a33 + a35;
        GNp[10] = a33*a37 + a37*a38;
        GNp[11] = a31*a38 + a35;
        GNp[12] = a40*a41 + a43;
        GNp[13] = a41*a46 + a45*(64 - 64*X[1]);
        GNp[14] = a43 + a45*(a36 - 32);
        GNp[15] = a31*a47 + a50;
        GNp[16] = a37*a45 + a37*a47;
        GNp[17] = a31*a45 + a50;
        GNp[18] = a0*a53 + a54;
        GNp[19] = a14*a53 + a14*a55;
        GNp[20] = a0*a55 + a54;
        GNp[21] = a18*a56 + a58;
        GNp[22] = a1*a59 + a59*X[0];
        GNp[23] = a24*a56 + a58;
        GNp[24] = a0*a62 + a63;
        GNp[25] = a14*a62 + a14*a64;
        GNp[26] = a0*a64 + a63;
        GNp[27] = a65*a8 + a67;
        GNp[28] = a18*a68 + a71;
        GNp[29] = a15*a65 + a73;
        GNp[30] = a19*a8 + a20*a69;
        GNp[31] = a68*a74 + a74*a75;
        GNp[32] = a15*a19 + a72*a76;
        GNp[33] = a27*a65 + a67;
        GNp[34] = a18*a75 + a71;
        GNp[35] = a30*a65 + a73;
        GNp[36] = a33*a77 + a79;
        GNp[37] = a38*a40 + a81;
        GNp[38] = a38*a77 + a82;
        GNp[39] = a33*a42 + a42*a47;
        GNp[40] = a38*a46 + a45*a46;
        GNp[41] = a38*a42 + a42*a45;
        GNp[42] = a47*a77 + a79;
        GNp[43] = a40*a45 + a81;
        GNp[44] = a45*a77 + a82;
        GNp[45] = a53*a65 + a84;
        GNp[46] = a18*a85 + a87;
        GNp[47] = a55*a65 + a89;
        GNp[48] = a1*a57 + a19*a53;
        GNp[49] = a74*a85 + a74*a90;
        GNp[50] = a19*a55 + a57*X[0];
        GNp[51] = a62*a65 + a84;
        GNp[52] = a18*a90 + a87;
        GNp[53] = a64*a65 + a89;
        GNp[54] = 2*a1*a91 + a93;
        GNp[55] = a68*a94 + a95;
        GNp[56] = a9*a91 + a96;
        GNp[57] = a1*a97 + a98;
        GNp[58] = a100 + a101*a40;
        GNp[59] = a102 + a97*X[0];
        GNp[60] = a103*a66 + a105;
        GNp[61] = a106 + a75*a94;
        GNp[62] = a107*a25 + a108;
        GNp[63] = a109*a51 + a109*a6;
        GNp[64] = a110*a51 + a111*a68;
        GNp[65] = a34*a68 + a34*a85;
        GNp[66] = a112*a16 + a112*a56;
        GNp[67] = a101*a46 + a113*a46;
        GNp[68] = a101*a42 + a113*a42;
        GNp[69] = a1*a114 + a1*a115;
        GNp[70] = a110*a60 + a111*a75;
        GNp[71] = a114*X[0] + a115*X[0];
        GNp[72] = a12*a2*a92 + a93;
        GNp[73] = a85*a94 + a95;
        GNp[74] = a107*a51 + a96;
        GNp[75] = a1*a56*a77 + a98;
        GNp[76] = a100 + a113*a40;
        GNp[77] = a102 + a113*a77;
        GNp[78] = a103*a83 + a105;
        GNp[79] = a106 + a90*a94;
        GNp[80] = a107*a60 + a108;
        return GNm;
#include "pbat/warning/Pop.h"
    }
};

template <>
struct Hexahedron<3>
{
    using AffineBaseType = Hexahedron<1>;

    static int constexpr kOrder = 3;
    static int constexpr kDims  = 3;
    static int constexpr kNodes = 64;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,3,0,0,0,1,0,1,1,0,2,1,0,3,1,0,0,2,0,1,2,0,2,2,0,3,2,0,0,3,0,1,3,0,2,3,0,3,3,0,0,0,1,1,0,1,2,0,1,3,0,1,0,1,1,1,1,1,2,1,1,3,1,1,0,2,1,1,2,1,2,2,1,3,2,1,0,3,1,1,3,1,2,3,1,3,3,1,0,0,2,1,0,2,2,0,2,3,0,2,0,1,2,1,1,2,2,1,2,3,1,2,0,2,2,1,2,2,2,2,2,3,2,2,0,3,2,1,3,2,2,3,2,3,3,2,0,0,3,1,0,3,2,0,3,3,0,3,0,1,3,1,1,3,2,1,3,3,1,3,0,2,3,1,2,3,2,2,3,3,2,3,0,3,3,1,3,3,2,3,3,3,3,3}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,3,12,15,48,51,60,63}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const X = X_.reshaped();
        auto const a0 = X[0] - 1;
        auto const a1 = X[1] - 1;
        auto const a2 = X[2] - 1;
        auto const a3 = 3*X[0];
        auto const a4 = a3 - 2;
        auto const a5 = 3*X[1];
        auto const a6 = a5 - 2;
        auto const a7 = 3*X[2];
        auto const a8 = a7 - 2;
        auto const a9 = a0*a1*a2*a4*a6*a8;
        auto const a10 = a7 - 1;
        auto const a11 = a3 - 1;
        auto const a12 = a5 - 1;
        auto const a13 = a11*a12;
        auto const a14 = a10*a13;
        auto const a15 = (1.0/8.0)*a14;
        auto const a16 = a10*a9;
        auto const a17 = (9.0/8.0)*X[0];
        auto const a18 = a12*a17;
        auto const a19 = a6*a8;
        auto const a20 = a19*a2;
        auto const a21 = a17*a20;
        auto const a22 = a0*a1;
        auto const a23 = a14*a22;
        auto const a24 = a1*a4;
        auto const a25 = a20*X[0];
        auto const a26 = a15*a25;
        auto const a27 = a16*X[1];
        auto const a28 = (9.0/8.0)*a11;
        auto const a29 = (81.0/8.0)*X[0];
        auto const a30 = a10*X[1];
        auto const a31 = a11*a30;
        auto const a32 = (81.0/8.0)*a22*a25;
        auto const a33 = a21*a24;
        auto const a34 = a4*X[1];
        auto const a35 = (9.0/8.0)*a34;
        auto const a36 = a2*a23;
        auto const a37 = a36*a8;
        auto const a38 = a2*a8;
        auto const a39 = a22*a38;
        auto const a40 = a30*a4;
        auto const a41 = a12*a29;
        auto const a42 = a40*a41;
        auto const a43 = a29*X[1];
        auto const a44 = a24*a38;
        auto const a45 = a14*X[1];
        auto const a46 = a17*a45;
        auto const a47 = a0*a20;
        auto const a48 = a15*a34;
        auto const a49 = a18*a40;
        auto const a50 = a9*X[2];
        auto const a51 = a13*X[2];
        auto const a52 = a50*X[1];
        auto const a53 = (81.0/8.0)*a11;
        auto const a54 = (729.0/8.0)*X[0];
        auto const a55 = a25*X[2];
        auto const a56 = a55*X[1];
        auto const a57 = a34*a39;
        auto const a58 = (81.0/8.0)*a51;
        auto const a59 = a54*X[2];
        auto const a60 = a12*a59;
        auto const a61 = (81.0/8.0)*a34;
        auto const a62 = a6*X[2];
        auto const a63 = a4*a62;
        auto const a64 = a2*a22;
        auto const a65 = a63*a64;
        auto const a66 = a29*a62;
        auto const a67 = a2*a24;
        auto const a68 = a14*a67;
        auto const a69 = a17*a62;
        auto const a70 = a40*a64;
        auto const a71 = a59*a6;
        auto const a72 = a43*X[2];
        auto const a73 = a14*a2;
        auto const a74 = a0*a62;
        auto const a75 = a73*a74;
        auto const a76 = a19*X[2];
        auto const a77 = a15*a76;
        auto const a78 = a22*a4;
        auto const a79 = a17*a76;
        auto const a80 = a22*a76;
        auto const a81 = a40*a80;
        auto const a82 = a8*X[2];
        auto const a83 = a0*a76;
        Nm[0] = -a15*a9;
        Nm[1] = a16*a18;
        Nm[2] = -a21*a23;
        Nm[3] = a24*a26;
        Nm[4] = a27*a28;
        Nm[5] = -a27*a29;
        Nm[6] = a31*a32;
        Nm[7] = -a31*a33;
        Nm[8] = -a35*a37;
        Nm[9] = a39*a42;
        Nm[10] = -a37*a43;
        Nm[11] = a44*a46;
        Nm[12] = a47*a48;
        Nm[13] = -a47*a49;
        Nm[14] = a0*a21*a45;
        Nm[15] = -a26*a34;
        Nm[16] = (9.0/8.0)*a13*a50;
        Nm[17] = -a41*a50;
        Nm[18] = a32*a51;
        Nm[19] = -a33*a51;
        Nm[20] = -a52*a53;
        Nm[21] = a52*a54;
        Nm[22] = -729.0/8.0*a11*a22*a56;
        Nm[23] = a24*a53*a56;
        Nm[24] = a57*a58;
        Nm[25] = -a57*a60;
        Nm[26] = a39*a51*a54*X[1];
        Nm[27] = -a43*a44*a51;
        Nm[28] = -a35*a47*a51;
        Nm[29] = a0*a12*a55*a61;
        Nm[30] = -a0*a25*a58*X[1];
        Nm[31] = a21*a34*a51;
        Nm[32] = -9.0/8.0*a36*a63;
        Nm[33] = a10*a41*a65;
        Nm[34] = -a36*a66;
        Nm[35] = a68*a69;
        Nm[36] = (81.0/8.0)*a31*a65;
        Nm[37] = -a70*a71;
        Nm[38] = a31*a64*a71;
        Nm[39] = -a31*a66*a67;
        Nm[40] = -a36*a61*X[2];
        Nm[41] = a60*a70;
        Nm[42] = -a36*a59*X[1];
        Nm[43] = a68*a72;
        Nm[44] = a35*a75;
        Nm[45] = -a2*a42*a74;
        Nm[46] = a43*a75;
        Nm[47] = -a34*a69*a73;
        Nm[48] = a77*a78;
        Nm[49] = -a10*a18*a76*a78;
        Nm[50] = a23*a79;
        Nm[51] = -a24*a77*X[0];
        Nm[52] = -a28*a81;
        Nm[53] = a29*a81;
        Nm[54] = -a29*a31*a80;
        Nm[55] = a24*a31*a79;
        Nm[56] = a23*a35*a82;
        Nm[57] = -a22*a42*a82;
        Nm[58] = a23*a72*a8;
        Nm[59] = -a24*a46*a82;
        Nm[60] = -a48*a83;
        Nm[61] = a49*a83;
        Nm[62] = -a46*a83;
        Nm[63] = a48*a76*X[0];
        return Nm;
#include "pbat/warning/Pop.h"
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Matrix<TScalar, kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X_)
    {
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        [[maybe_unused]] auto const X = X_.reshaped();
        auto const a0 = (9.0/8.0)*X[1] - 3.0/8.0;
        auto const a1 = -a0;
        auto const a2 = X[0] - 1;
        auto const a3 = X[1] - 1;
        auto const a4 = X[2] - 1;
        auto const a5 = 3*X[2];
        auto const a6 = a5 - 2;
        auto const a7 = a3*a4*a6;
        auto const a8 = 3*X[0];
        auto const a9 = a8 - 2;
        auto const a10 = a5 - 1;
        auto const a11 = 3*X[1];
        auto const a12 = a11 - 2;
        auto const a13 = a10*a12;
        auto const a14 = a13*a9;
        auto const a15 = a14*a7;
        auto const a16 = a15*a2;
        auto const a17 = (3.0/8.0)*X[0] - 1.0/8.0;
        auto const a18 = -a17;
        auto const a19 = a13*a2;
        auto const a20 = a18*a19;
        auto const a21 = a11 - 1;
        auto const a22 = a21*a7;
        auto const a23 = 3*a22;
        auto const a24 = a14*a18;
        auto const a25 = (27.0/8.0)*X[1] - 9.0/8.0;
        auto const a26 = a19*a25;
        auto const a27 = a7*a8;
        auto const a28 = a15*X[0];
        auto const a29 = (81.0/8.0)*X[1] - 27.0/8.0;
        auto const a30 = -a29;
        auto const a31 = a7*X[0];
        auto const a32 = a30*a31;
        auto const a33 = (27.0/8.0)*X[0] - 9.0/8.0;
        auto const a34 = -a33;
        auto const a35 = a22*a34;
        auto const a36 = a13*X[0];
        auto const a37 = a17*a22*a8;
        auto const a38 = a14*a17;
        auto const a39 = a7*X[1];
        auto const a40 = (81.0/8.0)*X[2] - 27.0/8.0;
        auto const a41 = a12*a9;
        auto const a42 = a2*a41;
        auto const a43 = a40*a42;
        auto const a44 = a19*a33;
        auto const a45 = a11*a7;
        auto const a46 = a15*X[1];
        auto const a47 = a12*a31;
        auto const a48 = (243.0/8.0)*X[2] - 81.0/8.0;
        auto const a49 = -a48;
        auto const a50 = a11*a2;
        auto const a51 = a49*a50;
        auto const a52 = a31*X[1];
        auto const a53 = a41*a52;
        auto const a54 = a42*a49;
        auto const a55 = (729.0/8.0)*X[2] - 243.0/8.0;
        auto const a56 = a2*a52;
        auto const a57 = a55*a56;
        auto const a58 = (243.0/8.0)*X[0];
        auto const a59 = a58 - 81.0/8.0;
        auto const a60 = a39*a59;
        auto const a61 = -a40;
        auto const a62 = a10*a9;
        auto const a63 = a2*a39;
        auto const a64 = a62*a63;
        auto const a65 = a10*a50;
        auto const a66 = a62*X[1];
        auto const a67 = (243.0/8.0)*X[1];
        auto const a68 = a67 - 81.0/8.0;
        auto const a69 = a31*a68;
        auto const a70 = a52*a62;
        auto const a71 = (729.0/8.0)*X[1];
        auto const a72 = a71 - 243.0/8.0;
        auto const a73 = -a72;
        auto const a74 = a10*a56;
        auto const a75 = -a59;
        auto const a76 = a22*a75;
        auto const a77 = X[0]*X[1];
        auto const a78 = a10*a77;
        auto const a79 = a2*X[1];
        auto const a80 = a10*a79;
        auto const a81 = a22*a33;
        auto const a82 = a11*X[0];
        auto const a83 = a10*a82;
        auto const a84 = a4*a6;
        auto const a85 = a14*a84;
        auto const a86 = a79*a85;
        auto const a87 = a21*a84;
        auto const a88 = a11*a87;
        auto const a89 = a17*a19;
        auto const a90 = a21*a38;
        auto const a91 = a84*X[1];
        auto const a92 = -a25;
        auto const a93 = a19*X[0];
        auto const a94 = a92*a93;
        auto const a95 = a11*a84;
        auto const a96 = a77*a85;
        auto const a97 = a77*a84;
        auto const a98 = a19*a29;
        auto const a99 = a21*a91;
        auto const a100 = a33*a99;
        auto const a101 = a18*a36;
        auto const a102 = a42*X[2];
        auto const a103 = a102*a7;
        auto const a104 = a2*a5;
        auto const a105 = a104*a81;
        auto const a106 = a41*X[2];
        auto const a107 = -a68;
        auto const a108 = a104*a107;
        auto const a109 = a106*a31;
        auto const a110 = a12*X[2];
        auto const a111 = a110*a2;
        auto const a112 = a111*a31;
        auto const a113 = a22*a59;
        auto const a114 = a110*X[0];
        auto const a115 = a35*X[0];
        auto const a116 = a115*a5;
        auto const a117 = a45*a75;
        auto const a118 = a106*a39;
        auto const a119 = a58 - 243.0/8.0;
        auto const a120 = -a119;
        auto const a121 = (729.0/8.0)*X[0];
        auto const a122 = a121 - 729.0/8.0;
        auto const a123 = a11*a31;
        auto const a124 = a110*a123;
        auto const a125 = a106*a97;
        auto const a126 = (2187.0/8.0)*X[0];
        auto const a127 = a126 - 729.0/8.0;
        auto const a128 = -a127;
        auto const a129 = a110*a128;
        auto const a130 = 2187.0/8.0 - a126;
        auto const a131 = a130*a52;
        auto const a132 = a9*X[2];
        auto const a133 = a132*a63;
        auto const a134 = a50*X[2];
        auto const a135 = a132*X[1];
        auto const a136 = (2187.0/8.0)*X[1] - 729.0/8.0;
        auto const a137 = -a136;
        auto const a138 = a137*a31;
        auto const a139 = a132*a52;
        auto const a140 = (6561.0/8.0)*X[1] - 2187.0/8.0;
        auto const a141 = a56*X[2];
        auto const a142 = a127*X[2];
        auto const a143 = a142*a22;
        auto const a144 = X[0]*X[2];
        auto const a145 = a11*a144;
        auto const a146 = a102*a91;
        auto const a147 = a34*a87;
        auto const a148 = a11*a111;
        auto const a149 = a106*a34;
        auto const a150 = a84*X[0];
        auto const a151 = a150*a68;
        auto const a152 = a79*a84;
        auto const a153 = a114*a152;
        auto const a154 = a21*a75;
        auto const a155 = a110*a154;
        auto const a156 = a33*a88;
        auto const a157 = a3*a4;
        auto const a158 = a14*a157;
        auto const a159 = a158*X[2];
        auto const a160 = a159*a2;
        auto const a161 = a157*a19;
        auto const a162 = a21*a34;
        auto const a163 = a162*a5;
        auto const a164 = a21*X[2];
        auto const a165 = a164*a34;
        auto const a166 = a161*X[0];
        auto const a167 = a5*a68;
        auto const a168 = a159*X[0];
        auto const a169 = a73*X[2];
        auto const a170 = a154*X[2];
        auto const a171 = a157*a36;
        auto const a172 = a21*a33;
        auto const a173 = a172*a5;
        auto const a174 = a164*a33;
        auto const a175 = a102*X[1];
        auto const a176 = a157*a175;
        auto const a177 = a11*X[2];
        auto const a178 = a177*a59;
        auto const a179 = a159*X[1];
        auto const a180 = 729.0/8.0 - 2187.0/8.0*X[2];
        auto const a181 = a157*a180;
        auto const a182 = a111*a82;
        auto const a183 = a106*a77;
        auto const a184 = a114*a79;
        auto const a185 = a157*((6561.0/8.0)*X[2] - 2187.0/8.0);
        auto const a186 = a142*X[1];
        auto const a187 = -a55;
        auto const a188 = a157*a183;
        auto const a189 = a157*a66;
        auto const a190 = a189*a2;
        auto const a191 = a157*a170;
        auto const a192 = a136*X[2];
        auto const a193 = a157*a192;
        auto const a194 = a2*X[0];
        auto const a195 = a10*a194;
        auto const a196 = a11*a195;
        auto const a197 = a189*X[0];
        auto const a198 = a157*a80;
        auto const a199 = a144*a198;
        auto const a200 = a128*a164;
        auto const a201 = a157*a200;
        auto const a202 = a72*X[2];
        auto const a203 = a164*a59;
        auto const a204 = a157*a203;
        auto const a205 = a14*a4;
        auto const a206 = a205*X[2];
        auto const a207 = a206*a79;
        auto const a208 = a164*a44;
        auto const a209 = a11*a4;
        auto const a210 = a205*X[1];
        auto const a211 = a19*a4;
        auto const a212 = a107*a144;
        auto const a213 = a11*a212;
        auto const a214 = a206*a77;
        auto const a215 = a211*a77;
        auto const a216 = a203*X[1];
        auto const a217 = a36*a4;
        auto const a218 = a11*a165;
        auto const a219 = a3*a6;
        auto const a220 = a219*X[2];
        auto const a221 = a14*a220;
        auto const a222 = a2*a221;
        auto const a223 = a21*a219;
        auto const a224 = a223*a5;
        auto const a225 = a219*a5;
        auto const a226 = a144*a219;
        auto const a227 = a14*a92;
        auto const a228 = a174*a219;
        auto const a229 = a164*a219;
        auto const a230 = a1*a14;
        auto const a231 = a175*a219;
        auto const a232 = a11*a220;
        auto const a233 = a232*a34;
        auto const a234 = a221*X[1];
        auto const a235 = a219*X[0];
        auto const a236 = a183*a219;
        auto const a237 = a187*a219;
        auto const a238 = a220*a75;
        auto const a239 = a36*X[1];
        auto const a240 = a19*X[1];
        auto const a241 = a2*a66;
        auto const a242 = a220*a241;
        auto const a243 = a107*a226;
        auto const a244 = a226*a66;
        auto const a245 = a226*a80;
        auto const a246 = a203*a219;
        auto const a247 = a165*a219;
        auto const a248 = a6*X[2];
        auto const a249 = a248*a79;
        auto const a250 = a11*a6;
        auto const a251 = a164*a250;
        auto const a252 = a6*X[1];
        auto const a253 = a164*a24;
        auto const a254 = a144*a6;
        auto const a255 = a11*a254;
        auto const a256 = a14*a25;
        auto const a257 = a248*a77;
        auto const a258 = a19*a257;
        auto const a259 = a165*a6;
        auto const a260 = a14*a257;
        auto const a261 = a17*a251;
        auto const a262 = (9.0/8.0)*X[0] - 3.0/8.0;
        auto const a263 = -a262;
        auto const a264 = a2*a62;
        auto const a265 = a18*a264;
        auto const a266 = a24*a87;
        auto const a267 = a25*a264;
        auto const a268 = a194*a85;
        auto const a269 = (81.0/8.0)*X[0] - 27.0/8.0;
        auto const a270 = -a269;
        auto const a271 = a19*a31;
        auto const a272 = a2*a35*a8;
        auto const a273 = a31*a9;
        auto const a274 = a19*a97;
        auto const a275 = a121 - 243.0/8.0;
        auto const a276 = -a275;
        auto const a277 = a150*a154;
        auto const a278 = a87*X[0];
        auto const a279 = a278*a33;
        auto const a280 = a62*X[0];
        auto const a281 = a17*a264;
        auto const a282 = a2*a90;
        auto const a283 = a42*a97;
        auto const a284 = a194*a62;
        auto const a285 = a284*a92;
        auto const a286 = a18*a280;
        auto const a287 = a102*a33;
        auto const a288 = a102*a107;
        auto const a289 = a113*a194;
        auto const a290 = a111*a59;
        auto const a291 = a132*a2;
        auto const a292 = a123*a132;
        auto const a293 = a132*a152;
        auto const a294 = a21*a59;
        auto const a295 = a137*X[0];
        auto const a296 = (6561.0/8.0)*X[0] - 2187.0/8.0;
        auto const a297 = a21*X[0];
        auto const a298 = a142*a297;
        auto const a299 = a132*a154;
        auto const a300 = a132*X[0];
        auto const a301 = a11*a291;
        auto const a302 = a194*a68;
        auto const a303 = a11*a132;
        auto const a304 = a157*a264;
        auto const a305 = a2*a205;
        auto const a306 = a102*X[0];
        auto const a307 = a157*a306;
        auto const a308 = a157*a284;
        auto const a309 = a144*a305;
        auto const a310 = a276*X[2];
        auto const a311 = a195*a5;
        auto const a312 = a154*a157;
        auto const a313 = a211*X[0];
        auto const a314 = a157*a280;
        auto const a315 = a205*X[0];
        auto const a316 = a194*a303;
        auto const a317 = a102*a77;
        auto const a318 = a317*a4;
        auto const a319 = a142*a157;
        auto const a320 = a157*a75;
        auto const a321 = a241*a4;
        auto const a322 = a300*a79;
        auto const a323 = a192*X[0];
        auto const a324 = a4*X[0];
        auto const a325 = a200*a80;
        auto const a326 = a275*X[2];
        auto const a327 = a203*a66;
        auto const a328 = a209*a264;
        auto const a329 = a196*a203;
        auto const a330 = a219*a306;
        auto const a331 = a2*a254;
        auto const a332 = a19*a226;
        auto const a333 = a223*a33;
        auto const a334 = a6*X[0];
        auto const a335 = a14*a226;
        auto const a336 = a14*a249;
        auto const a337 = a194*a219;
        auto const a338 = a317*a6;
        auto const a339 = a11*a33;
        auto const a340 = a107*a241;
        auto const a341 = a2*a253;
        auto const a342 = a2*a24;
        auto const a343 = a157*a21;
        auto const a344 = a31*a42;
        auto const a345 = a158*a2;
        auto const a346 = a223*a34;
        auto const a347 = a39*a42;
        auto const a348 = a14*a219;
        auto const a349 = a33*a79;
        auto const a350 = a157*a82;
        auto const a351 = a219*a77;
        auto const a352 = a12*a56;
        auto const a353 = a19*a351;
        auto const a354 = a34*a77;
        auto const a355 = a56*a9;
        auto const a356 = a11*a194;
        auto const a357 = a235*a80;
        auto const a358 = a219*a297;
        auto const a359 = a358*a66;
        auto const a360 = a17*a42;
        auto const a361 = a6*a79;
        auto const a362 = a334*a79;
        auto const a363 = a12*a194;
        auto const a364 = a209*a297;
        auto const a365 = a21*a6;
        auto const a366 = a365*a77;
        auto const a367 = a18*a41;
        auto const a368 = a157*a42;
        auto const a369 = a157*a5;
        auto const a370 = a369*a41;
        auto const a371 = a297*a34;
        auto const a372 = a102*a11;
        auto const a373 = a106*a350;
        auto const a374 = a184*a219;
        auto const a375 = a157*a294;
        auto const a376 = a219*a79;
        auto const a377 = a132*a376;
        auto const a378 = a11*a300;
        auto const a379 = a34*a42;
        auto const a380 = a12*a154;
        auto const a381 = a297*a33;
        auto const a382 = a14*a3;
        auto const a383 = a2*a382;
        auto const a384 = a170*a3;
        auto const a385 = a59*a79;
        auto const a386 = a382*X[2];
        auto const a387 = a19*a77;
        auto const a388 = a75*a77;
        auto const a389 = a3*X[0];
        auto const a390 = a14*a79;
        auto const a391 = a205*a21;
        auto const a392 = a3*a5;
        auto const a393 = a297*a392;
        auto const a394 = a11*a3;
        GNp[0] = a1*a16 + a20*a23 + a22*a24;
        GNp[1] = a16*a25 + a25*a28 + a26*a27;
        GNp[2] = a19*a32 + a19*a35 + a35*a36;
        GNp[3] = a0*a28 + a13*a37 + a22*a38;
        GNp[4] = a33*a46 + a39*a43 + a44*a45;
        GNp[5] = a39*a54 + a47*a51 + a49*a53;
        GNp[6] = a12*a57 + a19*a60 + a36*a60;
        GNp[7] = a34*a36*a45 + a34*a46 + a53*a61;
        GNp[8] = a30*a64 + a35*a65 + a35*a66;
        GNp[9] = a64*a68 + a65*a69 + a68*a70;
        GNp[10] = a73*a74 + a76*a78 + a76*a80;
        GNp[11] = a29*a70 + a66*a81 + a81*a83;
        GNp[12] = a0*a86 + a88*a89 + a90*a91;
        GNp[13] = a86*a92 + a92*a96 + a94*a95;
        GNp[14] = a100*a36 + a44*a99 + a97*a98;
        GNp[15] = a1*a96 + a101*a88 + a24*a99;
        GNp[16] = a103*a29 + a105*a12 + a106*a81;
        GNp[17] = a103*a107 + a107*a109 + a108*a47;
        GNp[18] = a111*a113 + a112*a72 + a113*a114;
        GNp[19] = a106*a32 + a106*a35 + a116*a12;
        GNp[20] = a111*a117 + a118*a120 + a118*a75;
        GNp[21] = a118*a122 + a122*a124 + a125*(a71 - 729.0/8.0);
        GNp[22] = a110*a131 + a129*a52 + a129*a63;
        GNp[23] = a106*a60 + a124*a59 + a125*(a67 - 243.0/8.0);
        GNp[24] = a113*a134 + a113*a135 + a133*a72;
        GNp[25] = a133*a137 + a134*a138 + a137*a139;
        GNp[26] = a140*a141 + a143*a77 + a143*a79;
        GNp[27] = a135*a76 + a139*a73 + a145*a76;
        GNp[28] = a146*a30 + a147*a148 + a149*a99;
        GNp[29] = a125*a68 + a146*a68 + a148*a151;
        GNp[30] = a152*a155 + a153*a73 + a155*a97;
        GNp[31] = a100*a106 + a114*a156 + a125*a29;
        GNp[32] = a158*a165 + a160*a30 + a161*a163;
        GNp[33] = a160*a68 + a166*a167 + a168*a68;
        GNp[34] = a161*a170 + a166*a169 + a170*a171;
        GNp[35] = a158*a174 + a168*a29 + a171*a173;
        GNp[36] = a161*a178 + a176*a55 + a179*a59;
        GNp[37] = a176*a180 + a181*a182 + a181*a183;
        GNp[38] = a161*a186 + a171*a186 + a184*a185;
        GNp[39] = a171*a177*a75 + a179*a75 + a187*a188;
        GNp[40] = a169*a190 + a170*a189 + a191*a65;
        GNp[41] = a190*a192 + a192*a197 + a193*a196;
        GNp[42] = -a140*a199 + a198*a200 + a201*a78;
        GNp[43] = a189*a203 + a197*a202 + a204*a83;
        GNp[44] = a174*a210 + a207*a29 + a208*a209;
        GNp[45] = a107*a207 + a107*a214 + a211*a213;
        GNp[46] = a202*a215 + a211*a216 + a216*a217;
        GNp[47] = a165*a210 + a214*a30 + a217*a218;
        GNp[48] = a0*a222 + a220*a90 + a224*a89;
        GNp[49] = a222*a92 + a225*a94 + a226*a227;
        GNp[50] = a226*a98 + a228*a36 + a229*a44;
        GNp[51] = a101*a224 + a226*a230 + a229*a24;
        GNp[52] = a19*a233 + a231*a61 + a234*a34;
        GNp[53] = a148*a235*a48 + a231*a48 + a236*a48;
        GNp[54] = a184*a237 + a238*a239 + a238*a240;
        GNp[55] = a232*a33*a36 + a234*a33 + a236*a40;
        GNp[56] = a228*a65 + a228*a66 + a242*a29;
        GNp[57] = a107*a242 + a107*a244 + a243*a65;
        GNp[58] = a245*a72 + a246*a78 + a246*a80;
        GNp[59] = a244*a30 + a247*a66 + a247*a83;
        GNp[60] = a20*a251 + a230*a249 + a252*a253;
        GNp[61] = a249*a256 + a255*a26 + a256*a257;
        GNp[62] = a239*a259 + a240*a259 + a258*a30;
        GNp[63] = a0*a260 + a252*a90*X[2] + a261*a36;
        GNp[64] = a16*a263 + a2*a266 + a23*a265;
        GNp[65] = a25*a268 + a267*a27 + a31*a43;
        GNp[66] = a10*a272 + a147*a93 + a270*a271;
        GNp[67] = a150*a90 + a262*a28 + a37*a62;
        GNp[68] = a16*a33 + a264*a33*a45 + a33*a86;
        GNp[69] = a273*a51 + a31*a54 + a54*a97;
        GNp[70] = a271*a59 + a274*a59 + a31*a59*a65;
        GNp[71] = a123*a34*a62 + a28*a34 + a34*a96;
        GNp[72] = a147*a241 + a264*a35 + a270*a64;
        GNp[73] = a151*a241 + a264*a69 + a57*a9;
        GNp[74] = a195*a76 + a276*a74 + a277*a80;
        GNp[75] = a269*a70 + a279*a66 + a280*a81;
        GNp[76] = a262*a86 + a281*a88 + a282*a84;
        GNp[77] = a268*a92 + a283*a61 + a285*a95;
        GNp[78] = a156*a195 + a269*a274 + a278*a44;
        GNp[79] = a263*a96 + a266*X[0] + a286*a88;
        GNp[80] = a103*a269 + a105*a9 + a287*a87;
        GNp[81] = a108*a273 + a109*a120 + a150*a288;
        GNp[82] = a112*a275 + a278*a290 + a289*a5;
        GNp[83] = a109*a270 + a116*a9 + a149*a278;
        GNp[84] = a103*a75 + a117*a291 + a146*a75;
        GNp[85] = a109*a122 + a122*a125 + a122*a292;
        GNp[86] = a112*a128 + a128*a134*a31 + a128*a153;
        GNp[87] = a109*a59 + a125*a59 + a292*a59;
        GNp[88] = a113*a291 + a133*a275 + a293*a294;
        GNp[89] = a131*a132 + a138*a291 + a293*a295;
        GNp[90] = a141*a296 + a143*a194 + a152*a298;
        GNp[91] = a139*a276 + a299*a97 + a300*a76;
        GNp[92] = a102*a147 + a146*a270 + a147*a301;
        GNp[93] = a102*a151 + a119*a125 + a132*a302*a95;
        GNp[94] = a111*a277 + a153*a276 + a170*a194*a95;
        GNp[95] = a106*a279 + a125*a269 + a279*a303;
        GNp[96] = a160*a270 + a163*a304 + a165*a305;
        GNp[97] = a167*a308 + a307*a55 + a309*a68;
        GNp[98] = a166*a310 + a170*a313 + a311*a312;
        GNp[99] = a168*a269 + a173*a314 + a174*a315;
        GNp[100] = a160*a59 + a178*a304 + a207*a59;
        GNp[101] = a180*a318 + a181*a306 + a181*a316;
        GNp[102] = a142*a166 + a142*a215 + a196*a319;
        GNp[103] = a145*a320*a62 + a168*a75 + a214*a75;
        GNp[104] = a170*a321 + a190*a310 + a191*a264;
        GNp[105] = a185*a322 + a193*a284 + a321*a323;
        GNp[106] = a195*a201 - a199*a296 + a324*a325;
        GNp[107] = a197*a326 + a204*a280 + a324*a327;
        GNp[108] = a174*a305 + a174*a328 + a207*a269;
        GNp[109] = a107*a309 + a187*a318 + a212*a328;
        GNp[110] = a203*a313 + a215*a326 + a329*a4;
        GNp[111] = a165*a209*a280 + a165*a315 + a214*a270;
        GNp[112] = a222*a262 + a224*a281 + a248*a282;
        GNp[113] = a225*a285 + a227*a331 + a330*a61;
        GNp[114] = a208*a334 + a269*a332 + a311*a333;
        GNp[115] = a224*a286 + a253*a334 + a263*a335;
        GNp[116] = a222*a34 + a233*a264 + a336*a34;
        GNp[117] = a303*a337*a48 + a330*a48 + a338*a48;
        GNp[118] = a226*a65*a75 + a258*a75 + a332*a75;
        GNp[119] = a226*a339*a62 + a260*a33 + a33*a335;
        GNp[120] = a174*a241*a6 + a228*a264 + a242*a269;
        GNp[121] = a237*a322 + a243*a264 + a254*a340;
        GNp[122] = a195*a246 + a203*a334*a80 + a245*a275;
        GNp[123] = a244*a270 + a247*a280 + a259*a66*X[0];
        GNp[124] = a251*a265 + a263*a336 + a341*a6;
        GNp[125] = a255*a267 + a256*a331 + a257*a43;
        GNp[126] = a196*a259 + a258*a270 + a259*a93;
        GNp[127] = a254*a90 + a260*a262 + a261*a280;
        GNp[128] = a18*a23*a42 + a223*a342 + 3*a342*a343;
        GNp[129] = a25*a345*a8 + a256*a337 + a29*a344;
        GNp[130] = a12*a272 + a161*a162*a8 + a346*a93;
        GNp[131] = a157*a8*a90 + a235*a90 + a37*a41;
        GNp[132] = a269*a347 + a339*a345 + a348*a349;
        GNp[133] = a120*a53 + a350*a54 + a351*a54;
        GNp[134] = a11*a166*a59 + a275*a352 + a353*a59;
        GNp[135] = a158*a34*a82 + a270*a53 + a348*a354;
        GNp[136] = a11*a162*a304 + a241*a346 + a35*a50*a9;
        GNp[137] = a11*a308*a68 + a235*a241*a68 + a355*a72;
        GNp[138] = a154*a357 + a196*a312 + a356*a76;
        GNp[139] = a11*a172*a314 + a33*a359 + a81*a82*a9;
        GNp[140] = a209*a282 + a360*a88 + a361*a90;
        GNp[141] = a205*a356*a92 + a227*a362 + a283*a30;
        GNp[142] = a156*a363 + a364*a44 + a366*a44;
        GNp[143] = a11*a278*a367 + a24*a364 + a24*a366;
        GNp[144] = a173*a368 + a223*a287 + a42*a81;
        GNp[145] = a107*a344 + a107*a368*a5*X[0] + a235*a288;
        GNp[146] = a12*a289 + a290*a358 + a294*a363*a369;
        GNp[147] = a115*a41 + a149*a358 + a370*a371;
        GNp[148] = a231*a75 + a320*a372 + a347*a75;
        GNp[149] = a122*a236 + a122*a373 + a122*a53;
        GNp[150] = a128*a157*a182 + a128*a352 + a128*a374;
        GNp[151] = a236*a59 + a373*a59 + a53*a59;
        GNp[152] = a113*a79*a9 + a294*a377 + a301*a375;
        GNp[153] = a137*a157*a316 + a137*a355 + a295*a377;
        GNp[154] = a127*a22*a79*X[0] + a21*a319*a356 + a298*a376;
        GNp[155] = a299*a351 + a312*a378 + a76*a77*a9;
        GNp[156] = a102*a162*a209 + a175*a34*a365 + a379*a99;
        GNp[157] = a209*a306*a68 + a283*a68 + a338*a68;
        GNp[158] = a148*a154*a324 + a152*a380*X[0] + a154*a184*a6;
        GNp[159] = a106*a209*a381 + a172*a41*a97 + a183*a33*a365;
        GNp[160] = a162*a345 + a165*a383 + a343*a379*a5;
        GNp[161] = a144*a383*a68 + a158*a302 + a307*a72;
        GNp[162] = a154*a166 + a194*a369*a380 + a384*a93;
        GNp[163] = a158*a381 + a174*a382*X[0] + a370*a381;
        GNp[164] = a158*a385 + a176*a275 + a385*a386;
        GNp[165] = a130*a188 + a180*a3*a317 + a181*a42*a77;
        GNp[166] = a127*a161*a77 + a142*a3*a387 + a157*a184*a296;
        GNp[167] = a158*a388 + a188*a276 + a386*a388;
        GNp[168] = a154*a190 + a241*a384 + a301*a312;
        GNp[169] = a136*a190*X[0] + a140*a157*a322 + a241*a3*a323;
        GNp[170] = a128*a198*a297 + a201*a356 + a325*a389;
        GNp[171] = a197*a294 + a327*a389 + a375*a378;
        GNp[172] = a174*a390 + a209*a21*a287 + a349*a391;
        GNp[173] = a107*a315*a79 + a212*a390 + a318*a73;
        GNp[174] = a203*a387 + a215*a294 + a290*a364;
        GNp[175] = a14*a165*a77 + a149*a364 + a354*a391;
        GNp[176] = a219*a282 + a224*a360 + a282*a392;
        GNp[177] = a194*a227*a392 + a227*a337 + a30*a330;
        GNp[178] = a333*a363*a5 + a358*a44 + a393*a44;
        GNp[179] = a24*a358 + a24*a393 + a358*a367*a5;
        GNp[180] = a177*a34*a383 + a231*a270 + a34*a348*a79;
        GNp[181] = a119*a236 + a306*a394*a48 + a351*a42*a48;
        GNp[182] = a145*a19*a3*a75 + a276*a374 + a353*a75;
        GNp[183] = a145*a33*a382 + a236*a269 + a33*a348*a77;
        GNp[184] = a174*a264*a394 + a241*a333 + a301*a333;
        GNp[185] = a213*a264*a3 + a219*a322*a73 + a235*a340;
        GNp[186] = a246*a356 + a294*a357 + a3*a329;
        GNp[187] = a218*a280*a3 + a303*a34*a358 + a34*a359;
        GNp[188] = a11*a341 + a18*a365*a372 + a21*a24*a361;
        GNp[189] = a144*a256*a50 + a256*a362 + a29*a338;
        GNp[190] = a148*a371*a6 + a19*a354*a365 + a218*a93;
        GNp[191] = a106*a17*a250*a297 + a145*a90 + a6*a77*a90;
        return GNm;
#include "pbat/warning/Pop.h"
    }
};

} // namespace detail
} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_HEXAHEDRON_H
