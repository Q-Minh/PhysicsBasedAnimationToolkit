
#ifndef PBAT_FEM_QUADRILATERAL_H
#define PBAT_FEM_QUADRILATERAL_H

#include "QuadratureRules.h"

#include <array>
#include <pbat/Aliases.h>

namespace pbat {
namespace fem {

template <int Order>
struct Quadrilateral;

template <>
struct Quadrilateral<1>
{
    using AffineBaseType = Quadrilateral<1>;

    static int constexpr kOrder                                  = 1;
    static int constexpr kDims                                   = 2;
    static int constexpr kNodes                                  = 4;
    static std::array<int, kNodes * kDims> constexpr Coordinates = {
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices =
        {0, 1, 2, 3}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = X[1] - 1;
        Nm[0]         = a0 * a1;
        Nm[1]         = -a1 * X[0];
        Nm[2]         = -a0 * X[1];
        Nm[3]         = X[0] * X[1];
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp     = GNm.data();
        Scalar const a0 = X[1] - 1;
        Scalar const a1 = X[0] - 1;
        GNp[0]          = a0;
        GNp[1]          = -a0;
        GNp[2]          = -X[1];
        GNp[3]          = X[1];
        GNp[4]          = a1;
        GNp[5]          = -X[0];
        GNp[6]          = -a1;
        GNp[7]          = X[0];
        return GNm;
    }
};

template <>
struct Quadrilateral<2>
{
    using AffineBaseType = Quadrilateral<1>;

    static int constexpr kOrder                                  = 2;
    static int constexpr kDims                                   = 2;
    static int constexpr kNodes                                  = 9;
    static std::array<int, kNodes * kDims> constexpr Coordinates = {
        0,
        0,
        1,
        0,
        2,
        0,
        0,
        1,
        1,
        1,
        2,
        1,
        0,
        2,
        1,
        2,
        2,
        2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices =
        {0, 2, 6, 8}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0  = 2 * X[0] - 1;
        auto const a1  = 2 * X[1] - 1;
        auto const a2  = a0 * a1;
        auto const a3  = X[0] - 1;
        auto const a4  = X[1] - 1;
        auto const a5  = a3 * a4;
        auto const a6  = 4 * a5;
        auto const a7  = a1 * X[0];
        auto const a8  = a2 * X[0];
        auto const a9  = a0 * X[1];
        auto const a10 = a3 * X[1];
        Nm[0]          = a2 * a5;
        Nm[1]          = -a6 * a7;
        Nm[2]          = a4 * a8;
        Nm[3]          = -a6 * a9;
        Nm[4]          = 16 * a5 * X[0] * X[1];
        Nm[5]          = -4 * a4 * a9 * X[0];
        Nm[6]          = a10 * a2;
        Nm[7]          = -4 * a10 * a7;
        Nm[8]          = a8 * X[1];
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp      = GNm.data();
        Scalar const a0  = 4 * X[1] - 2;
        Scalar const a1  = X[1] - 1;
        Scalar const a2  = X[0] - 1;
        Scalar const a3  = a1 * a2;
        Scalar const a4  = (2 * X[0] - 1) * (2 * X[1] - 1);
        Scalar const a5  = a1 * a4;
        Scalar const a6  = 8 * X[1];
        Scalar const a7  = 4 - a6;
        Scalar const a8  = a1 * X[0];
        Scalar const a9  = 8 * X[0];
        Scalar const a10 = 4 - a9;
        Scalar const a11 = a1 * X[1];
        Scalar const a12 = a10 * a11;
        Scalar const a13 = 8 - a9;
        Scalar const a14 = X[0] * X[1];
        Scalar const a15 = 16 * X[0] - 16;
        Scalar const a16 = a2 * X[1];
        Scalar const a17 = a4 * X[1];
        Scalar const a18 = 4 * X[0] - 2;
        Scalar const a19 = a2 * a4;
        Scalar const a20 = a2 * a7 * X[0];
        Scalar const a21 = a4 * X[0];
        GNp[0]           = a0 * a3 + a5;
        GNp[1]           = a3 * a7 + a7 * a8;
        GNp[2]           = a0 * a8 + a5;
        GNp[3]           = a11 * a13 + a12;
        GNp[4]           = a11 * a15 + a14 * (16 * X[1] - 16);
        GNp[5]           = a12 + a14 * (8 - a6);
        GNp[6]           = a0 * a16 + a17;
        GNp[7]           = a14 * a7 + a16 * a7;
        GNp[8]           = a0 * a14 + a17;
        GNp[9]           = a18 * a3 + a19;
        GNp[10]          = a13 * a8 + a20;
        GNp[11]          = a18 * a8 + a21;
        GNp[12]          = a10 * a16 + a10 * a3;
        GNp[13]          = a14 * a15 + a15 * a8;
        GNp[14]          = a10 * a14 + a10 * a8;
        GNp[15]          = a16 * a18 + a19;
        GNp[16]          = a13 * a14 + a20;
        GNp[17]          = a14 * a18 + a21;
        return GNm;
    }
};

template <>
struct Quadrilateral<3>
{
    using AffineBaseType = Quadrilateral<1>;

    static int constexpr kOrder                                  = 3;
    static int constexpr kDims                                   = 2;
    static int constexpr kNodes                                  = 16;
    static std::array<int, kNodes * kDims> constexpr Coordinates = {
        0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1,
        0, 2, 1, 2, 2, 2, 3, 2, 0, 3, 1, 3, 2, 3, 3, 3}; ///< Divide coordinates by kOrder to obtain
                                                         ///< actual coordinates in the reference
                                                         ///< element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices =
        {0, 3, 12, 15}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0  = X[0] - 1;
        auto const a1  = X[1] - 1;
        auto const a2  = 3 * X[0];
        auto const a3  = a2 - 2;
        auto const a4  = 3 * X[1];
        auto const a5  = a4 - 2;
        auto const a6  = a0 * a1 * a3 * a5;
        auto const a7  = a2 - 1;
        auto const a8  = a4 - 1;
        auto const a9  = a7 * a8;
        auto const a10 = (1.0 / 4.0) * a9;
        auto const a11 = (9.0 / 4.0) * X[0];
        auto const a12 = a11 * a8;
        auto const a13 = a1 * a5;
        auto const a14 = a0 * a13;
        auto const a15 = a11 * a9;
        auto const a16 = a13 * a3;
        auto const a17 = a10 * X[0];
        auto const a18 = a6 * X[1];
        auto const a19 = (81.0 / 4.0) * X[0];
        auto const a20 = a7 * X[1];
        auto const a21 = a0 * X[1];
        auto const a22 = a1 * a3;
        auto const a23 = a21 * a22;
        auto const a24 = a21 * a5;
        auto const a25 = a24 * a3;
        Nm[0]          = a10 * a6;
        Nm[1]          = -a12 * a6;
        Nm[2]          = a14 * a15;
        Nm[3]          = -a16 * a17;
        Nm[4]          = -9.0 / 4.0 * a18 * a7;
        Nm[5]          = a18 * a19;
        Nm[6]          = -a14 * a19 * a20;
        Nm[7]          = a11 * a16 * a20;
        Nm[8]          = (9.0 / 4.0) * a23 * a9;
        Nm[9]          = -a19 * a23 * a8;
        Nm[10]         = a1 * a19 * a21 * a9;
        Nm[11]         = -a15 * a22 * X[1];
        Nm[12]         = -a10 * a25;
        Nm[13]         = a12 * a25;
        Nm[14]         = -a15 * a24;
        Nm[15]         = a17 * a3 * a5 * X[1];
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp      = GNm.data();
        Scalar const a0  = (9.0 / 4.0) * X[1] - 3.0 / 4.0;
        Scalar const a1  = X[0] - 1;
        Scalar const a2  = 3 * X[0];
        Scalar const a3  = a2 - 2;
        Scalar const a4  = X[1] - 1;
        Scalar const a5  = 3 * X[1];
        Scalar const a6  = a5 - 2;
        Scalar const a7  = a4 * a6;
        Scalar const a8  = a3 * a7;
        Scalar const a9  = a1 * a8;
        Scalar const a10 = a1 * a7;
        Scalar const a11 = a5 - 1;
        Scalar const a12 = (3.0 / 4.0) * X[0] - 1.0 / 4.0;
        Scalar const a13 = a11 * a12;
        Scalar const a14 = a11 * a3;
        Scalar const a15 = a12 * a14;
        Scalar const a16 = (27.0 / 4.0) * X[1];
        Scalar const a17 = a16 - 9.0 / 4.0;
        Scalar const a18 = -a17;
        Scalar const a19 = a18 * a2;
        Scalar const a20 = a8 * X[0];
        Scalar const a21 = (81.0 / 4.0) * X[1];
        Scalar const a22 = a21 - 27.0 / 4.0;
        Scalar const a23 = a10 * X[0];
        Scalar const a24 = (27.0 / 4.0) * X[0];
        Scalar const a25 = a24 - 9.0 / 4.0;
        Scalar const a26 = a11 * a25;
        Scalar const a27 = a7 * X[0];
        Scalar const a28 = -a0;
        Scalar const a29 = -a12;
        Scalar const a30 = a11 * a29;
        Scalar const a31 = a14 * a29;
        Scalar const a32 = -a25;
        Scalar const a33 = a32 * a5;
        Scalar const a34 = a8 * X[1];
        Scalar const a35 = a24 - 27.0 / 4.0;
        Scalar const a36 = -a35;
        Scalar const a37 = (81.0 / 4.0) * X[0];
        Scalar const a38 = a37 - 81.0 / 4.0;
        Scalar const a39 = a27 * a5;
        Scalar const a40 = a3 * X[1];
        Scalar const a41 = a6 * X[0];
        Scalar const a42 = a40 * a41;
        Scalar const a43 = (243.0 / 4.0) * X[0];
        Scalar const a44 = a43 - 81.0 / 4.0;
        Scalar const a45 = -a44;
        Scalar const a46 = a45 * X[1];
        Scalar const a47 = 243.0 / 4.0 - a43;
        Scalar const a48 = a4 * a40;
        Scalar const a49 = a1 * a48;
        Scalar const a50 = a1 * a4;
        Scalar const a51 = a5 * a50;
        Scalar const a52 = a4 * X[1];
        Scalar const a53 = a14 * a52;
        Scalar const a54 = 81.0 / 4.0 - 243.0 / 4.0 * X[1];
        Scalar const a55 = a54 * X[0];
        Scalar const a56 = a48 * X[0];
        Scalar const a57 = a1 * a52;
        Scalar const a58 = a57 * X[0];
        Scalar const a59 = a11 * a44;
        Scalar const a60 = a59 * X[0];
        Scalar const a61 = -a22;
        Scalar const a62 = a11 * a32;
        Scalar const a63 = a4 * X[0];
        Scalar const a64 = a5 * a63;
        Scalar const a65 = a1 * a6;
        Scalar const a66 = a40 * a65;
        Scalar const a67 = a6 * X[1];
        Scalar const a68 = a1 * a41;
        Scalar const a69 = a17 * a5;
        Scalar const a70 = a41 * X[1];
        Scalar const a71 = a1 * a70;
        Scalar const a72 = a1 * a62;
        Scalar const a73 = (9.0 / 4.0) * X[0] - 3.0 / 4.0;
        Scalar const a74 = a3 * a50;
        Scalar const a75 = a3 * a68;
        Scalar const a76 = a37 - 27.0 / 4.0;
        Scalar const a77 = -a73;
        Scalar const a78 = a3 * a64;
        Scalar const a79 = a14 * a25;
        Scalar const a80 = a1 * X[1];
        Scalar const a81 = -a76;
        Scalar const a82 = a14 * a32;
        Scalar const a83 = a5 * X[0];
        GNp[0]           = a0 * a9 + 3 * a10 * a13 + a15 * a7;
        GNp[1]           = a10 * a19 + a18 * a20 + a18 * a9;
        GNp[2]           = a10 * a26 + a22 * a23 + a26 * a27;
        GNp[3]           = a2 * a30 * a7 + a20 * a28 + a31 * a7;
        GNp[4]           = a10 * a33 + a32 * a34 + a34 * a36;
        GNp[5]           = a34 * a38 + a38 * a39 + a42 * (a21 - 81.0 / 4.0);
        GNp[6]           = a10 * a46 + a27 * a46 + a27 * a47 * X[1];
        GNp[7]           = a25 * a34 + a25 * a39 + a42 * (a16 - 27.0 / 4.0);
        GNp[8]           = a22 * a49 + a25 * a53 + a26 * a51;
        GNp[9]           = a49 * a54 + a51 * a55 + a54 * a56;
        GNp[10]          = a52 * a60 + a57 * a59 + a58 * ((729.0 / 4.0) * X[1] - 243.0 / 4.0);
        GNp[11]          = a32 * a53 + a56 * a61 + a62 * a64;
        GNp[12]          = a28 * a66 + a30 * a5 * a65 + a31 * a67;
        GNp[13]          = a17 * a42 + a17 * a66 + a68 * a69;
        GNp[14]          = a61 * a71 + a62 * a70 + a67 * a72;
        GNp[15]          = a0 * a42 + a13 * a41 * a5 + a15 * a67;
        GNp[16]          = 3 * a15 * a50 + a15 * a65 + a73 * a9;
        GNp[17]          = a18 * a75 + a19 * a74 + a20 * a36;
        GNp[18]          = a2 * a26 * a50 + a23 * a76 + a26 * a68;
        GNp[19]          = a2 * a31 * a4 + a20 * a77 + a31 * a41;
        GNp[20]          = a32 * a66 + a32 * a9 + a33 * a74;
        GNp[21]          = a20 * a38 + a38 * a42 + a38 * a78;
        GNp[22]          = a23 * a45 + a45 * a51 * X[0] + a46 * a68;
        GNp[23]          = a20 * a25 + a25 * a42 + a25 * a78;
        GNp[24]          = a49 * a76 + a50 * a79 + a79 * a80;
        GNp[25]          = a1 * a40 * a55 + a47 * a56 + a55 * a74;
        GNp[26]          = a50 * a60 + a58 * ((729.0 / 4.0) * X[0] - 243.0 / 4.0) + a60 * a80;
        GNp[27]          = a56 * a81 + a63 * a82 + a82 * X[0] * X[1];
        GNp[28]          = a1 * a31 * a5 + a31 * a65 + a66 * a77;
        GNp[29]          = a1 * a3 * a69 * X[0] + a17 * a75 + a35 * a42;
        GNp[30]          = a41 * a72 + a71 * a81 + a72 * a83;
        GNp[31]          = a15 * a41 + a15 * a83 + a42 * a73;
        return GNm;
    }
};

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_QUADRILATERAL_H
