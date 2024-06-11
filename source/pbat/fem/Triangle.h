
#ifndef PBAT_FEM_TRIANGLE_H
#define PBAT_FEM_TRIANGLE_H

#include "QuadratureRules.h"

#include <array>
#include <pbat/Aliases.h>

namespace pbat {
namespace fem {

template <int Order>
struct Triangle;

template <>
struct Triangle<1>
{
    using AffineBaseType = Triangle<1>;

    static int constexpr kOrder                                  = 1;
    static int constexpr kDims                                   = 2;
    static int constexpr kNodes                                  = 3;
    static std::array<int, kNodes * kDims> constexpr Coordinates = {
        0,
        0,
        1,
        0,
        0,
        1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {
        0,
        1,
        2}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = true;

    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        Nm[0] = -X[0] - X[1] + 1;
        Nm[1] = X[0];
        Nm[2] = X[1];
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
        GNp[0]      = -1;
        GNp[1]      = 1;
        GNp[2]      = 0;
        GNp[3]      = -1;
        GNp[4]      = 0;
        GNp[5]      = 1;
        return GNm;
    }
};

template <>
struct Triangle<2>
{
    using AffineBaseType = Triangle<1>;

    static int constexpr kOrder = 2;
    static int constexpr kDims  = 2;
    static int constexpr kNodes = 6;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 0, 2}; ///< Divide coordinates by kOrder to obtain actual
                                              ///< coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {
        0,
        2,
        5}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] + X[1] - 1;
        auto const a1 = 2 * X[1];
        auto const a2 = 2 * X[0] - 1;
        auto const a3 = 4 * a0;
        Nm[0]         = a0 * (a1 + a2);
        Nm[1]         = -a3 * X[0];
        Nm[2]         = a2 * X[0];
        Nm[3]         = -a3 * X[1];
        Nm[4]         = 4 * X[0] * X[1];
        Nm[5]         = (a1 - 1) * X[1];
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp     = GNm.data();
        Scalar const a0 = 4 * X[0];
        Scalar const a1 = 4 * X[1];
        Scalar const a2 = a0 + a1 - 3;
        GNp[0]          = a2;
        GNp[1]          = -a1 - 8 * X[0] + 4;
        GNp[2]          = a0 - 1;
        GNp[3]          = -a1;
        GNp[4]          = a1;
        GNp[5]          = 0;
        GNp[6]          = a2;
        GNp[7]          = -a0;
        GNp[8]          = 0;
        GNp[9]          = -a0 - 8 * X[1] + 4;
        GNp[10]         = a0;
        GNp[11]         = a1 - 1;
        return GNm;
    }
};

template <>
struct Triangle<3>
{
    using AffineBaseType = Triangle<1>;

    static int constexpr kOrder                                  = 3;
    static int constexpr kDims                                   = 2;
    static int constexpr kNodes                                  = 10;
    static std::array<int, kNodes * kDims> constexpr Coordinates = {
        0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1,
        1, 2, 1, 0, 2, 1, 2, 0, 3}; ///< Divide coordinates by kOrder to obtain actual coordinates
                                    ///< in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {
        0,
        3,
        9}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes>
    N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0  = 3 * X[1];
        auto const a1  = 3 * X[0];
        auto const a2  = a1 - 1;
        auto const a3  = X[0] + X[1] - 1;
        auto const a4  = a1 - 2;
        auto const a5  = a3 * (a0 + a4);
        auto const a6  = (9.0 / 2.0) * X[0];
        auto const a7  = a2 * a6;
        auto const a8  = (9.0 / 2.0) * X[1];
        auto const a9  = a0 - 1;
        auto const a10 = a9 * X[1];
        Nm[0]          = -1.0 / 2.0 * a5 * (a0 + a2);
        Nm[1]          = a5 * a6;
        Nm[2]          = -a3 * a7;
        Nm[3]          = (1.0 / 2.0) * a2 * a4 * X[0];
        Nm[4]          = a5 * a8;
        Nm[5]          = -27 * a3 * X[0] * X[1];
        Nm[6]          = a7 * X[1];
        Nm[7]          = -a3 * a8 * a9;
        Nm[8]          = a10 * a6;
        Nm[9]          = (1.0 / 2.0) * a10 * (a0 - 2);
        return Nm;
    }

    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp      = GNm.data();
        Scalar const a0  = X[0] + X[1] - 1;
        Scalar const a1  = (3.0 / 2.0) * X[1];
        Scalar const a2  = (3.0 / 2.0) * X[0];
        Scalar const a3  = a2 - 1.0 / 2.0;
        Scalar const a4  = -a1 - a3;
        Scalar const a5  = 3 * X[1];
        Scalar const a6  = 3 * X[0] - 2;
        Scalar const a7  = a5 + a6;
        Scalar const a8  = 3 * a0 * a4 + a4 * a7 + a7 * (-a1 - a2 + 3.0 / 2.0);
        Scalar const a9  = (9.0 / 2.0) * X[0];
        Scalar const a10 = (9.0 / 2.0) * X[1];
        Scalar const a11 = a7 * (a10 + a9 - 9.0 / 2.0);
        Scalar const a12 = (27.0 / 2.0) * X[0];
        Scalar const a13 = (27.0 / 2.0) * X[1];
        Scalar const a14 = a12 + a13;
        Scalar const a15 = a14 - 27.0 / 2.0;
        Scalar const a16 = a14 - 9;
        Scalar const a17 = a15 * X[0] + a16 * X[0];
        Scalar const a18 = a12 - 9.0 / 2.0;
        Scalar const a19 = -a18;
        Scalar const a20 = a19 * X[0];
        Scalar const a21 = -a15;
        Scalar const a22 = a15 * X[1] + a16 * X[1];
        Scalar const a23 = 27 * X[0];
        Scalar const a24 = -a23 * X[1];
        Scalar const a25 = -a23 - 27 * X[1] + 27;
        Scalar const a26 = a12 * X[1];
        Scalar const a27 = a13 - 9.0 / 2.0;
        Scalar const a28 = -a27;
        Scalar const a29 = a28 * X[1];
        GNp[0]           = a8;
        GNp[1]           = a11 + a17;
        GNp[2]           = a0 * a19 + a20 + a21 * X[0];
        GNp[3]           = a3 * a6 + (a9 - 3) * X[0] + (a9 - 3.0 / 2.0) * X[0];
        GNp[4]           = a22;
        GNp[5]           = a24 + a25 * X[1];
        GNp[6]           = a18 * X[1] + a26;
        GNp[7]           = a29;
        GNp[8]           = a27 * X[1];
        GNp[9]           = 0;
        GNp[10]          = a8;
        GNp[11]          = a17;
        GNp[12]          = a20;
        GNp[13]          = 0;
        GNp[14]          = a11 + a22;
        GNp[15]          = a24 + a25 * X[0];
        GNp[16]          = a18 * X[0];
        GNp[17]          = a0 * a28 + a21 * X[1] + a29;
        GNp[18]          = a26 + a27 * X[0];
        GNp[19] = (a1 - 1.0 / 2.0) * (a5 - 2) + (a10 - 3) * X[1] + (a10 - 3.0 / 2.0) * X[1];
        return GNm;
    }
};

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_TRIANGLE_H
