
/**
 * @file Line.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Line finite element
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef PBAT_FEM_LINE_H
#define PBAT_FEM_LINE_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "QuadratureRules.h"

#include <array>

namespace pbat {
namespace fem {

namespace detail {

template <int Order>
struct Line;

} // namespace detail

/**
 * @brief Line finite element
 * 
 * Satisfies concept CElement  
 *
 * @tparam Order Polynomial order
 */
template <int Order>
using Line = typename detail::Line<Order>;

namespace detail {

template <>
struct Line<1>
{
    using AffineBaseType = Line<1>;

    static int constexpr kOrder = 1;
    static int constexpr kDims  = 1;
    static int constexpr kNodes = 2;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,1}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = true;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        Nm[0] = 1 - X[0];
        Nm[1] = X[0];
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        GNp[0] = -1;
        GNp[1] = 1;
        return GNm;
    }
};

template <>
struct Line<2>
{
    using AffineBaseType = Line<1>;

    static int constexpr kOrder = 2;
    static int constexpr kDims  = 1;
    static int constexpr kNodes = 3;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,1,2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,2}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = 2*X[0] - 1;
        Nm[0] = a0*a1;
        Nm[1] = -4*a0*X[0];
        Nm[2] = a1*X[0];
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        auto const a0 = 4*X[0];
        GNp[0] = a0 - 3;
        GNp[1] = 4 - 8*X[0];
        GNp[2] = a0 - 1;
        return GNm;
    }
};

template <>
struct Line<3>
{
    using AffineBaseType = Line<1>;

    static int constexpr kOrder = 3;
    static int constexpr kDims  = 1;
    static int constexpr kNodes = 4;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,1,2,3}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,3}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    static bool constexpr bHasConstantJacobian = false;

    template <int PolynomialOrder, common::CFloatingPoint TScalar = Scalar>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder, TScalar>;

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        using namespace pbat::math;
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = 3*X[0];
        auto const a2 = a1 - 2;
        auto const a3 = a0*a2;
        auto const a4 = a1 - 1;
        auto const a5 = (1.0/2.0)*a4;
        auto const a6 = (9.0/2.0)*X[0];
        Nm[0] = -a3*a5;
        Nm[1] = a3*a6;
        Nm[2] = -a0*a4*a6;
        Nm[3] = a2*a5*X[0];
        return Nm;
    }

    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Matrix<TScalar, kNodes, kDims> GNm;
        TScalar* GNp = GNm.data();
        auto const a0 = X[0] - 1;
        auto const a1 = (3.0/2.0)*X[0];
        auto const a2 = a1 - 1.0/2.0;
        auto const a3 = -a2;
        auto const a4 = 3*X[0] - 2;
        auto const a5 = (27.0/2.0)*X[0];
        auto const a6 = a5 - 27.0/2.0;
        auto const a7 = (9.0/2.0)*X[0];
        auto const a8 = 9.0/2.0 - a5;
        GNp[0] = 3*a0*a3 + a3*a4 + a4*(3.0/2.0 - a1);
        GNp[1] = a4*(a7 - 9.0/2.0) + a6*X[0] + (a5 - 9)*X[0];
        GNp[2] = a0*a8 - a6*X[0] + a8*X[0];
        GNp[3] = a2*a4 + (a7 - 3)*X[0] + (a7 - 3.0/2.0)*X[0];
        return GNm;
    }
};

} // namespace detail
} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LINE_H
