
#ifndef PBA_CORE_FEM_LINE_H    
#define PBA_CORE_FEM_LINE_H

#include "pba/aliases.h"
#include "QuadratureRules.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Line;

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
    
    template <int PolynomialOrder>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder>;
      
    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        Nm[0] = 1 - X[0];
        Nm[1] = X[0];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
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
    
    template <int PolynomialOrder>
    using QuadratureType = math::GaussLegendreQuadrature<kDims, PolynomialOrder>;
      
    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = 2*X[0] - 1;
        Nm[0] = a0*a1;
        Nm[1] = -4*a0*X[0];
        Nm[2] = a1*X[0];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
        Scalar const a0 = 4*X[0];
        GNp[0] = a0 - 3;
        GNp[1] = 4 - 8*X[0];
        GNp[2] = a0 - 1;
        return GNm;
    }
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_LINE_H
