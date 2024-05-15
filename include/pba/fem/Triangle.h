
#ifndef PBA_CORE_FEM_TRIANGLE_H    
#define PBA_CORE_FEM_TRIANGLE_H

#include "pba/aliases.h"
#include "QuadratureRules.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Triangle;

template <>
struct Triangle<1>
{
    using AffineBaseType = Triangle<1>;
    
    static int constexpr kOrder = 1;
    static int constexpr kDims  = 2;
    static int constexpr kNodes = 3;
    static std::array<int, kNodes * kDims> constexpr Coordinates =
        {0,0,1,0,0,1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,1,2}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    
    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;
      
    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
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
        GNp[0] = -1;
        GNp[1] = 1;
        GNp[2] = 0;
        GNp[3] = -1;
        GNp[4] = 0;
        GNp[5] = 1;
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
        {0,0,1,0,2,0,0,1,1,1,0,2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
    static std::array<int, AffineBaseType::kNodes> constexpr Vertices = {0,2,5}; ///< Indices into nodes [0,kNodes-1] revealing vertices of the element
    
    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;
      
    template <class TDerived, class TScalar = typename TDerived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<TDerived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        auto const a0 = X[0] + X[1] - 1;
        auto const a1 = 2*X[1];
        auto const a2 = 2*X[0] - 1;
        auto const a3 = 4*a0;
        Nm[0] = a0*(a1 + a2);
        Nm[1] = -a3*X[0];
        Nm[2] = a2*X[0];
        Nm[3] = -a3*X[1];
        Nm[4] = 4*X[0]*X[1];
        Nm[5] = (a1 - 1)*X[1];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
        Scalar const a0 = 4*X[0];
        Scalar const a1 = 4*X[1];
        Scalar const a2 = a0 + a1 - 3;
        GNp[0] = a2;
        GNp[1] = -a1 - 8*X[0] + 4;
        GNp[2] = a0 - 1;
        GNp[3] = -a1;
        GNp[4] = a1;
        GNp[5] = 0;
        GNp[6] = a2;
        GNp[7] = -a0;
        GNp[8] = 0;
        GNp[9] = -a0 - 8*X[1] + 4;
        GNp[10] = a0;
        GNp[11] = a1 - 1;
        return GNm;
    }
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_TRIANGLE_H
