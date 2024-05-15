
#ifndef PBA_CORE_FEM_TETRAHEDRON_H    
#define PBA_CORE_FEM_TETRAHEDRON_H

#include "pba/aliases.h"
#include "QuadratureRules.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Tetrahedron;

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
    
    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, kNodes> Nm;
        Nm[0] = -X[0] - X[1] - X[2] + 1;
        Nm[1] = X[0];
        Nm[2] = X[1];
        Nm[3] = X[2];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
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
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, kDims> Jacobian(
        [[maybe_unused]] Vector<kDims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == kNodes);
        auto constexpr kDimsOut = Derived::RowsAtCompileTime;
        Matrix<kDimsOut, kDims> const J = x * GradN(X);
        return J;
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
    
    template <int PolynomialOrder>
    using QuadratureType = math::SymmetricSimplexPolynomialQuadratureRule<kDims, PolynomialOrder>;
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
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
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
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
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, kDims> Jacobian(
        [[maybe_unused]] Vector<kDims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == kNodes);
        auto constexpr kDimsOut = Derived::RowsAtCompileTime;
        Matrix<kDimsOut, kDims> const J = x * GradN(X);
        return J;
    }
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_TETRAHEDRON_H
