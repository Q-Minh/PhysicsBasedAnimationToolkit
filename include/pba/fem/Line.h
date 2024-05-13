
#ifndef PBA_CORE_FEM_LINE_H    
#define PBA_CORE_FEM_LINE_H

#include "pba/aliases.h"

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
    static std::array<int, kNodes * kDims> constexpr kCoordinates =
        {0,1}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
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
struct Line<2>
{
    using AffineBaseType = Line<1>;
    
    static int constexpr kOrder = 2;
    static int constexpr kDims  = 1;
    static int constexpr kNodes = 3;
    static std::array<int, kNodes * kDims> constexpr kCoordinates =
        {0,1,2}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
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
struct Line<3>
{
    using AffineBaseType = Line<1>;
    
    static int constexpr kOrder = 3;
    static int constexpr kDims  = 1;
    static int constexpr kNodes = 4;
    static std::array<int, kNodes * kDims> constexpr kCoordinates =
        {0,1,2,3}; ///< Divide coordinates by kOrder to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, kNodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
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
    
    [[maybe_unused]] static Matrix<kNodes, kDims> GradN([[maybe_unused]] Vector<kDims> const& X)
    {
        Matrix<kNodes, kDims> GNm;
        Scalar* GNp = GNm.data();
        Scalar const a0 = X[0] - 1;
        Scalar const a1 = (3.0/2.0)*X[0];
        Scalar const a2 = a1 - 1.0/2.0;
        Scalar const a3 = -a2;
        Scalar const a4 = 3*X[0] - 2;
        Scalar const a5 = (27.0/2.0)*X[0];
        Scalar const a6 = a5 - 27.0/2.0;
        Scalar const a7 = (9.0/2.0)*X[0];
        Scalar const a8 = 9.0/2.0 - a5;
        GNp[0] = 3*a0*a3 + a3*a4 + a4*(3.0/2.0 - a1);
        GNp[1] = a4*(a7 - 9.0/2.0) + a6*X[0] + (a5 - 9)*X[0];
        GNp[2] = a0*a8 - a6*X[0] + a8*X[0];
        GNp[3] = a2*a4 + (a7 - 3)*X[0] + (a7 - 3.0/2.0)*X[0];
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

#endif // PBA_CORE_FEM_LINE_H
