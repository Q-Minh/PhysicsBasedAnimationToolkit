
#ifndef PBA_CORE_FEM_TRIANGLE_H    
#define PBA_CORE_FEM_TRIANGLE_H

#include "pba/aliases.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Triangle;

template <>
struct Triangle<1>
{
    using AffineBase = Triangle<1>;
    
    static int constexpr Order = 1;
    static int constexpr Dims  = 2;
    static int constexpr Nodes = 3;
    static int constexpr Vertices = 3;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,1,0,0,1}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
        Nm[0] = -X[0] - X[1] + 1;
        Nm[1] = X[0];
        Nm[2] = X[1];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        GNp[0] = -1;
        GNp[1] = 1;
        GNp[2] = 0;
        GNp[3] = -1;
        GNp[4] = 0;
        GNp[5] = 1;
        return GNm;
    }
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, Dims> Jacobian(
        [[maybe_unused]] Vector<Dims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == Nodes);
        auto constexpr DimsOut = Derived::RowsAtCompileTime;
        Matrix<DimsOut, Dims> const J = x * GradN(X);
        return J;
    }
};    

template <>
struct Triangle<2>
{
    using AffineBase = Triangle<1>;
    
    static int constexpr Order = 2;
    static int constexpr Dims  = 2;
    static int constexpr Nodes = 6;
    static int constexpr Vertices = 6;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,1,0,2,0,0,1,1,1,0,2}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<Scalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
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
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        auto const a0 = 4*X[0];
        auto const a1 = 4*X[1];
        auto const a2 = a0 + a1 - 3;
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
    
    template <class Derived>
    [[maybe_unused]] static Matrix<Derived::RowsAtCompileTime, Dims> Jacobian(
        [[maybe_unused]] Vector<Dims> const& X, 
        [[maybe_unused]] Eigen::DenseBase<Derived> const& x)
    {
        static_assert(Derived::RowsAtCompileTime != Eigen::Dynamic);
        assert(x.cols() == Nodes);
        auto constexpr DimsOut = Derived::RowsAtCompileTime;
        Matrix<DimsOut, Dims> const J = x * GradN(X);
        return J;
    }
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_TRIANGLE_H
