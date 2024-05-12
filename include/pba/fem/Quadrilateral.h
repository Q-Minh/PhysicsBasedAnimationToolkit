
#ifndef PBA_CORE_FEM_QUADRILATERAL_H    
#define PBA_CORE_FEM_QUADRILATERAL_H

#include "pba/aliases.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Quadrilateral;

template <>
struct Quadrilateral<1>
{
    using AffineBase = Quadrilateral<1>;
    
    static int constexpr Order = 1;
    static int constexpr Dims  = 2;
    static int constexpr Nodes = 4;
    static int constexpr Vertices = 4;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,1,0,0,1,1,1}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<TScalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
        auto const a0 = X[0] - 1;
        auto const a1 = X[1] - 1;
        Nm[0] = a0*a1;
        Nm[1] = -a1*X[0];
        Nm[2] = -a0*X[1];
        Nm[3] = X[0]*X[1];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        auto const a0 = X[1] - 1;
        auto const a1 = X[0] - 1;
        GNp[0] = a0;
        GNp[1] = -a0;
        GNp[2] = -X[1];
        GNp[3] = X[1];
        GNp[4] = a1;
        GNp[5] = -X[0];
        GNp[6] = -a1;
        GNp[7] = X[0];
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
struct Quadrilateral<2>
{
    using AffineBase = Quadrilateral<1>;
    
    static int constexpr Order = 2;
    static int constexpr Dims  = 2;
    static int constexpr Nodes = 9;
    static int constexpr Vertices = 4;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,1,0,2,0,0,1,1,1,2,1,0,2,1,2,2,2}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
      
    template <class Derived, class TScalar = typename Derived::Scalar>
    [[maybe_unused]] static Eigen::Vector<Scalar, Nodes> N([[maybe_unused]] Eigen::DenseBase<Derived> const& X)
    {
        Eigen::Vector<TScalar, Nodes> Nm;
        auto const a0 = 2*X[0] - 1;
        auto const a1 = 2*X[1] - 1;
        auto const a2 = a0*a1;
        auto const a3 = X[0] - 1;
        auto const a4 = X[1] - 1;
        auto const a5 = a3*a4;
        auto const a6 = 4*a5;
        auto const a7 = a1*X[0];
        auto const a8 = a2*X[0];
        auto const a9 = a0*X[1];
        auto const a10 = a3*X[1];
        Nm[0] = a2*a5;
        Nm[1] = -a6*a7;
        Nm[2] = a4*a8;
        Nm[3] = -a6*a9;
        Nm[4] = 16*a5*X[0]*X[1];
        Nm[5] = -4*a4*a9*X[0];
        Nm[6] = a10*a2;
        Nm[7] = -4*a10*a7;
        Nm[8] = a8*X[1];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        auto const a0 = 4*X[1] - 2;
        auto const a1 = X[1] - 1;
        auto const a2 = X[0] - 1;
        auto const a3 = a1*a2;
        auto const a4 = (2*X[0] - 1)*(2*X[1] - 1);
        auto const a5 = a1*a4;
        auto const a6 = 8*X[1];
        auto const a7 = 4 - a6;
        auto const a8 = a1*X[0];
        auto const a9 = 8*X[0];
        auto const a10 = 4 - a9;
        auto const a11 = a1*X[1];
        auto const a12 = a10*a11;
        auto const a13 = 8 - a9;
        auto const a14 = X[0]*X[1];
        auto const a15 = 16*X[0] - 16;
        auto const a16 = a2*X[1];
        auto const a17 = a4*X[1];
        auto const a18 = 4*X[0] - 2;
        auto const a19 = a2*a4;
        auto const a20 = a2*a7*X[0];
        auto const a21 = a4*X[0];
        GNp[0] = a0*a3 + a5;
        GNp[1] = a3*a7 + a7*a8;
        GNp[2] = a0*a8 + a5;
        GNp[3] = a11*a13 + a12;
        GNp[4] = a11*a15 + a14*(16*X[1] - 16);
        GNp[5] = a12 + a14*(8 - a6);
        GNp[6] = a0*a16 + a17;
        GNp[7] = a14*a7 + a16*a7;
        GNp[8] = a0*a14 + a17;
        GNp[9] = a18*a3 + a19;
        GNp[10] = a13*a8 + a20;
        GNp[11] = a18*a8 + a21;
        GNp[12] = a10*a16 + a10*a3;
        GNp[13] = a14*a15 + a15*a8;
        GNp[14] = a10*a14 + a10*a8;
        GNp[15] = a16*a18 + a19;
        GNp[16] = a13*a14 + a20;
        GNp[17] = a14*a18 + a21;
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

#endif // PBA_CORE_FEM_QUADRILATERAL_H
