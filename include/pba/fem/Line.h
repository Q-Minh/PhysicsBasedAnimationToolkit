
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
    static int constexpr Order = 1;
    static int constexpr Dims  = 1;
    static int constexpr Nodes = 2;
    static int constexpr Vertices = 2;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,1}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
        
    [[maybe_unused]] static Vector<Nodes> N([[maybe_unused]] Vector<Dims> const& X)
    {
        Vector<Nodes> Nm;
        Nm[0] = 1 - X[0];
        Nm[1] = X[0];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
        Scalar* GNp = GNm.data();
        GNp[0] = -1;
        GNp[1] = 1;
        return GNm;
    }
};    

template <>
struct Line<2>
{
    static int constexpr Order = 2;
    static int constexpr Dims  = 1;
    static int constexpr Nodes = 3;
    static int constexpr Vertices = 2;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,1,2}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
        
    [[maybe_unused]] static Vector<Nodes> N([[maybe_unused]] Vector<Dims> const& X)
    {
        Vector<Nodes> Nm;
        Scalar const a0 = X[0] - 1;
        Scalar const a1 = 2*X[0] - 1;
        Nm[0] = a0*a1;
        Nm[1] = -4*a0*X[0];
        Nm[2] = a1*X[0];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
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
