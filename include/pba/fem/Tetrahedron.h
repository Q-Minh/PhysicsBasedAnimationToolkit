
#ifndef PBA_CORE_FEM_TETRAHEDRON_H    
#define PBA_CORE_FEM_TETRAHEDRON_H

#include "pba/aliases.h"

#include <array>

namespace pba {
namespace fem {
    
template <int Order>
struct Tetrahedron;

template <>
struct Tetrahedron<1>
{
    static int constexpr Order = 1;
    static int constexpr Dims  = 3;
    static int constexpr Nodes = 4;
    static int constexpr Vertices = 4;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,0,1,0,0,0,1,0,0,0,1}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
        
    [[maybe_unused]] static Vector<Nodes> N([[maybe_unused]] Vector<Dims> const& X)
    {
        Vector<Nodes> Nm;
        Nm[0] = -X[0] - X[1] - X[2] + 1;
        Nm[1] = X[0];
        Nm[2] = X[1];
        Nm[3] = X[2];
        return Nm;
    }
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
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
};    

template <>
struct Tetrahedron<2>
{
    static int constexpr Order = 2;
    static int constexpr Dims  = 3;
    static int constexpr Nodes = 10;
    static int constexpr Vertices = 10;
    static std::array<int, Nodes * Dims> constexpr Coordinates =
        {0,0,0,1,0,0,2,0,0,0,1,0,1,1,0,0,2,0,0,0,1,1,0,1,0,1,1,0,0,2}; ///< Divide coordinates by Order to obtain actual coordinates in the reference element
        
    [[maybe_unused]] static Vector<Nodes> N([[maybe_unused]] Vector<Dims> const& X)
    {
        Vector<Nodes> Nm;
        Scalar const a0 = X[0] + X[1] + X[2] - 1;
        Scalar const a1 = 2*X[1];
        Scalar const a2 = 2*X[2];
        Scalar const a3 = 2*X[0] - 1;
        Scalar const a4 = 4*a0;
        Scalar const a5 = 4*X[0];
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
    
    [[maybe_unused]] static Matrix<Nodes, Dims> GradN([[maybe_unused]] Vector<Dims> const& X)
    {
        Matrix<Nodes, Dims> GNm;
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
};    

} // fem
} // pba

#endif // PBA_CORE_FEM_TETRAHEDRON_H
