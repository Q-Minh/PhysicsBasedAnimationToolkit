#ifndef PBA_CORE_MATH_POLYNOMIAL_BASIS_H
#define PBA_CORE_MATH_POLYNOMIAL_BASIS_H

/**
* @file PolynomialBasis.h
*
* All the polynomials defined are based on expressions computed symbolically in the script
* polynomial_basis.py (or equivalently polynomial_basis.ipynb).
*
*/

#include "pba/aliases.h"

#include <cmath>
#include <numbers>

namespace pba {
namespace math {

template <int Dims, int Order>
class MonomialBasis;

template <int Dims, int Order>
class OrthonormalPolynomialBasis;

template <int Dims, int Order>
class DivergenceFreePolynomialBasis;

/**
 * Monomial basis in 1D
 */

template <>
class MonomialBasis<1, 1>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 2;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 1;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        return P;
    }
};

template <>
class MonomialBasis<1, 2>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 3;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[0]*X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 1;
        G[2] = 2*X[0];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        P[2] = (1.0/3.0)*X[0]*X[0]*X[0];
        return P;
    }
};

template <>
class MonomialBasis<1, 3>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 4;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[0]*X[0];
        P[3] = X[0]*X[0]*X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 1;
        G[2] = 2*X[0];
        G[3] = 3*X[0]*X[0];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        P[2] = (1.0/3.0)*X[0]*X[0]*X[0];
        P[3] = (1.0/4.0)*X[0]*X[0]*X[0]*X[0];
        return P;
    }
};

template <>
class MonomialBasis<1, 4>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 5;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[0]*X[0];
        P[3] = X[0]*X[0]*X[0];
        P[4] = X[0]*X[0]*X[0]*X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 1;
        G[2] = 2*X[0];
        G[3] = 3*X[0]*X[0];
        G[4] = 4*X[0]*X[0]*X[0];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        P[2] = (1.0/3.0)*X[0]*X[0]*X[0];
        P[3] = (1.0/4.0)*X[0]*X[0]*X[0]*X[0];
        P[4] = (1.0/5.0)*std::pow(X[0], 5);
        return P;
    }
};

/**
 * Monomial basis in 2D
 */

template <>
class MonomialBasis<2, 1>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 3;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[1];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 0;
        G[2] = 1;
        G[3] = 0;
        G[4] = 0;
        G[5] = 1;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[1];
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        P[2] = a0;
        P[3] = X[1];
        P[4] = a0;
        P[5] = (1.0/2.0)*X[1]*X[1];
        return P;
    }
};

template <>
class MonomialBasis<2, 2>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 6;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[0]*X[0];
        P[3] = X[1];
        P[4] = X[0]*X[1];
        P[5] = X[1]*X[1];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 0;
        G[2] = 1;
        G[3] = 0;
        G[4] = 2*X[0];
        G[5] = 0;
        G[6] = 0;
        G[7] = 1;
        G[8] = X[1];
        G[9] = X[0];
        G[10] = 0;
        G[11] = 2*X[1];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[1];
        Scalar const a3 = X[1]*X[1];
        Scalar const a4 = a3*X[0];
        P[0] = X[0];
        P[1] = a1;
        P[2] = (1.0/3.0)*X[0]*X[0]*X[0];
        P[3] = a2;
        P[4] = a1*X[1];
        P[5] = a4;
        P[6] = X[1];
        P[7] = a2;
        P[8] = a0*X[1];
        P[9] = (1.0/2.0)*a3;
        P[10] = (1.0/2.0)*a4;
        P[11] = (1.0/3.0)*X[1]*X[1]*X[1];
        return P;
    }
};

template <>
class MonomialBasis<2, 3>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 10;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = X[1]*X[1];
        P[0] = 1;
        P[1] = X[0];
        P[2] = a0;
        P[3] = X[0]*X[0]*X[0];
        P[4] = X[1];
        P[5] = X[0]*X[1];
        P[6] = a0*X[1];
        P[7] = a1;
        P[8] = a1*X[0];
        P[9] = X[1]*X[1]*X[1];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = 2*X[0];
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = a0*X[1];
        Scalar const a3 = X[1]*X[1];
        G[0] = 0;
        G[1] = 0;
        G[2] = 1;
        G[3] = 0;
        G[4] = a0;
        G[5] = 0;
        G[6] = 3*a1;
        G[7] = 0;
        G[8] = 0;
        G[9] = 1;
        G[10] = X[1];
        G[11] = X[0];
        G[12] = a2;
        G[13] = a1;
        G[14] = 0;
        G[15] = 2*X[1];
        G[16] = a3;
        G[17] = a2;
        G[18] = 0;
        G[19] = 3*a3;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[0]*X[0];
        Scalar const a3 = (1.0/3.0)*a2;
        Scalar const a4 = X[0]*X[1];
        Scalar const a5 = X[1]*X[1];
        Scalar const a6 = a5*X[0];
        Scalar const a7 = a1*a5;
        Scalar const a8 = X[1]*X[1]*X[1];
        Scalar const a9 = a8*X[0];
        P[0] = X[0];
        P[1] = a1;
        P[2] = a3;
        P[3] = (1.0/4.0)*X[0]*X[0]*X[0]*X[0];
        P[4] = a4;
        P[5] = a1*X[1];
        P[6] = a3*X[1];
        P[7] = a6;
        P[8] = a7;
        P[9] = a9;
        P[10] = X[1];
        P[11] = a4;
        P[12] = a0*X[1];
        P[13] = a2*X[1];
        P[14] = (1.0/2.0)*a5;
        P[15] = (1.0/2.0)*a6;
        P[16] = a7;
        P[17] = (1.0/3.0)*a8;
        P[18] = (1.0/3.0)*a9;
        P[19] = (1.0/4.0)*X[1]*X[1]*X[1]*X[1];
        return P;
    }
};

template <>
class MonomialBasis<2, 4>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 15;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = X[0]*X[0]*X[0];
        Scalar const a2 = X[1]*X[1];
        Scalar const a3 = X[1]*X[1]*X[1];
        P[0] = 1;
        P[1] = X[0];
        P[2] = a0;
        P[3] = a1;
        P[4] = X[0]*X[0]*X[0]*X[0];
        P[5] = X[1];
        P[6] = X[0]*X[1];
        P[7] = a0*X[1];
        P[8] = a1*X[1];
        P[9] = a2;
        P[10] = a2*X[0];
        P[11] = a0*a2;
        P[12] = a3;
        P[13] = a3*X[0];
        P[14] = X[1]*X[1]*X[1]*X[1];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = 2*X[0];
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = 3*a1;
        Scalar const a3 = X[0]*X[0]*X[0];
        Scalar const a4 = a0*X[1];
        Scalar const a5 = 2*X[1];
        Scalar const a6 = X[1]*X[1];
        Scalar const a7 = 3*a6;
        Scalar const a8 = X[1]*X[1]*X[1];
        G[0] = 0;
        G[1] = 0;
        G[2] = 1;
        G[3] = 0;
        G[4] = a0;
        G[5] = 0;
        G[6] = a2;
        G[7] = 0;
        G[8] = 4*a3;
        G[9] = 0;
        G[10] = 0;
        G[11] = 1;
        G[12] = X[1];
        G[13] = X[0];
        G[14] = a4;
        G[15] = a1;
        G[16] = a2*X[1];
        G[17] = a3;
        G[18] = 0;
        G[19] = a5;
        G[20] = a6;
        G[21] = a4;
        G[22] = a0*a6;
        G[23] = a1*a5;
        G[24] = 0;
        G[25] = a7;
        G[26] = a8;
        G[27] = a7*X[0];
        G[28] = 0;
        G[29] = 4*a8;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[0]*X[0];
        Scalar const a3 = (1.0/3.0)*a2;
        Scalar const a4 = X[0]*X[0]*X[0]*X[0];
        Scalar const a5 = (1.0/4.0)*a4;
        Scalar const a6 = X[0]*X[1];
        Scalar const a7 = X[1]*X[1];
        Scalar const a8 = a7*X[0];
        Scalar const a9 = a1*a7;
        Scalar const a10 = X[1]*X[1]*X[1];
        Scalar const a11 = a10*X[0];
        Scalar const a12 = X[1]*X[1]*X[1]*X[1];
        Scalar const a13 = a12*X[0];
        Scalar const a14 = (1.0/2.0)*a7;
        Scalar const a15 = (1.0/3.0)*a10;
        P[0] = X[0];
        P[1] = a1;
        P[2] = a3;
        P[3] = a5;
        P[4] = (1.0/5.0)*std::pow(X[0], 5);
        P[5] = a6;
        P[6] = a1*X[1];
        P[7] = a3*X[1];
        P[8] = a5*X[1];
        P[9] = a8;
        P[10] = a9;
        P[11] = a3*a7;
        P[12] = a11;
        P[13] = a1*a10;
        P[14] = a13;
        P[15] = X[1];
        P[16] = a6;
        P[17] = a0*X[1];
        P[18] = a2*X[1];
        P[19] = a4*X[1];
        P[20] = a14;
        P[21] = (1.0/2.0)*a8;
        P[22] = a9;
        P[23] = a14*a2;
        P[24] = a15;
        P[25] = (1.0/3.0)*a11;
        P[26] = a0*a15;
        P[27] = (1.0/4.0)*a12;
        P[28] = (1.0/4.0)*a13;
        P[29] = (1.0/5.0)*std::pow(X[1], 5);
        return P;
    }
};

/**
 * Monomial basis in 3D
 */

template <>
class MonomialBasis<3, 1>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 4;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[1];
        P[3] = X[2];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 1;
        G[4] = 0;
        G[5] = 0;
        G[6] = 0;
        G[7] = 1;
        G[8] = 0;
        G[9] = 0;
        G[10] = 0;
        G[11] = 1;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[1];
        Scalar const a1 = X[0]*X[2];
        Scalar const a2 = X[1]*X[2];
        P[0] = X[0];
        P[1] = (1.0/2.0)*X[0]*X[0];
        P[2] = a0;
        P[3] = a1;
        P[4] = X[1];
        P[5] = a0;
        P[6] = (1.0/2.0)*X[1]*X[1];
        P[7] = a2;
        P[8] = X[2];
        P[9] = a1;
        P[10] = a2;
        P[11] = (1.0/2.0)*X[2]*X[2];
        return P;
    }
};

template <>
class MonomialBasis<3, 2>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 10;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[0]*X[0];
        P[3] = X[1];
        P[4] = X[0]*X[1];
        P[5] = X[1]*X[1];
        P[6] = X[2];
        P[7] = X[0]*X[2];
        P[8] = X[1]*X[2];
        P[9] = X[2]*X[2];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 1;
        G[4] = 0;
        G[5] = 0;
        G[6] = 2*X[0];
        G[7] = 0;
        G[8] = 0;
        G[9] = 0;
        G[10] = 1;
        G[11] = 0;
        G[12] = X[1];
        G[13] = X[0];
        G[14] = 0;
        G[15] = 0;
        G[16] = 2*X[1];
        G[17] = 0;
        G[18] = 0;
        G[19] = 0;
        G[20] = 1;
        G[21] = X[2];
        G[22] = 0;
        G[23] = X[0];
        G[24] = 0;
        G[25] = X[2];
        G[26] = X[1];
        G[27] = 0;
        G[28] = 0;
        G[29] = 2*X[2];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[1];
        Scalar const a3 = X[1]*X[1];
        Scalar const a4 = a3*X[0];
        Scalar const a5 = X[0]*X[2];
        Scalar const a6 = a2*X[2];
        Scalar const a7 = X[2]*X[2];
        Scalar const a8 = a7*X[0];
        Scalar const a9 = (1.0/2.0)*a3;
        Scalar const a10 = X[1]*X[2];
        Scalar const a11 = a7*X[1];
        P[0] = X[0];
        P[1] = a1;
        P[2] = (1.0/3.0)*X[0]*X[0]*X[0];
        P[3] = a2;
        P[4] = a1*X[1];
        P[5] = a4;
        P[6] = a5;
        P[7] = a1*X[2];
        P[8] = a6;
        P[9] = a8;
        P[10] = X[1];
        P[11] = a2;
        P[12] = a0*X[1];
        P[13] = a9;
        P[14] = (1.0/2.0)*a4;
        P[15] = (1.0/3.0)*X[1]*X[1]*X[1];
        P[16] = a10;
        P[17] = a6;
        P[18] = a9*X[2];
        P[19] = a11;
        P[20] = X[2];
        P[21] = a5;
        P[22] = a0*X[2];
        P[23] = a10;
        P[24] = a6;
        P[25] = a3*X[2];
        P[26] = (1.0/2.0)*a7;
        P[27] = (1.0/2.0)*a8;
        P[28] = (1.0/2.0)*a11;
        P[29] = (1.0/3.0)*X[2]*X[2]*X[2];
        return P;
    }
};

template <>
class MonomialBasis<3, 3>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 20;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = X[0]*X[1];
        Scalar const a2 = X[1]*X[1];
        Scalar const a3 = X[2]*X[2];
        P[0] = 1;
        P[1] = X[0];
        P[2] = a0;
        P[3] = X[0]*X[0]*X[0];
        P[4] = X[1];
        P[5] = a1;
        P[6] = a0*X[1];
        P[7] = a2;
        P[8] = a2*X[0];
        P[9] = X[1]*X[1]*X[1];
        P[10] = X[2];
        P[11] = X[0]*X[2];
        P[12] = a0*X[2];
        P[13] = X[1]*X[2];
        P[14] = a1*X[2];
        P[15] = a2*X[2];
        P[16] = a3;
        P[17] = a3*X[0];
        P[18] = a3*X[1];
        P[19] = X[2]*X[2]*X[2];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = 2*X[0];
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = a0*X[1];
        Scalar const a3 = 2*X[1];
        Scalar const a4 = X[1]*X[1];
        Scalar const a5 = a0*X[2];
        Scalar const a6 = a3*X[2];
        Scalar const a7 = X[2]*X[2];
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 1;
        G[4] = 0;
        G[5] = 0;
        G[6] = a0;
        G[7] = 0;
        G[8] = 0;
        G[9] = 3*a1;
        G[10] = 0;
        G[11] = 0;
        G[12] = 0;
        G[13] = 1;
        G[14] = 0;
        G[15] = X[1];
        G[16] = X[0];
        G[17] = 0;
        G[18] = a2;
        G[19] = a1;
        G[20] = 0;
        G[21] = 0;
        G[22] = a3;
        G[23] = 0;
        G[24] = a4;
        G[25] = a2;
        G[26] = 0;
        G[27] = 0;
        G[28] = 3*a4;
        G[29] = 0;
        G[30] = 0;
        G[31] = 0;
        G[32] = 1;
        G[33] = X[2];
        G[34] = 0;
        G[35] = X[0];
        G[36] = a5;
        G[37] = 0;
        G[38] = a1;
        G[39] = 0;
        G[40] = X[2];
        G[41] = X[1];
        G[42] = X[1]*X[2];
        G[43] = X[0]*X[2];
        G[44] = X[0]*X[1];
        G[45] = 0;
        G[46] = a6;
        G[47] = a4;
        G[48] = 0;
        G[49] = 0;
        G[50] = 2*X[2];
        G[51] = a7;
        G[52] = 0;
        G[53] = a5;
        G[54] = 0;
        G[55] = a7;
        G[56] = a6;
        G[57] = 0;
        G[58] = 0;
        G[59] = 3*a7;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[0]*X[0];
        Scalar const a3 = (1.0/3.0)*a2;
        Scalar const a4 = X[0]*X[1];
        Scalar const a5 = X[1]*X[1];
        Scalar const a6 = a5*X[0];
        Scalar const a7 = a1*a5;
        Scalar const a8 = X[1]*X[1]*X[1];
        Scalar const a9 = a8*X[0];
        Scalar const a10 = X[0]*X[2];
        Scalar const a11 = a4*X[2];
        Scalar const a12 = X[1]*X[2];
        Scalar const a13 = a6*X[2];
        Scalar const a14 = X[2]*X[2];
        Scalar const a15 = a14*X[0];
        Scalar const a16 = a1*a14;
        Scalar const a17 = a14*a4;
        Scalar const a18 = X[2]*X[2]*X[2];
        Scalar const a19 = a18*X[0];
        Scalar const a20 = a0*X[1];
        Scalar const a21 = (1.0/2.0)*a5;
        Scalar const a22 = (1.0/3.0)*a8;
        Scalar const a23 = a20*X[2];
        Scalar const a24 = a14*X[1];
        Scalar const a25 = a14*a21;
        Scalar const a26 = a18*X[1];
        Scalar const a27 = (1.0/2.0)*a14;
        P[0] = X[0];
        P[1] = a1;
        P[2] = a3;
        P[3] = (1.0/4.0)*X[0]*X[0]*X[0]*X[0];
        P[4] = a4;
        P[5] = a1*X[1];
        P[6] = a3*X[1];
        P[7] = a6;
        P[8] = a7;
        P[9] = a9;
        P[10] = a10;
        P[11] = a1*X[2];
        P[12] = a3*X[2];
        P[13] = a11;
        P[14] = a1*a12;
        P[15] = a13;
        P[16] = a15;
        P[17] = a16;
        P[18] = a17;
        P[19] = a19;
        P[20] = X[1];
        P[21] = a4;
        P[22] = a20;
        P[23] = a2*X[1];
        P[24] = a21;
        P[25] = (1.0/2.0)*a6;
        P[26] = a7;
        P[27] = a22;
        P[28] = (1.0/3.0)*a9;
        P[29] = (1.0/4.0)*X[1]*X[1]*X[1]*X[1];
        P[30] = a12;
        P[31] = a11;
        P[32] = a23;
        P[33] = a21*X[2];
        P[34] = (1.0/2.0)*a13;
        P[35] = a22*X[2];
        P[36] = a24;
        P[37] = a17;
        P[38] = a25;
        P[39] = a26;
        P[40] = X[2];
        P[41] = a10;
        P[42] = a0*X[2];
        P[43] = a2*X[2];
        P[44] = a12;
        P[45] = a11;
        P[46] = a23;
        P[47] = a5*X[2];
        P[48] = a13;
        P[49] = a8*X[2];
        P[50] = a27;
        P[51] = (1.0/2.0)*a15;
        P[52] = a16;
        P[53] = (1.0/2.0)*a24;
        P[54] = a27*a4;
        P[55] = a25;
        P[56] = (1.0/3.0)*a18;
        P[57] = (1.0/3.0)*a19;
        P[58] = (1.0/3.0)*a26;
        P[59] = (1.0/4.0)*X[2]*X[2]*X[2]*X[2];
        return P;
    }
};

template <>
class MonomialBasis<3, 4>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 35;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = X[0]*X[0]*X[0];
        Scalar const a2 = X[0]*X[1];
        Scalar const a3 = a0*X[1];
        Scalar const a4 = X[1]*X[1];
        Scalar const a5 = a4*X[0];
        Scalar const a6 = X[1]*X[1]*X[1];
        Scalar const a7 = X[2]*X[2];
        Scalar const a8 = X[2]*X[2]*X[2];
        P[0] = 1;
        P[1] = X[0];
        P[2] = a0;
        P[3] = a1;
        P[4] = X[0]*X[0]*X[0]*X[0];
        P[5] = X[1];
        P[6] = a2;
        P[7] = a3;
        P[8] = a1*X[1];
        P[9] = a4;
        P[10] = a5;
        P[11] = a0*a4;
        P[12] = a6;
        P[13] = a6*X[0];
        P[14] = X[1]*X[1]*X[1]*X[1];
        P[15] = X[2];
        P[16] = X[0]*X[2];
        P[17] = a0*X[2];
        P[18] = a1*X[2];
        P[19] = X[1]*X[2];
        P[20] = a2*X[2];
        P[21] = a3*X[2];
        P[22] = a4*X[2];
        P[23] = a5*X[2];
        P[24] = a6*X[2];
        P[25] = a7;
        P[26] = a7*X[0];
        P[27] = a0*a7;
        P[28] = a7*X[1];
        P[29] = a2*a7;
        P[30] = a4*a7;
        P[31] = a8;
        P[32] = a8*X[0];
        P[33] = a8*X[1];
        P[34] = X[2]*X[2]*X[2]*X[2];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = 2*X[0];
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = 3*a1;
        Scalar const a3 = X[0]*X[0]*X[0];
        Scalar const a4 = a0*X[1];
        Scalar const a5 = 2*X[1];
        Scalar const a6 = X[1]*X[1];
        Scalar const a7 = 3*a6;
        Scalar const a8 = X[1]*X[1]*X[1];
        Scalar const a9 = a0*X[2];
        Scalar const a10 = X[1]*X[2];
        Scalar const a11 = a0*a10;
        Scalar const a12 = a1*X[2];
        Scalar const a13 = a5*X[2];
        Scalar const a14 = a6*X[2];
        Scalar const a15 = X[2]*X[2];
        Scalar const a16 = a15*X[1];
        Scalar const a17 = a15*X[0];
        Scalar const a18 = X[2]*X[2]*X[2];
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 1;
        G[4] = 0;
        G[5] = 0;
        G[6] = a0;
        G[7] = 0;
        G[8] = 0;
        G[9] = a2;
        G[10] = 0;
        G[11] = 0;
        G[12] = 4*a3;
        G[13] = 0;
        G[14] = 0;
        G[15] = 0;
        G[16] = 1;
        G[17] = 0;
        G[18] = X[1];
        G[19] = X[0];
        G[20] = 0;
        G[21] = a4;
        G[22] = a1;
        G[23] = 0;
        G[24] = a2*X[1];
        G[25] = a3;
        G[26] = 0;
        G[27] = 0;
        G[28] = a5;
        G[29] = 0;
        G[30] = a6;
        G[31] = a4;
        G[32] = 0;
        G[33] = a0*a6;
        G[34] = a1*a5;
        G[35] = 0;
        G[36] = 0;
        G[37] = a7;
        G[38] = 0;
        G[39] = a8;
        G[40] = a7*X[0];
        G[41] = 0;
        G[42] = 0;
        G[43] = 4*a8;
        G[44] = 0;
        G[45] = 0;
        G[46] = 0;
        G[47] = 1;
        G[48] = X[2];
        G[49] = 0;
        G[50] = X[0];
        G[51] = a9;
        G[52] = 0;
        G[53] = a1;
        G[54] = a2*X[2];
        G[55] = 0;
        G[56] = a3;
        G[57] = 0;
        G[58] = X[2];
        G[59] = X[1];
        G[60] = a10;
        G[61] = X[0]*X[2];
        G[62] = X[0]*X[1];
        G[63] = a11;
        G[64] = a12;
        G[65] = a1*X[1];
        G[66] = 0;
        G[67] = a13;
        G[68] = a6;
        G[69] = a14;
        G[70] = a11;
        G[71] = a6*X[0];
        G[72] = 0;
        G[73] = a7*X[2];
        G[74] = a8;
        G[75] = 0;
        G[76] = 0;
        G[77] = 2*X[2];
        G[78] = a15;
        G[79] = 0;
        G[80] = a9;
        G[81] = a0*a15;
        G[82] = 0;
        G[83] = 2*a12;
        G[84] = 0;
        G[85] = a15;
        G[86] = a13;
        G[87] = a16;
        G[88] = a17;
        G[89] = a11;
        G[90] = 0;
        G[91] = a15*a5;
        G[92] = 2*a14;
        G[93] = 0;
        G[94] = 0;
        G[95] = 3*a15;
        G[96] = a18;
        G[97] = 0;
        G[98] = 3*a17;
        G[99] = 0;
        G[100] = a18;
        G[101] = 3*a16;
        G[102] = 0;
        G[103] = 0;
        G[104] = 4*a18;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = X[0]*X[0];
        Scalar const a1 = (1.0/2.0)*a0;
        Scalar const a2 = X[0]*X[0]*X[0];
        Scalar const a3 = (1.0/3.0)*a2;
        Scalar const a4 = X[0]*X[0]*X[0]*X[0];
        Scalar const a5 = (1.0/4.0)*a4;
        Scalar const a6 = X[0]*X[1];
        Scalar const a7 = X[1]*X[1];
        Scalar const a8 = a7*X[0];
        Scalar const a9 = a1*a7;
        Scalar const a10 = X[1]*X[1]*X[1];
        Scalar const a11 = a10*X[0];
        Scalar const a12 = X[1]*X[1]*X[1]*X[1];
        Scalar const a13 = a12*X[0];
        Scalar const a14 = X[0]*X[2];
        Scalar const a15 = a6*X[2];
        Scalar const a16 = X[1]*X[2];
        Scalar const a17 = a8*X[2];
        Scalar const a18 = a7*X[2];
        Scalar const a19 = a1*a18;
        Scalar const a20 = a11*X[2];
        Scalar const a21 = X[2]*X[2];
        Scalar const a22 = a21*X[0];
        Scalar const a23 = a1*a21;
        Scalar const a24 = a21*a6;
        Scalar const a25 = a21*X[1];
        Scalar const a26 = a1*a25;
        Scalar const a27 = X[2]*X[2]*X[2];
        Scalar const a28 = a27*X[0];
        Scalar const a29 = a27*a6;
        Scalar const a30 = X[2]*X[2]*X[2]*X[2];
        Scalar const a31 = a30*X[0];
        Scalar const a32 = a0*X[1];
        Scalar const a33 = a2*X[1];
        Scalar const a34 = (1.0/2.0)*a7;
        Scalar const a35 = (1.0/3.0)*a10;
        Scalar const a36 = (1.0/4.0)*a12;
        Scalar const a37 = a32*X[2];
        Scalar const a38 = a33*X[2];
        Scalar const a39 = a21*a34;
        Scalar const a40 = (1.0/2.0)*a21;
        Scalar const a41 = a40*a8;
        Scalar const a42 = a27*X[1];
        Scalar const a43 = a30*X[1];
        Scalar const a44 = a0*X[2];
        Scalar const a45 = (1.0/3.0)*a27;
        P[0] = X[0];
        P[1] = a1;
        P[2] = a3;
        P[3] = a5;
        P[4] = (1.0/5.0)*std::pow(X[0], 5);
        P[5] = a6;
        P[6] = a1*X[1];
        P[7] = a3*X[1];
        P[8] = a5*X[1];
        P[9] = a8;
        P[10] = a9;
        P[11] = a3*a7;
        P[12] = a11;
        P[13] = a1*a10;
        P[14] = a13;
        P[15] = a14;
        P[16] = a1*X[2];
        P[17] = a3*X[2];
        P[18] = a5*X[2];
        P[19] = a15;
        P[20] = a1*a16;
        P[21] = a16*a3;
        P[22] = a17;
        P[23] = a19;
        P[24] = a20;
        P[25] = a22;
        P[26] = a23;
        P[27] = a21*a3;
        P[28] = a24;
        P[29] = a26;
        P[30] = a21*a8;
        P[31] = a28;
        P[32] = a1*a27;
        P[33] = a29;
        P[34] = a31;
        P[35] = X[1];
        P[36] = a6;
        P[37] = a32;
        P[38] = a33;
        P[39] = a4*X[1];
        P[40] = a34;
        P[41] = (1.0/2.0)*a8;
        P[42] = a9;
        P[43] = a2*a34;
        P[44] = a35;
        P[45] = (1.0/3.0)*a11;
        P[46] = a0*a35;
        P[47] = a36;
        P[48] = (1.0/4.0)*a13;
        P[49] = (1.0/5.0)*std::pow(X[1], 5);
        P[50] = a16;
        P[51] = a15;
        P[52] = a37;
        P[53] = a38;
        P[54] = a34*X[2];
        P[55] = (1.0/2.0)*a17;
        P[56] = a19;
        P[57] = a35*X[2];
        P[58] = (1.0/3.0)*a20;
        P[59] = a36*X[2];
        P[60] = a25;
        P[61] = a24;
        P[62] = a21*a32;
        P[63] = a39;
        P[64] = a41;
        P[65] = a21*a35;
        P[66] = a42;
        P[67] = a29;
        P[68] = a27*a34;
        P[69] = a43;
        P[70] = X[2];
        P[71] = a14;
        P[72] = a44;
        P[73] = a2*X[2];
        P[74] = a4*X[2];
        P[75] = a16;
        P[76] = a15;
        P[77] = a37;
        P[78] = a38;
        P[79] = a18;
        P[80] = a17;
        P[81] = a44*a7;
        P[82] = a10*X[2];
        P[83] = a20;
        P[84] = a12*X[2];
        P[85] = a40;
        P[86] = (1.0/2.0)*a22;
        P[87] = a23;
        P[88] = a2*a40;
        P[89] = (1.0/2.0)*a25;
        P[90] = a40*a6;
        P[91] = a26;
        P[92] = a39;
        P[93] = a41;
        P[94] = a10*a40;
        P[95] = a45;
        P[96] = (1.0/3.0)*a28;
        P[97] = a0*a45;
        P[98] = (1.0/3.0)*a42;
        P[99] = a45*a6;
        P[100] = a45*a7;
        P[101] = (1.0/4.0)*a30;
        P[102] = (1.0/4.0)*a31;
        P[103] = (1.0/4.0)*a43;
        P[104] = (1.0/5.0)*std::pow(X[2], 5);
        return P;
    }
};

/**
 * Orthonormalized polynomial basis on reference line
 */

template <>
class OrthonormalPolynomialBasis<1, 1>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 2;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        P[0] = 1;
        P[1] = 2*a0*X[0] - a0;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        G[0] = 0;
        G[1] = 2*std::numbers::sqrt3_v<Scalar>;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        P[0] = X[0];
        P[1] = a0*X[0]*X[0] - a0*X[0];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<1, 2>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 3;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = 6*a1;
        P[0] = 1;
        P[1] = 2*a0*X[0] - a0;
        P[2] = a1 + a2*X[0]*X[0] - a2*X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(5);
        G[0] = 0;
        G[1] = 2*std::numbers::sqrt3_v<Scalar>;
        G[2] = 12*a0*X[0] - 6*a0;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = std::sqrt(5);
        P[0] = X[0];
        P[1] = a0*a1 - a0*X[0];
        P[2] = -3*a1*a2 + 2*a2*X[0]*X[0]*X[0] + a2*X[0];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<1, 3>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 4;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = 6*a1;
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::sqrt(7);
        P[0] = 1;
        P[1] = 2*a0*X[0] - a0;
        P[2] = a1 + a2*a3 - a2*X[0];
        P[3] = -30*a3*a4 + 20*a4*X[0]*X[0]*X[0] + 12*a4*X[0] - a4;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(5);
        Scalar const a1 = std::sqrt(7);
        Scalar const a2 = 60*a1;
        G[0] = 0;
        G[1] = 2*std::numbers::sqrt3_v<Scalar>;
        G[2] = 12*a0*X[0] - 6*a0;
        G[3] = 12*a1 + a2*X[0]*X[0] - a2*X[0];
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = std::sqrt(5);
        Scalar const a3 = X[0]*X[0]*X[0];
        Scalar const a4 = std::sqrt(7);
        P[0] = X[0];
        P[1] = a0*a1 - a0*X[0];
        P[2] = -3*a1*a2 + 2*a2*a3 + a2*X[0];
        P[3] = 6*a1*a4 - 10*a3*a4 + 5*a4*X[0]*X[0]*X[0]*X[0] - a4*X[0];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<1, 4>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 5;

    Vector<Size> eval(Vector<1> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = 6*a1;
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::sqrt(7);
        Scalar const a5 = X[0]*X[0]*X[0];
        P[0] = 1;
        P[1] = 2*a0*X[0] - a0;
        P[2] = a1 + a2*a3 - a2*X[0];
        P[3] = -30*a3*a4 + 20*a4*a5 + 12*a4*X[0] - a4;
        P[4] = 270*a3 - 420*a5 + 210*X[0]*X[0]*X[0]*X[0] - 60*X[0] + 3;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<1> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(5);
        Scalar const a1 = std::sqrt(7);
        Scalar const a2 = 60*a1;
        Scalar const a3 = X[0]*X[0];
        G[0] = 0;
        G[1] = 2*std::numbers::sqrt3_v<Scalar>;
        G[2] = 12*a0*X[0] - 6*a0;
        G[3] = 12*a1 + a2*a3 - a2*X[0];
        G[4] = -1260*a3 + 840*X[0]*X[0]*X[0] + 540*X[0] - 60;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<1> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = X[0]*X[0];
        Scalar const a2 = std::sqrt(5);
        Scalar const a3 = X[0]*X[0]*X[0];
        Scalar const a4 = std::sqrt(7);
        Scalar const a5 = X[0]*X[0]*X[0]*X[0];
        P[0] = X[0];
        P[1] = a0*a1 - a0*X[0];
        P[2] = -3*a1*a2 + 2*a2*a3 + a2*X[0];
        P[3] = 6*a1*a4 - 10*a3*a4 + 5*a4*a5 - a4*X[0];
        P[4] = -30*a1 + 90*a3 - 105*a5 + 42*std::pow(X[0], 5) + 3*X[0];
        return P;
    }
};

/**
 * Orthonormalized polynomial basis on reference triangle
 */

template <>
class OrthonormalPolynomialBasis<2, 1>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 3;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = 2*a0;
        P[0] = std::numbers::sqrt2_v<Scalar>;
        P[1] = 6*X[0] - 2;
        P[2] = 4*a0*X[1] + a1*X[0] - a1;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        G[0] = 0;
        G[1] = 0;
        G[2] = 6;
        G[3] = 0;
        G[4] = 2*a0;
        G[5] = 4*a0;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 2*X[0];
        Scalar const a2 = X[0]*X[0];
        Scalar const a3 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a4 = 2*a3;
        Scalar const a5 = -a4;
        P[0] = a0*X[0];
        P[1] = -a1 + 3*a2;
        P[2] = a2*a3 + (4*a3*X[1] + a5)*X[0];
        P[3] = a0*X[1];
        P[4] = (6*X[0] - 2)*X[1];
        P[5] = a4*X[1]*X[1] + (a1*a3 + a5)*X[1];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<2, 2>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 6;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = 2*a2;
        Scalar const a4 = std::sqrt(6);
        Scalar const a5 = X[0]*X[0];
        Scalar const a6 = a0*X[0];
        Scalar const a7 = 6*X[1];
        Scalar const a8 = std::sqrt(30);
        P[0] = a0;
        P[1] = a1 - 2;
        P[2] = 4*a2*X[1] + a3*X[0] - a3;
        P[3] = 10*a4*a5 - 8*a4*X[0] + a4;
        P[4] = 15*a0*a5 - a0*a7 + 3*a0 + 30*a6*X[1] - 18*a6;
        P[5] = a1*a8*X[1] + a5*a8 - a7*a8 - 2*a8*X[0] + 6*a8*X[1]*X[1] + a8;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(6);
        Scalar const a2 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a3 = 30*a2;
        Scalar const a4 = a3*X[0];
        Scalar const a5 = std::sqrt(30);
        Scalar const a6 = 2*a5;
        Scalar const a7 = 6*a5;
        G[0] = 0;
        G[1] = 0;
        G[2] = 6;
        G[3] = 0;
        G[4] = 2*a0;
        G[5] = 4*a0;
        G[6] = 20*a1*X[0] - 8*a1;
        G[7] = 0;
        G[8] = -18*a2 + a3*X[1] + a4;
        G[9] = -6*a2 + a4;
        G[10] = a6*X[0] - a6 + a7*X[1];
        G[11] = 12*a5*X[1] + a7*X[0] - a7;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = a0*X[0];
        Scalar const a2 = 2*X[0];
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5 = 2*a4;
        Scalar const a6 = -a5;
        Scalar const a7 = std::sqrt(6);
        Scalar const a8 = a7*X[0];
        Scalar const a9 = a3*a7;
        Scalar const a10 = X[0]*X[0]*X[0];
        Scalar const a11 = 3*a0;
        Scalar const a12 = -a11;
        Scalar const a13 = a0*X[1];
        Scalar const a14 = std::sqrt(30);
        Scalar const a15 = 3*a14;
        Scalar const a16 = 6*a14;
        Scalar const a17 = X[1]*X[1];
        P[0] = a1;
        P[1] = -a2 + 3*a3;
        P[2] = a3*a4 + (4*a4*X[1] + a6)*X[0];
        P[3] = (10.0/3.0)*a10*a7 + a8 - 4*a9;
        P[4] = 5*a0*a10 + a3*(-9*a0 + 15*a13) + (-a12 - 6*a13)*X[0];
        P[5] = (1.0/3.0)*a10*a14 + a3*(-a14 + a15*X[1]) + (a14 + a16*a17 - a16*X[1])*X[0];
        P[6] = a13;
        P[7] = (6*X[0] - 2)*X[1];
        P[8] = a17*a5 + (a2*a4 + a6)*X[1];
        P[9] = (a7 - 8*a8 + 10*a9)*X[1];
        P[10] = a17*(15*a1 + a12) + (15*a0*a3 - 18*a1 + a11)*X[1];
        P[11] = 2*a14*X[1]*X[1]*X[1] + a17*(a15*X[0] - a15) + (-a14*a2 + a14*a3 + a14)*X[1];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<2, 3>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 10;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = 2*a2;
        Scalar const a4 = 4*X[1];
        Scalar const a5 = std::sqrt(6);
        Scalar const a6 = a5*X[0];
        Scalar const a7 = X[0]*X[0];
        Scalar const a8 = a5*a7;
        Scalar const a9 = a0*X[0];
        Scalar const a10 = 6*X[1];
        Scalar const a11 = a0*a7;
        Scalar const a12 = 30*a9;
        Scalar const a13 = std::sqrt(30);
        Scalar const a14 = X[1]*X[1];
        Scalar const a15 = X[0]*X[0]*X[0];
        Scalar const a16 = 84*X[1];
        Scalar const a17 = std::sqrt(10);
        Scalar const a18 = a17*X[0];
        Scalar const a19 = 12*a17;
        Scalar const a20 = a17*a7;
        Scalar const a21 = std::sqrt(14);
        Scalar const a22 = 2*a21;
        Scalar const a23 = a21*X[1];
        Scalar const a24 = a21*a7;
        Scalar const a25 = 60*a14*a21;
        P[0] = a0;
        P[1] = a1 - 2;
        P[2] = a2*a4 + a3*X[0] - a3;
        P[3] = a5 - 8*a6 + 10*a8;
        P[4] = -a0*a10 + 3*a0 + 15*a11 + a12*X[1] - 18*a9;
        P[5] = a1*a13*X[1] - a10*a13 + 6*a13*a14 + a13*a7 - 2*a13*X[0] + a13;
        P[6] = 70*a0*a15 - 2*a0 - 90*a11 + a12;
        P[7] = 42*a15*a5 + a16*a8 + a4*a5 - 2*a5 - 48*a6*X[1] + 26*a6 - 66*a8;
        P[8] = 84*a14*a18 - a14*a19 + 14*a15*a17 + a16*a20 - 2*a17 - 96*a18*X[1] + 18*a18 + a19*X[1] - 30*a20;
        P[9] = a1*a21 + a15*a22 + 40*a21*X[1]*X[1]*X[1] - a22 - 48*a23*X[0] + 24*a23 + 24*a24*X[1] - 6*a24 + a25*X[0] - a25;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(6);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a4 = 30*a3;
        Scalar const a5 = a4*X[0];
        Scalar const a6 = std::sqrt(30);
        Scalar const a7 = 2*a6;
        Scalar const a8 = 6*a6;
        Scalar const a9 = X[0]*X[0];
        Scalar const a10 = 48*X[1];
        Scalar const a11 = a1*a9;
        Scalar const a12 = 168*X[1];
        Scalar const a13 = std::sqrt(10);
        Scalar const a14 = a13*X[0];
        Scalar const a15 = 96*a13;
        Scalar const a16 = a13*a9;
        Scalar const a17 = X[1]*X[1];
        Scalar const a18 = a12*a14;
        Scalar const a19 = std::sqrt(14);
        Scalar const a20 = 6*a19;
        Scalar const a21 = a19*X[0];
        Scalar const a22 = a10*a19;
        Scalar const a23 = a17*a19;
        Scalar const a24 = 24*a19;
        Scalar const a25 = 120*X[1];
        G[0] = 0;
        G[1] = 0;
        G[2] = 6;
        G[3] = 0;
        G[4] = 2*a0;
        G[5] = 4*a0;
        G[6] = -8*a1 + 20*a2;
        G[7] = 0;
        G[8] = -18*a3 + a4*X[1] + a5;
        G[9] = -6*a3 + a5;
        G[10] = a7*X[0] - a7 + a8*X[1];
        G[11] = 12*a6*X[1] + a8*X[0] - a8;
        G[12] = 210*a3*a9 - 180*a3*X[0] + a4;
        G[13] = 0;
        G[14] = -a1*a10 + 26*a1 + 126*a11 + a12*a2 - 132*a2;
        G[15] = 4*a1 + 84*a11 - 48*a2;
        G[16] = 84*a13*a17 + 18*a13 - 60*a14 - a15*X[1] + 42*a16 + a18;
        G[17] = -24*a13*X[1] + 12*a13 - a15*X[0] + 84*a16 + a18;
        G[18] = a20*a9 + a20 - 12*a21 + a22*X[0] - a22 + 60*a23;
        G[19] = -a19*a25 + a21*a25 - 48*a21 + 120*a23 + a24*a9 + a24;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = a0*X[0];
        Scalar const a2 = 2*X[0];
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5 = 2*a4;
        Scalar const a6 = -a5;
        Scalar const a7 = 4*X[1];
        Scalar const a8 = std::sqrt(6);
        Scalar const a9 = a8*X[0];
        Scalar const a10 = a3*a8;
        Scalar const a11 = X[0]*X[0]*X[0];
        Scalar const a12 = a11*a8;
        Scalar const a13 = a0*a11;
        Scalar const a14 = 3*a0;
        Scalar const a15 = -a14;
        Scalar const a16 = a0*X[1];
        Scalar const a17 = std::sqrt(30);
        Scalar const a18 = 3*a17;
        Scalar const a19 = 6*a17;
        Scalar const a20 = X[1]*X[1];
        Scalar const a21 = a0*a3;
        Scalar const a22 = 15*a21;
        Scalar const a23 = X[0]*X[0]*X[0]*X[0];
        Scalar const a24 = 2*a8;
        Scalar const a25 = -a24;
        Scalar const a26 = a8*X[1];
        Scalar const a27 = std::sqrt(10);
        Scalar const a28 = a27*X[1];
        Scalar const a29 = 2*a27;
        Scalar const a30 = a20*a27;
        Scalar const a31 = std::sqrt(14);
        Scalar const a32 = 2*a31;
        Scalar const a33 = -a32;
        Scalar const a34 = a31*X[1];
        Scalar const a35 = 24*a34;
        Scalar const a36 = a20*a31;
        Scalar const a37 = X[1]*X[1]*X[1];
        Scalar const a38 = 6*X[0];
        Scalar const a39 = a27*X[0];
        Scalar const a40 = a27*a3;
        Scalar const a41 = 20*a31;
        Scalar const a42 = 12*a31;
        P[0] = a1;
        P[1] = -a2 + 3*a3;
        P[2] = a3*a4 + (a4*a7 + a6)*X[0];
        P[3] = -4*a10 + (10.0/3.0)*a12 + a9;
        P[4] = 5*a13 + a3*(-9*a0 + 15*a16) + (-a15 - 6*a16)*X[0];
        P[5] = (1.0/3.0)*a11*a17 + a3*(-a17 + a18*X[1]) + (a17 + a19*a20 - a19*X[1])*X[0];
        P[6] = (35.0/2.0)*a0*a23 - 2*a1 - 30*a13 + a22;
        P[7] = a11*(28*a26 - 22*a8) + (21.0/2.0)*a23*a8 + a3*(-24*a26 + 13*a8) + (a25 + a7*a8)*X[0];
        P[8] = a11*(-10*a27 + 28*a28) + (7.0/2.0)*a23*a27 + a3*(9*a27 - 48*a28 + 42*a30) + (12*a27*X[1] - a29 - 12*a30)*X[0];
        P[9] = a11*(a33 + 8*a34) + (1.0/2.0)*a23*a31 + a3*(3*a31 - a35 + 30*a36) + (40*a31*a37 + a33 + a35 - 60*a36)*X[0];
        P[10] = a16;
        P[11] = (a38 - 2)*X[1];
        P[12] = a20*a5 + (a2*a4 + a6)*X[1];
        P[13] = (10*a10 + a8 - 8*a9)*X[1];
        P[14] = a20*(15*a1 + a15) + (-18*a1 + a14 + a22)*X[1];
        P[15] = 2*a17*a37 + a20*(a18*X[0] - a18) + (-a17*a2 + a17*a3 + a17)*X[1];
        P[16] = (-2*a0 + 30*a1 + 70*a13 - 90*a21)*X[1];
        P[17] = a20*(42*a10 + a24 - 24*a9) + (-66*a10 + 42*a12 + a25 + 26*a9)*X[1];
        P[18] = a20*(6*a27 - 48*a39 + 42*a40) + a37*(-4*a27 + 28*a39) + (14*a11*a27 - a29 + 18*a39 - 30*a40)*X[1];
        P[19] = a20*(a3*a42 - 24*a31*X[0] + a42) + 10*a31*X[1]*X[1]*X[1]*X[1] + a37*(a41*X[0] - a41) + (a11*a32 - 6*a3*a31 + a31*a38 + a33)*X[1];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<2, 4>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 15;

    Vector<Size> eval(Vector<2> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = 2*a2;
        Scalar const a4 = 4*X[1];
        Scalar const a5 = std::sqrt(6);
        Scalar const a6 = a5*X[0];
        Scalar const a7 = X[0]*X[0];
        Scalar const a8 = a5*a7;
        Scalar const a9 = a0*X[0];
        Scalar const a10 = 6*X[1];
        Scalar const a11 = a0*a7;
        Scalar const a12 = 30*a9;
        Scalar const a13 = std::sqrt(30);
        Scalar const a14 = a13*X[0];
        Scalar const a15 = a13*a7;
        Scalar const a16 = X[1]*X[1];
        Scalar const a17 = a13*X[1];
        Scalar const a18 = X[0]*X[0]*X[0];
        Scalar const a19 = a0*a18;
        Scalar const a20 = 84*X[1];
        Scalar const a21 = std::sqrt(10);
        Scalar const a22 = a21*X[0];
        Scalar const a23 = 12*a21;
        Scalar const a24 = a21*a7;
        Scalar const a25 = a18*a21;
        Scalar const a26 = a22*X[1];
        Scalar const a27 = a16*a22;
        Scalar const a28 = std::sqrt(14);
        Scalar const a29 = 2*a28;
        Scalar const a30 = a28*X[1];
        Scalar const a31 = a28*a7;
        Scalar const a32 = 60*a16*a28;
        Scalar const a33 = X[1]*X[1]*X[1];
        Scalar const a34 = X[0]*X[0]*X[0]*X[0];
        Scalar const a35 = 168*a13*a18;
        Scalar const a36 = 30*a0;
        Scalar const a37 = std::sqrt(70);
        Scalar const a38 = 12*a37;
        Scalar const a39 = 30*a37;
        Scalar const a40 = a18*a37;
        Scalar const a41 = a33*a37;
        Scalar const a42 = a37*X[0];
        Scalar const a43 = a37*a7;
        Scalar const a44 = 270*a16;
        Scalar const a45 = 3*a21;
        Scalar const a46 = 60*X[1];
        Scalar const a47 = 420*a33;
        P[0] = a0;
        P[1] = a1 - 2;
        P[2] = a2*a4 + a3*X[0] - a3;
        P[3] = a5 - 8*a6 + 10*a8;
        P[4] = -a0*a10 + 3*a0 + 15*a11 + a12*X[1] - 18*a9;
        P[5] = a1*a17 - a10*a13 + 6*a13*a16 + a13 - 2*a14 + a15;
        P[6] = -2*a0 - 90*a11 + a12 + 70*a19;
        P[7] = 42*a18*a5 + a20*a8 + a4*a5 - 2*a5 - 48*a6*X[1] + 26*a6 - 66*a8;
        P[8] = -a16*a23 + a20*a24 - 2*a21 + 18*a22 + a23*X[1] - 30*a24 + 14*a25 - 96*a26 + 84*a27;
        P[9] = a1*a28 + a18*a29 + 40*a28*a33 - a29 - 48*a30*X[0] + 24*a30 + 24*a31*X[1] - 6*a31 + a32*X[0] - a32;
        P[10] = 126*a21*a34 + a21 - 24*a22 + 126*a24 - 224*a25;
        P[11] = 84*a13*a34 + a13 + 42*a14*X[1] - 22*a14 - 168*a15*X[1] + 105*a15 - 2*a17 + a35*X[1] - a35;
        P[12] = 180*a0*a34 + 5*a0 + 1080*a11*a16 - 1560*a11*X[1] + 345*a11 + a16*a36 - 480*a16*a9 + 1080*a19*X[1] - 440*a19 - a36*X[1] + 510*a9*X[1] - 90*a9;
        P[13] = a16*a39 - 300*a16*a42 + 9*a34*a37 + a37 - a38*X[0] - a38*X[1] + a39*a7 + 108*a40*X[1] - 28*a40 + 180*a41*X[0] - 20*a41 + 132*a42*X[1] + a43*a44 - 228*a43*X[1];
        P[14] = -a18*a23 + a21*a44 - a21*a46 - a21*a47 + 210*a21*X[1]*X[1]*X[1]*X[1] + a22*a47 - a23*X[0] + a24*a44 - 180*a24*X[1] + 18*a24 + a25*a46 + 180*a26 - 540*a27 + a34*a45 + a45;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<2> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::sqrt(6);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a4 = 30*a3;
        Scalar const a5 = a4*X[0];
        Scalar const a6 = std::sqrt(30);
        Scalar const a7 = 2*a6;
        Scalar const a8 = -a7;
        Scalar const a9 = 6*a6;
        Scalar const a10 = a6*X[1];
        Scalar const a11 = a3*X[0];
        Scalar const a12 = X[0]*X[0];
        Scalar const a13 = a12*a3;
        Scalar const a14 = 48*X[1];
        Scalar const a15 = a1*a12;
        Scalar const a16 = 168*X[1];
        Scalar const a17 = std::sqrt(10);
        Scalar const a18 = 60*a17;
        Scalar const a19 = 96*a17;
        Scalar const a20 = a12*a17;
        Scalar const a21 = X[1]*X[1];
        Scalar const a22 = a17*a21;
        Scalar const a23 = a17*X[0];
        Scalar const a24 = a16*a23;
        Scalar const a25 = 12*a17;
        Scalar const a26 = 24*a17;
        Scalar const a27 = std::sqrt(14);
        Scalar const a28 = 6*a27;
        Scalar const a29 = a27*X[0];
        Scalar const a30 = a14*a27;
        Scalar const a31 = a21*a27;
        Scalar const a32 = 24*a27;
        Scalar const a33 = 120*X[1];
        Scalar const a34 = X[0]*X[0]*X[0];
        Scalar const a35 = a6*X[0];
        Scalar const a36 = a12*a6;
        Scalar const a37 = a34*a6;
        Scalar const a38 = a3*X[1];
        Scalar const a39 = a3*a34;
        Scalar const a40 = a11*X[1];
        Scalar const a41 = a13*X[1];
        Scalar const a42 = std::sqrt(70);
        Scalar const a43 = a21*a42;
        Scalar const a44 = a12*a42;
        Scalar const a45 = a34*a42;
        Scalar const a46 = 60*a42;
        Scalar const a47 = 132*a42;
        Scalar const a48 = X[1]*X[1]*X[1];
        Scalar const a49 = a42*X[0]*X[1];
        Scalar const a50 = a44*X[1];
        Scalar const a51 = -12*a42 + 540*a43*X[0];
        Scalar const a52 = a17*X[1];
        Scalar const a53 = 540*a22;
        Scalar const a54 = a17*a48;
        Scalar const a55 = a23*X[1];
        Scalar const a56 = 180*a20;
        Scalar const a57 = 1260*a22;
        G[0] = 0;
        G[1] = 0;
        G[2] = 6;
        G[3] = 0;
        G[4] = 2*a0;
        G[5] = 4*a0;
        G[6] = -8*a1 + 20*a2;
        G[7] = 0;
        G[8] = -18*a3 + a4*X[1] + a5;
        G[9] = -6*a3 + a5;
        G[10] = a7*X[0] + a8 + a9*X[1];
        G[11] = 12*a10 + a9*X[0] - a9;
        G[12] = -180*a11 + 210*a13 + a4;
        G[13] = 0;
        G[14] = -a1*a14 + 26*a1 + 126*a15 + a16*a2 - 132*a2;
        G[15] = 4*a1 + 84*a15 - 48*a2;
        G[16] = 18*a17 - a18*X[0] - a19*X[1] + 42*a20 + 84*a22 + a24;
        G[17] = -a19*X[0] + 84*a20 + a24 + a25 - a26*X[1];
        G[18] = a12*a28 + a28 - 12*a29 + a30*X[0] - a30 + 60*a31;
        G[19] = a12*a32 - a27*a33 + a29*a33 - 48*a29 + 120*a31 + a32;
        G[20] = 504*a17*a34 - 672*a20 + 252*a23 - a26;
        G[21] = 0;
        G[22] = 504*a10*a12 - 336*a10*X[0] + 42*a10 + 210*a35 - 504*a36 + 336*a37 - 22*a6;
        G[23] = 42*a35 - 168*a36 + 168*a37 + a8;
        G[24] = 2160*a11*a21 + 690*a11 - 1320*a13 - 480*a21*a3 - 90*a3 + 510*a38 + 720*a39 - 3120*a40 + 3240*a41;
        G[25] = 510*a11 - 1560*a13 + 60*a38 + 1080*a39 - a4 - 960*a40 + 2160*a41;
        G[26] = 180*a42*a48 - 300*a43 - 84*a44 + 36*a45 + a46*X[0] + a47*X[1] - 456*a49 + 324*a50 + a51;
        G[27] = -60*a43 - 228*a44 + 108*a45 + a46*X[1] + a47*X[0] - 600*a49 + 540*a50 + a51;
        G[28] = -36*a20 + 36*a23 + a25*a34 - a25 + 180*a52 + a53*X[0] - a53 + 420*a54 - 360*a55 + a56*X[1];
        G[29] = a18*a34 - a18 + 540*a20*X[1] + 180*a23 + 540*a52 + 840*a54 - 1080*a55 - a56 + a57*X[0] - a57;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<2> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = a0*X[0];
        Scalar const a2 = 2*X[0];
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5 = 2*a4;
        Scalar const a6 = -a5;
        Scalar const a7 = 4*X[1];
        Scalar const a8 = std::sqrt(6);
        Scalar const a9 = a8*X[0];
        Scalar const a10 = a3*a8;
        Scalar const a11 = X[0]*X[0]*X[0];
        Scalar const a12 = a11*a8;
        Scalar const a13 = 5*a0;
        Scalar const a14 = 3*a0;
        Scalar const a15 = -a14;
        Scalar const a16 = a0*X[1];
        Scalar const a17 = std::sqrt(30);
        Scalar const a18 = a11*a17;
        Scalar const a19 = -a17;
        Scalar const a20 = 3*a17;
        Scalar const a21 = 6*a17;
        Scalar const a22 = X[1]*X[1];
        Scalar const a23 = 15*a0;
        Scalar const a24 = a23*a3;
        Scalar const a25 = 30*a0;
        Scalar const a26 = X[0]*X[0]*X[0]*X[0];
        Scalar const a27 = a0*a26;
        Scalar const a28 = 2*a8;
        Scalar const a29 = -a28;
        Scalar const a30 = a8*X[1];
        Scalar const a31 = std::sqrt(10);
        Scalar const a32 = a26*a31;
        Scalar const a33 = a31*X[1];
        Scalar const a34 = 2*a31;
        Scalar const a35 = a22*a31;
        Scalar const a36 = std::sqrt(14);
        Scalar const a37 = 2*a36;
        Scalar const a38 = -a37;
        Scalar const a39 = a36*X[1];
        Scalar const a40 = 24*a39;
        Scalar const a41 = a22*a36;
        Scalar const a42 = X[1]*X[1]*X[1];
        Scalar const a43 = a31*X[0];
        Scalar const a44 = a3*a31;
        Scalar const a45 = a11*a31;
        Scalar const a46 = std::pow(X[0], 5);
        Scalar const a47 = a31*a46;
        Scalar const a48 = a17*X[1];
        Scalar const a49 = 42*a17;
        Scalar const a50 = a0*a22;
        Scalar const a51 = std::sqrt(70);
        Scalar const a52 = a51*X[1];
        Scalar const a53 = 10*a51;
        Scalar const a54 = a22*a51;
        Scalar const a55 = a42*a51;
        Scalar const a56 = -6*a51;
        Scalar const a57 = 3*a31;
        Scalar const a58 = 6*a31;
        Scalar const a59 = -60*a33;
        Scalar const a60 = 90*a31;
        Scalar const a61 = 270*a35;
        Scalar const a62 = a31*a42;
        Scalar const a63 = X[1]*X[1]*X[1]*X[1];
        Scalar const a64 = 6*X[0];
        Scalar const a65 = a17*a3;
        Scalar const a66 = a0*a3;
        Scalar const a67 = a0*a11;
        Scalar const a68 = 30*a31;
        Scalar const a69 = 20*a36;
        Scalar const a70 = 12*a36;
        Scalar const a71 = a17*X[0];
        Scalar const a72 = a51*X[0];
        Scalar const a73 = a3*a51;
        Scalar const a74 = a11*a51;
        Scalar const a75 = a3*a60;
        P[0] = a1;
        P[1] = -a2 + 3*a3;
        P[2] = a3*a4 + (a4*a7 + a6)*X[0];
        P[3] = -4*a10 + (10.0/3.0)*a12 + a9;
        P[4] = a11*a13 + a3*(-9*a0 + 15*a16) + (-a15 - 6*a16)*X[0];
        P[5] = (1.0/3.0)*a18 + a3*(a19 + a20*X[1]) + (a17 + a21*a22 - a21*X[1])*X[0];
        P[6] = -2*a1 - a11*a25 + a24 + (35.0/2.0)*a27;
        P[7] = a11*(28*a30 - 22*a8) + (21.0/2.0)*a26*a8 + a3*(-24*a30 + 13*a8) + (a29 + a7*a8)*X[0];
        P[8] = a11*(-10*a31 + 28*a33) + a3*(9*a31 - 48*a33 + 42*a35) + (7.0/2.0)*a32 + (12*a31*X[1] - a34 - 12*a35)*X[0];
        P[9] = a11*(a38 + 8*a39) + (1.0/2.0)*a26*a36 + a3*(3*a36 - a40 + 30*a41) + (40*a36*a42 + a38 + a40 - 60*a41)*X[0];
        P[10] = -56*a32 + a43 - 12*a44 + 42*a45 + (126.0/5.0)*a47;
        P[11] = a11*(35*a17 - 56*a48) + (84.0/5.0)*a17*a46 + a26*(a49*X[1] - a49) + a3*(-11*a17 + 21*a48) + (-a19 - 2*a48)*X[0];
        P[12] = 36*a0*a46 + a11*(115*a0 - 520*a16 + 360*a50) + a26*(-110*a0 + 270*a16) + a3*(255*a0*X[1] - 45*a0 - 240*a50) + (a13 - 30*a16 + a22*a25)*X[0];
        P[13] = a11*(-76*a52 + a53 + 90*a54) + a26*(-7*a51 + 27*a52) + a3*(66*a52 - 150*a54 + 90*a55 + a56) + (9.0/5.0)*a46*a51 + (30*a22*a51 + a51 - 12*a52 - 20*a55)*X[0];
        P[14] = a11*(a22*a60 + a58 + a59) + a26*(15*a33 - a57) + a3*(-a58 + a60*X[1] - a61 + 210*a62) + (3.0/5.0)*a47 + (210*a31*a63 + a57 + a59 + a61 - 420*a62)*X[0];
        P[15] = a16;
        P[16] = (a64 - 2)*X[1];
        P[17] = a22*a5 + (a2*a4 + a6)*X[1];
        P[18] = (10*a10 + a8 - 8*a9)*X[1];
        P[19] = a22*(15*a1 + a15) + (-18*a1 + a14 + a24)*X[1];
        P[20] = 2*a17*a42 + a22*(a20*X[0] - a20) + (-a17*a2 + a17 + a65)*X[1];
        P[21] = (-2*a0 + 30*a1 - 90*a66 + 70*a67)*X[1];
        P[22] = a22*(42*a10 + a28 - 24*a9) + (-66*a10 + 42*a12 + a29 + 26*a9)*X[1];
        P[23] = a22*(-48*a43 + 42*a44 + a58) + a42*(-4*a31 + 28*a43) + (-a3*a68 - a34 + 18*a43 + 14*a45)*X[1];
        P[24] = a22*(a3*a70 - 24*a36*X[0] + a70) + 10*a36*a63 + a42*(a69*X[0] - a69) + (a11*a37 - 6*a3*a36 + a36*a64 + a38)*X[1];
        P[25] = (a31 + 126*a32 - 24*a43 + 126*a44 - 224*a45)*X[1];
        P[26] = a22*(84*a18 + a19 - 84*a65 + 21*a71) + (84*a17*a26 + a17 - 168*a18 + 105*a65 - 22*a71)*X[1];
        P[27] = a22*(255*a1 - a23 - 780*a66 + 540*a67) + a42*(10*a0 - 160*a1 + 360*a66) + (-90*a1 + a13 + 180*a27 + 345*a66 - 440*a67)*X[1];
        P[28] = a22*(a56 + 66*a72 - 114*a73 + 54*a74) + a42*(a53 - 100*a72 + 90*a73) + a63*(-5*a51 + 45*a72) + (9*a26*a51 + a51 - 12*a72 + 30*a73 - 28*a74)*X[1];
        P[29] = a22*(a11*a68 + 90*a43 - a68 - a75) + 42*a31*std::pow(X[1], 5) + a42*(-180*a43 + a60 + a75) + a63*(-105*a31 + 105*a43) + (a26*a57 - 12*a43 + 18*a44 - 12*a45 + a57)*X[1];
        return P;
    }
};

/**
 * Orthonormalized polynomial basis on reference tetrahedron
 */

template <>
class OrthonormalPolynomialBasis<3, 1>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 4;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::sqrt(10);
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = 2*a1;
        Scalar const a3 = std::sqrt(15);
        Scalar const a4 = 2*a3;
        P[0] = std::sqrt(6);
        P[1] = 4*a0*X[0] - a0;
        P[2] = 6*a1*X[1] + a2*X[0] - a2;
        P[3] = 4*a3*X[2] + a4*X[0] + a4*X[1] - a4;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(5);
        Scalar const a1 = std::sqrt(15);
        Scalar const a2 = 2*a1;
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 4*std::sqrt(10);
        G[4] = 0;
        G[5] = 0;
        G[6] = 2*a0;
        G[7] = 6*a0;
        G[8] = 0;
        G[9] = a2;
        G[10] = a2;
        G[11] = 4*a1;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::sqrt(5);
        Scalar const a5 = 2*a4;
        Scalar const a6 = -a5;
        Scalar const a7 = 6*a4*X[1] + a6;
        Scalar const a8 = std::sqrt(15);
        Scalar const a9 = 2*a8;
        Scalar const a10 = a9*X[1];
        Scalar const a11 = -a9;
        Scalar const a12 = a11 + 4*a8*X[2];
        Scalar const a13 = -a1 + 4*a2;
        Scalar const a14 = X[1]*X[1];
        Scalar const a15 = a5*X[0];
        Scalar const a16 = a9*X[0];
        P[0] = a0*X[0];
        P[1] = 2*a1*a3 - a2;
        P[2] = a3*a4 + a7*X[0];
        P[3] = a3*a8 + (a10 + a12)*X[0];
        P[4] = a0*X[1];
        P[5] = a13*X[1];
        P[6] = 3*a14*a4 + (a15 + a6)*X[1];
        P[7] = a14*a8 + (a12 + a16)*X[1];
        P[8] = a0*X[2];
        P[9] = a13*X[2];
        P[10] = (a15 + a7)*X[2];
        P[11] = a9*X[2]*X[2] + (a10 + a11 + a16)*X[2];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<3, 2>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 10;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::sqrt(10);
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = 2*a1;
        Scalar const a3 = 6*X[1];
        Scalar const a4 = std::sqrt(15);
        Scalar const a5 = 2*a4;
        Scalar const a6 = 4*X[2];
        Scalar const a7 = std::sqrt(14);
        Scalar const a8 = a7*X[0];
        Scalar const a9 = X[0]*X[0];
        Scalar const a10 = 15*a7;
        Scalar const a11 = std::sqrt(7);
        Scalar const a12 = 14*X[0];
        Scalar const a13 = 12*a9;
        Scalar const a14 = X[0]*X[1];
        Scalar const a15 = std::sqrt(42);
        Scalar const a16 = 2*X[0];
        Scalar const a17 = 8*a15*X[1];
        Scalar const a18 = X[1]*X[1];
        Scalar const a19 = std::sqrt(21);
        Scalar const a20 = 2*a19;
        Scalar const a21 = 3*a7;
        Scalar const a22 = 6*a8;
        Scalar const a23 = 18*X[1];
        Scalar const a24 = 6*X[2];
        Scalar const a25 = std::sqrt(210);
        Scalar const a26 = a16*a25;
        Scalar const a27 = a24*a25;
        P[0] = std::sqrt(6);
        P[1] = 4*a0*X[0] - a0;
        P[2] = a1*a3 + a2*X[0] - a2;
        P[3] = a4*a6 + a5*X[0] + a5*X[1] - a5;
        P[4] = a10*a9 + a7 - 10*a8;
        P[5] = -a11*a12 + a11*a13 + 36*a11*a14 - a11*a3 + 2*a11;
        P[6] = -a15*a16 + 10*a15*a18 + a15*a9 + a15 + a17*X[0] - a17;
        P[7] = -a12*a19 + a13*a19 + 12*a14*a19 - a19*a6 + 24*a19*X[0]*X[2] - a20*X[1] + a20;
        P[8] = a10*a18 + a21*a9 + a21 + a22*X[2] - a22 - a23*a7 + a23*a8 - a24*a7 + 30*a7*X[1]*X[2];
        P[9] = a18*a25 + a25*a3*X[2] + a25*a9 - 2*a25*X[1] + 6*a25*X[2]*X[2] + a25 + a26*X[1] - a26 + a27*X[0] - a27;
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(5);
        Scalar const a1 = std::sqrt(15);
        Scalar const a2 = 2*a1;
        Scalar const a3 = std::sqrt(14);
        Scalar const a4 = 30*a3;
        Scalar const a5 = std::sqrt(7);
        Scalar const a6 = a5*X[0];
        Scalar const a7 = std::sqrt(42);
        Scalar const a8 = 2*a7;
        Scalar const a9 = 8*a7;
        Scalar const a10 = std::sqrt(21);
        Scalar const a11 = 24*a10;
        Scalar const a12 = a11*X[0];
        Scalar const a13 = 12*a10;
        Scalar const a14 = 6*a3;
        Scalar const a15 = 18*a3;
        Scalar const a16 = a14*X[0] - a14;
        Scalar const a17 = a4*X[1];
        Scalar const a18 = std::sqrt(210);
        Scalar const a19 = 2*a18;
        Scalar const a20 = 6*a18;
        Scalar const a21 = a19*X[0] + a19*X[1] - a19 + a20*X[2];
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 4*std::sqrt(10);
        G[4] = 0;
        G[5] = 0;
        G[6] = 2*a0;
        G[7] = 6*a0;
        G[8] = 0;
        G[9] = a2;
        G[10] = a2;
        G[11] = 4*a1;
        G[12] = -10*a3 + a4*X[0];
        G[13] = 0;
        G[14] = 0;
        G[15] = 36*a5*X[1] - 14*a5 + 24*a6;
        G[16] = -6*a5 + 36*a6;
        G[17] = 0;
        G[18] = a8*X[0] - a8 + a9*X[1];
        G[19] = 20*a7*X[1] + a9*X[0] - a9;
        G[20] = 0;
        G[21] = -14*a10 + a11*X[2] + a12 + a13*X[1];
        G[22] = -2*a10 + a13*X[0];
        G[23] = -4*a10 + a12;
        G[24] = a14*X[2] + a15*X[1] + a16;
        G[25] = a15*X[0] - a15 + a17 + a4*X[2];
        G[26] = a16 + a17;
        G[27] = a21;
        G[28] = a21;
        G[29] = 12*a18*X[2] + a20*X[0] + a20*X[1] - a20;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = std::sqrt(5);
        Scalar const a5 = 2*a4;
        Scalar const a6 = -a5;
        Scalar const a7 = 6*X[1];
        Scalar const a8 = a4*a7 + a6;
        Scalar const a9 = std::sqrt(15);
        Scalar const a10 = 2*a9;
        Scalar const a11 = a10*X[1];
        Scalar const a12 = -a10;
        Scalar const a13 = 4*X[2];
        Scalar const a14 = a12 + a13*a9;
        Scalar const a15 = std::sqrt(14);
        Scalar const a16 = a15*X[0];
        Scalar const a17 = a15*a3;
        Scalar const a18 = X[0]*X[0]*X[0];
        Scalar const a19 = a15*a18;
        Scalar const a20 = std::sqrt(7);
        Scalar const a21 = 4*a18;
        Scalar const a22 = a20*a7;
        Scalar const a23 = 18*X[1];
        Scalar const a24 = std::sqrt(42);
        Scalar const a25 = (1.0/3.0)*a18;
        Scalar const a26 = 4*a24;
        Scalar const a27 = 8*a24*X[1];
        Scalar const a28 = X[1]*X[1];
        Scalar const a29 = 10*a24*a28 + a24 - a27;
        Scalar const a30 = std::sqrt(21);
        Scalar const a31 = 2*a30;
        Scalar const a32 = -a31;
        Scalar const a33 = a31*X[1];
        Scalar const a34 = a13*a30;
        Scalar const a35 = 12*a30;
        Scalar const a36 = 3*a15;
        Scalar const a37 = -a36;
        Scalar const a38 = 9*a15;
        Scalar const a39 = 6*X[2];
        Scalar const a40 = -a15*a39 + a36;
        Scalar const a41 = 15*a15;
        Scalar const a42 = -a15*a23 + a28*a41;
        Scalar const a43 = std::sqrt(210);
        Scalar const a44 = a43*X[1];
        Scalar const a45 = 3*a43;
        Scalar const a46 = -a43 + a45*X[2];
        Scalar const a47 = X[2]*X[2];
        Scalar const a48 = -a39*a43 + 6*a43*a47 + a43;
        Scalar const a49 = 2*a44;
        Scalar const a50 = a28*a43 - a49;
        Scalar const a51 = -a1 + 4*a2;
        Scalar const a52 = a5*X[0];
        Scalar const a53 = a10*X[0];
        Scalar const a54 = a15 - 10*a16 + 15*a17;
        Scalar const a55 = a20*X[0];
        Scalar const a56 = 12*a20*a3 + 2*a20 - 14*a55;
        Scalar const a57 = X[1]*X[1]*X[1];
        Scalar const a58 = a24*a3 - 2*a24*X[0];
        Scalar const a59 = a30*X[0];
        Scalar const a60 = a3*a35 + a31 - 14*a59;
        Scalar const a61 = -6*a16 + a3*a36;
        Scalar const a62 = a43*X[0];
        Scalar const a63 = a3*a43 - 2*a62;
        Scalar const a64 = a35*X[0];
        P[0] = a0*X[0];
        P[1] = 2*a1*a3 - a2;
        P[2] = a3*a4 + a8*X[0];
        P[3] = a3*a9 + (a11 + a14)*X[0];
        P[4] = a16 - 5*a17 + 5*a19;
        P[5] = a20*a21 + a3*(a20*a23 - 7*a20) + (2*a20 - a22)*X[0];
        P[6] = a24*a25 + a29*X[0] + a3*(-a24 + a26*X[1]);
        P[7] = a21*a30 + a3*(a30*a7 - 7*a30 + a35*X[2]) + (-a32 - a33 - a34)*X[0];
        P[8] = a19 + a3*(a36*X[2] + a37 + a38*X[1]) + (30*a15*X[1]*X[2] + a40 + a42)*X[0];
        P[9] = a25*a43 + a3*(a44 + a46) + (a39*a44 + a48 + a50)*X[0];
        P[10] = a0*X[1];
        P[11] = a51*X[1];
        P[12] = 3*a28*a4 + (a52 + a6)*X[1];
        P[13] = a28*a9 + (a14 + a53)*X[1];
        P[14] = a54*X[1];
        P[15] = a28*(-3*a20 + 18*a55) + a56*X[1];
        P[16] = (10.0/3.0)*a24*a57 + a28*(a26*X[0] - a26) + (a24 + a58)*X[1];
        P[17] = a28*(-a30 + 6*a59) + (-a34 + 24*a59*X[2] + a60)*X[1];
        P[18] = 5*a15*a57 + a28*(9*a16 - a38 + a41*X[2]) + (a16*a39 + a40 + a61)*X[1];
        P[19] = a28*(a46 + a62) + (1.0/3.0)*a43*a57 + (a39*a62 + a48 + a63)*X[1];
        P[20] = a0*X[2];
        P[21] = a51*X[2];
        P[22] = (a52 + a8)*X[2];
        P[23] = a10*a47 + (a11 + a12 + a53)*X[2];
        P[24] = a54*X[2];
        P[25] = (-a22 + 36*a55*X[1] + a56)*X[2];
        P[26] = (a27*X[0] + a29 + a58)*X[2];
        P[27] = a47*(a32 + a64) + (-a33 + a60 + a64*X[1])*X[2];
        P[28] = a47*(3*a16 + a37 + a41*X[1]) + (a16*a23 + a36 + a42 + a61)*X[2];
        P[29] = 2*a43*X[2]*X[2]*X[2] + a47*(3*a44 - a45 + 3*a62) + (a43 + a49*X[0] + a50 + a63)*X[2];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<3, 3>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 20;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = std::sqrt(5);
        Scalar const a4 = 2*a3;
        Scalar const a5 = 6*X[1];
        Scalar const a6 = std::sqrt(15);
        Scalar const a7 = 2*a6;
        Scalar const a8 = 4*X[2];
        Scalar const a9 = std::sqrt(14);
        Scalar const a10 = a9*X[0];
        Scalar const a11 = X[0]*X[0];
        Scalar const a12 = 15*a9;
        Scalar const a13 = std::sqrt(7);
        Scalar const a14 = 14*X[0];
        Scalar const a15 = 12*a11;
        Scalar const a16 = X[0]*X[1];
        Scalar const a17 = std::sqrt(42);
        Scalar const a18 = 2*X[0];
        Scalar const a19 = 8*a17*X[1];
        Scalar const a20 = X[1]*X[1];
        Scalar const a21 = std::sqrt(21);
        Scalar const a22 = 2*a21;
        Scalar const a23 = X[0]*X[2];
        Scalar const a24 = 3*a9;
        Scalar const a25 = 6*a9;
        Scalar const a26 = a25*X[0];
        Scalar const a27 = 18*X[1];
        Scalar const a28 = a27*a9;
        Scalar const a29 = a9*X[2];
        Scalar const a30 = a29*X[1];
        Scalar const a31 = std::sqrt(210);
        Scalar const a32 = a18*a31;
        Scalar const a33 = 6*a31;
        Scalar const a34 = a33*X[2];
        Scalar const a35 = X[2]*X[2];
        Scalar const a36 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a37 = a36*X[0];
        Scalar const a38 = a11*a36;
        Scalar const a39 = X[0]*X[0]*X[0];
        Scalar const a40 = 168*a39;
        Scalar const a41 = 90*X[0];
        Scalar const a42 = 252*a11;
        Scalar const a43 = a11*X[1];
        Scalar const a44 = 30*a0;
        Scalar const a45 = 24*a0;
        Scalar const a46 = a0*a11;
        Scalar const a47 = a0*a16;
        Scalar const a48 = a0*a20;
        Scalar const a49 = a0*a43;
        Scalar const a50 = 6*a36;
        Scalar const a51 = a36*X[1];
        Scalar const a52 = 90*a51;
        Scalar const a53 = a20*a36;
        Scalar const a54 = X[1]*X[1]*X[1];
        Scalar const a55 = a37*X[1];
        Scalar const a56 = a20*a37;
        Scalar const a57 = a38*X[1];
        Scalar const a58 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a59 = 6*a58;
        Scalar const a60 = 12*X[2];
        Scalar const a61 = 168*a58;
        Scalar const a62 = 18*X[2];
        Scalar const a63 = 162*X[2];
        Scalar const a64 = 144*X[2];
        Scalar const a65 = 6*a0;
        Scalar const a66 = a0*X[1];
        Scalar const a67 = 198*a48;
        Scalar const a68 = 252*X[2];
        Scalar const a69 = std::sqrt(30);
        Scalar const a70 = 3*a69;
        Scalar const a71 = a69*X[0];
        Scalar const a72 = a11*a69;
        Scalar const a73 = 24*a69;
        Scalar const a74 = a35*a69;
        Scalar const a75 = a16*a69;
        Scalar const a76 = 6*a1;
        Scalar const a77 = a1*X[1];
        Scalar const a78 = 36*a1;
        Scalar const a79 = a78*X[2];
        Scalar const a80 = 18*a11;
        Scalar const a81 = a1*a20;
        Scalar const a82 = a2*X[1];
        Scalar const a83 = 288*X[2];
        Scalar const a84 = 72*a29;
        Scalar const a85 = 18*a20;
        Scalar const a86 = 180*a35;
        Scalar const a87 = a86*a9;
        Scalar const a88 = a10*a64;
        P[0] = a0;
        P[1] = -a1 + 4*a2;
        P[2] = a3*a5 + a4*X[0] - a4;
        P[3] = a6*a8 + a7*X[0] + a7*X[1] - a7;
        P[4] = -10*a10 + a11*a12 + a9;
        P[5] = -a13*a14 + a13*a15 + 36*a13*a16 - a13*a5 + 2*a13;
        P[6] = a11*a17 - a17*a18 + 10*a17*a20 + a17 + a19*X[0] - a19;
        P[7] = -a14*a21 + a15*a21 + 12*a16*a21 + 24*a21*a23 - a21*a8 - a22*X[1] + a22;
        P[8] = a10*a27 + a11*a24 + a12*a20 + a24 - a25*X[2] + a26*X[2] - a26 - a28 + 30*a30;
        P[9] = a11*a31 + a20*a31 + a31*a5*X[2] - 2*a31*X[1] + a31 + a32*X[1] - a32 + a33*a35 + a34*X[0] - a34;
        P[10] = a36*a40 - 3*a36 + 54*a37 - 189*a38;
        P[11] = -252*a16 + a27 + a40 + a41 - a42 + 504*a43 - 6;
        P[12] = -3*a0 - a20*a44 + a39*a45 + a44*X[0] + a45*X[1] - 51*a46 - 216*a47 + 240*a48*X[0] + 192*a49;
        P[13] = 210*a36*a54 + 18*a37 - 18*a38 + a39*a50 - a50 + a52 - 270*a53 - 180*a55 + 270*a56 + 90*a57;
        P[14] = 336*a11*a58*X[2] - 84*a16*a58 - a23*a61 + a40*a58 + a41*a58 - a42*a58 + a43*a61 + a58*a60 + a59*X[1] - a59;
        P[15] = 72*a36*a39 + a36*a41 + a36*a62 - 9*a36 - a37*a63 + a38*a64 - 153*a38 + 54*a51 - a52*X[2] - 45*a53 + 720*a55*X[2] - 486*a55 + 360*a56 + 432*a57;
        P[16] = a0*a15*X[2] + 126*a0*a54 + a0*a60 + 18*a0*X[0] - a23*a45 + a39*a65 - 18*a46 + a47*a64 - 156*a47 + a48*a68 + 78*a49 - a64*a66 - a65 + 78*a66 + a67*X[0] - a67;
        P[17] = -a20*a70 + a20*a73*X[0] - a27*a69*X[2] + a39*a73 + 48*a43*a69 + a5*a69 + a62*a69 - a63*a71 + a64*a72 + a64*a75 - a70 + 30*a71 - 51*a72 + 144*a74*X[0] - 18*a74 - 54*a75;
        P[18] = 54*a1*a43 + 42*a1*a54 - a1*a80 + a11*a79 + 36*a2*a35 - 72*a2*X[2] + 18*a2 + 252*a35*a77 - a35*a78 + a39*a76 + a41*a81 + a68*a81 - a76 - a77*a83 + 54*a77 + a79 - 90*a81 + a82*a83 - 108*a82;
        P[19] = a10*a85 + a10*a86 - 36*a10*X[1] + 18*a10 + a11*a28 + a11*a84 + a20*a84 + a25*a39 + a25*a54 - a25 + a28 - 144*a30 - a80*a9 + a84 - a85*a9 + a87*X[1] - a87 + a88*X[1] - a88 + 120*a9*X[2]*X[2]*X[2];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(10);
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = std::sqrt(15);
        Scalar const a3 = 2*a2;
        Scalar const a4 = std::sqrt(14);
        Scalar const a5 = 30*a4;
        Scalar const a6 = std::sqrt(7);
        Scalar const a7 = a6*X[0];
        Scalar const a8 = 36*X[1];
        Scalar const a9 = std::sqrt(42);
        Scalar const a10 = 2*a9;
        Scalar const a11 = 8*a9;
        Scalar const a12 = std::sqrt(21);
        Scalar const a13 = 24*a12;
        Scalar const a14 = a13*X[0];
        Scalar const a15 = 12*a12;
        Scalar const a16 = 6*a4;
        Scalar const a17 = 18*a4;
        Scalar const a18 = a16*X[0] - a16;
        Scalar const a19 = a5*X[1];
        Scalar const a20 = std::sqrt(210);
        Scalar const a21 = 2*a20;
        Scalar const a22 = 6*a20;
        Scalar const a23 = a21*X[0] + a21*X[1] - a21 + a22*X[2];
        Scalar const a24 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a25 = 54*a24;
        Scalar const a26 = a24*X[0];
        Scalar const a27 = X[0]*X[0];
        Scalar const a28 = 504*a27;
        Scalar const a29 = 504*X[0];
        Scalar const a30 = X[0]*X[1];
        Scalar const a31 = std::sqrt(6);
        Scalar const a32 = a31*X[0];
        Scalar const a33 = 216*a31;
        Scalar const a34 = a27*a31;
        Scalar const a35 = X[1]*X[1];
        Scalar const a36 = a31*a35;
        Scalar const a37 = a30*a31;
        Scalar const a38 = 24*a31;
        Scalar const a39 = a31*X[1];
        Scalar const a40 = 18*a24;
        Scalar const a41 = a24*X[1];
        Scalar const a42 = a24*a35;
        Scalar const a43 = 180*a26;
        Scalar const a44 = 90*a24;
        Scalar const a45 = a26*X[1];
        Scalar const a46 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a47 = 84*a46;
        Scalar const a48 = 168*a46;
        Scalar const a49 = 336*a46;
        Scalar const a50 = 162*X[2];
        Scalar const a51 = a24*a27;
        Scalar const a52 = a26*X[2];
        Scalar const a53 = -a44*X[1] + 720*a45;
        Scalar const a54 = 18*a31;
        Scalar const a55 = a38*X[2];
        Scalar const a56 = 144*X[2];
        Scalar const a57 = 78*a31;
        Scalar const a58 = 504*X[2];
        Scalar const a59 = 12*a31;
        Scalar const a60 = std::sqrt(30);
        Scalar const a61 = a60*X[0];
        Scalar const a62 = a60*X[1];
        Scalar const a63 = a27*a60;
        Scalar const a64 = X[2]*X[2];
        Scalar const a65 = 144*a60;
        Scalar const a66 = a30*a60;
        Scalar const a67 = 288*a61*X[2];
        Scalar const a68 = 6*a60;
        Scalar const a69 = 18*a60;
        Scalar const a70 = 18*a0;
        Scalar const a71 = a0*X[1];
        Scalar const a72 = 36*a0;
        Scalar const a73 = a0*a35;
        Scalar const a74 = a0*a30;
        Scalar const a75 = 288*a71;
        Scalar const a76 = a0*X[2];
        Scalar const a77 = 72*a76;
        Scalar const a78 = a77*X[0] - a77;
        Scalar const a79 = 54*a0;
        Scalar const a80 = a0*X[0];
        Scalar const a81 = 288*a76;
        Scalar const a82 = a58*a71;
        Scalar const a83 = a4*X[0];
        Scalar const a84 = a4*a56;
        Scalar const a85 = a4*a64;
        Scalar const a86 = a17*a27 + a17*a35 + a17 - a4*a8 + a8*a83 - 36*a83 + a84*X[0] + a84*X[1] - a84 + 180*a85;
        Scalar const a87 = 72*a4;
        Scalar const a88 = 144*a4;
        Scalar const a89 = 360*X[2];
        Scalar const a90 = a4*a89;
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 4*a0;
        G[4] = 0;
        G[5] = 0;
        G[6] = 2*a1;
        G[7] = 6*a1;
        G[8] = 0;
        G[9] = a3;
        G[10] = a3;
        G[11] = 4*a2;
        G[12] = -10*a4 + a5*X[0];
        G[13] = 0;
        G[14] = 0;
        G[15] = a6*a8 - 14*a6 + 24*a7;
        G[16] = -6*a6 + 36*a7;
        G[17] = 0;
        G[18] = a10*X[0] - a10 + a11*X[1];
        G[19] = a11*X[0] - a11 + 20*a9*X[1];
        G[20] = 0;
        G[21] = -14*a12 + a13*X[2] + a14 + a15*X[1];
        G[22] = -2*a12 + a15*X[0];
        G[23] = -4*a12 + a14;
        G[24] = a16*X[2] + a17*X[1] + a18;
        G[25] = a17*X[0] - a17 + a19 + a5*X[2];
        G[26] = a18 + a19;
        G[27] = a23;
        G[28] = a23;
        G[29] = 12*a20*X[2] + a22*X[0] + a22*X[1] - a22;
        G[30] = a24*a28 + a25 - 378*a26;
        G[31] = 0;
        G[32] = 0;
        G[33] = a28 - a29 + 1008*a30 - 252*X[1] + 90;
        G[34] = a28 - 252*X[0] + 18;
        G[35] = 0;
        G[36] = 30*a31 - 102*a32 - a33*X[1] + 72*a34 + 240*a36 + 384*a37;
        G[37] = -a33*X[0] + 192*a34 + 480*a37 + a38 - 60*a39;
        G[38] = 0;
        G[39] = -36*a26 + a27*a40 + a40 - 180*a41 + 270*a42 + a43*X[1];
        G[40] = a27*a44 - 540*a41 + 630*a42 - a43 + a44 + 540*a45;
        G[41] = 0;
        G[42] = a28*a46 - a29*a46 + a30*a49 + 672*a46*X[0]*X[2] + 90*a46 - a47*X[1] - a48*X[2];
        G[43] = a27*a48 + 6*a46 - a47*X[0];
        G[44] = a27*a49 + 12*a46 - a48*X[0];
        G[45] = -a24*a50 - 306*a26 + 720*a41*X[2] - 486*a41 + 360*a42 + a44 + 864*a45 + 216*a51 + 288*a52;
        G[46] = a25 - 486*a26 - a44*X[2] + 432*a51 + 720*a52 + a53;
        G[47] = -162*a26 + a40 + 144*a51 + a53;
        G[48] = a27*a54 - 36*a32 + 198*a36 + 156*a37 + a39*a56 - 156*a39 + a54 + a55*X[0] - a55;
        G[49] = a27*a57 - a31*a56 + a32*a56 - 156*a32 + 378*a36 + 396*a37 + a39*a58 - 396*a39 + a57;
        G[50] = a27*a59 + 252*a36 + 144*a37 - a38*X[0] - 144*a39 + a59;
        G[51] = 24*a35*a60 - a50*a60 + a56*a62 + 30*a60 - 102*a61 - 54*a62 + 72*a63 + a64*a65 + 96*a66 + a67;
        G[52] = a56*a61 - 54*a61 + 48*a63 + 48*a66 - a68*X[1] + a68 - a69*X[2];
        G[53] = a30*a65 - 36*a60*X[2] - 162*a61 + 144*a63 + a67 - a69*X[1] + a69;
        G[54] = a27*a70 + a64*a72 + a70 - 108*a71 - a72*X[0] + 90*a73 + 108*a74 + a75*X[2] + a78;
        G[55] = 252*a0*a64 + a27*a79 - 180*a71 + 126*a73 + 180*a74 + a79 - 108*a80 + a81*X[0] - a81 + a82;
        G[56] = a27*a72 + a72 + 252*a73 + 288*a74 - a75 + a78 - 72*a80 + a82;
        G[57] = a86;
        G[58] = a86;
        G[59] = a27*a87 + a30*a88 + a35*a87 + a83*a89 - 144*a83 + 360*a85 + a87 - a88*X[1] + a90*X[1] - a90;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = a0*X[0];
        Scalar const a2 = std::sqrt(10);
        Scalar const a3 = a2*X[0];
        Scalar const a4 = X[0]*X[0];
        Scalar const a5 = a2*a4;
        Scalar const a6 = std::sqrt(5);
        Scalar const a7 = 2*a6;
        Scalar const a8 = -a7;
        Scalar const a9 = 6*X[1];
        Scalar const a10 = a6*a9 + a8;
        Scalar const a11 = std::sqrt(15);
        Scalar const a12 = 2*a11;
        Scalar const a13 = a12*X[1];
        Scalar const a14 = -a12;
        Scalar const a15 = 4*X[2];
        Scalar const a16 = a11*a15 + a14;
        Scalar const a17 = std::sqrt(14);
        Scalar const a18 = a17*X[0];
        Scalar const a19 = a17*a4;
        Scalar const a20 = X[0]*X[0]*X[0];
        Scalar const a21 = a17*a20;
        Scalar const a22 = std::sqrt(7);
        Scalar const a23 = 4*a20;
        Scalar const a24 = a22*a9;
        Scalar const a25 = 18*X[1];
        Scalar const a26 = std::sqrt(42);
        Scalar const a27 = (1.0/3.0)*a20;
        Scalar const a28 = 4*a26;
        Scalar const a29 = 8*a26*X[1];
        Scalar const a30 = X[1]*X[1];
        Scalar const a31 = 10*a26*a30 + a26 - a29;
        Scalar const a32 = std::sqrt(21);
        Scalar const a33 = 2*a32;
        Scalar const a34 = -a33;
        Scalar const a35 = a33*X[1];
        Scalar const a36 = a15*a32;
        Scalar const a37 = 12*X[2];
        Scalar const a38 = 3*a17;
        Scalar const a39 = -a38;
        Scalar const a40 = 9*a17;
        Scalar const a41 = a17*X[2];
        Scalar const a42 = a41*X[1];
        Scalar const a43 = 6*a17;
        Scalar const a44 = a38 - a43*X[2];
        Scalar const a45 = a17*a25;
        Scalar const a46 = -a45;
        Scalar const a47 = a17*a30;
        Scalar const a48 = a46 + 15*a47;
        Scalar const a49 = std::sqrt(210);
        Scalar const a50 = a49*X[1];
        Scalar const a51 = 3*a49;
        Scalar const a52 = -a49 + a51*X[2];
        Scalar const a53 = 6*X[2];
        Scalar const a54 = 6*a49;
        Scalar const a55 = X[2]*X[2];
        Scalar const a56 = a49 + a54*a55 - a54*X[2];
        Scalar const a57 = 2*a50;
        Scalar const a58 = a30*a49 - a57;
        Scalar const a59 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a60 = 3*a59;
        Scalar const a61 = 27*a59;
        Scalar const a62 = a20*a59;
        Scalar const a63 = X[0]*X[0]*X[0]*X[0];
        Scalar const a64 = 42*a63;
        Scalar const a65 = a25 - 6;
        Scalar const a66 = 168*X[1];
        Scalar const a67 = 6*a0;
        Scalar const a68 = a0*X[1];
        Scalar const a69 = 3*a0;
        Scalar const a70 = a0*a30;
        Scalar const a71 = 30*a70;
        Scalar const a72 = a59*a63;
        Scalar const a73 = 6*a59;
        Scalar const a74 = -a73;
        Scalar const a75 = a59*X[1];
        Scalar const a76 = 9*a59;
        Scalar const a77 = 90*a59;
        Scalar const a78 = a77*X[1];
        Scalar const a79 = a30*a59;
        Scalar const a80 = 270*a79;
        Scalar const a81 = X[1]*X[1]*X[1];
        Scalar const a82 = 210*a59*a81 + a74 + a78 - a80;
        Scalar const a83 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a84 = 6*a83;
        Scalar const a85 = a84*X[1];
        Scalar const a86 = -a84;
        Scalar const a87 = a37*a83 + a86;
        Scalar const a88 = a83*X[1];
        Scalar const a89 = 84*a83;
        Scalar const a90 = a83*X[2];
        Scalar const a91 = a59*X[2];
        Scalar const a92 = 45*a59;
        Scalar const a93 = a30*a92;
        Scalar const a94 = 360*a75;
        Scalar const a95 = (3.0/2.0)*a63;
        Scalar const a96 = -a67;
        Scalar const a97 = a0*X[2];
        Scalar const a98 = 78*a68;
        Scalar const a99 = 12*a0;
        Scalar const a100 = a99*X[2];
        Scalar const a101 = 72*a68;
        Scalar const a102 = 144*X[2];
        Scalar const a103 = a100 + a96;
        Scalar const a104 = 126*a0*a81 - 198*a70 + a98;
        Scalar const a105 = std::sqrt(30);
        Scalar const a106 = 6*a105;
        Scalar const a107 = a105*X[1];
        Scalar const a108 = a105*X[2];
        Scalar const a109 = 3*a105;
        Scalar const a110 = a109*a30;
        Scalar const a111 = a105*a55;
        Scalar const a112 = 18*a111;
        Scalar const a113 = a105*a30;
        Scalar const a114 = 72*a108;
        Scalar const a115 = 6*a2;
        Scalar const a116 = -a115;
        Scalar const a117 = 12*a2;
        Scalar const a118 = a2*X[1];
        Scalar const a119 = 54*a118;
        Scalar const a120 = 36*a2;
        Scalar const a121 = a120*X[2];
        Scalar const a122 = a2*a30;
        Scalar const a123 = 18*a2;
        Scalar const a124 = 144*a118;
        Scalar const a125 = a116 - a120*a55 + a121;
        Scalar const a126 = a119 - 90*a122 + 42*a2*a81;
        Scalar const a127 = -a43;
        Scalar const a128 = a127 + 24*a41;
        Scalar const a129 = 72*a41;
        Scalar const a130 = a17*a55;
        Scalar const a131 = -a129 + 90*a130 + a40;
        Scalar const a132 = 72*X[2];
        Scalar const a133 = 180*a130;
        Scalar const a134 = X[2]*X[2]*X[2];
        Scalar const a135 = a127 + a129 - a133 + 120*a134*a17;
        Scalar const a136 = a43*a81 + a45 - 18*a47;
        Scalar const a137 = -a2 + 4*a3;
        Scalar const a138 = a7*X[0];
        Scalar const a139 = a12*X[0];
        Scalar const a140 = a17 - 10*a18 + 15*a19;
        Scalar const a141 = a22*X[0];
        Scalar const a142 = 12*a4;
        Scalar const a143 = -14*a141 + a142*a22 + 2*a22;
        Scalar const a144 = a26*a4 - 2*a26*X[0];
        Scalar const a145 = a32*X[0];
        Scalar const a146 = 24*X[2];
        Scalar const a147 = a142*a32 - 14*a145 + a33;
        Scalar const a148 = 6*a18;
        Scalar const a149 = -a148 + a38*a4;
        Scalar const a150 = a49*X[0];
        Scalar const a151 = -2*a150 + a4*a49;
        Scalar const a152 = a59*X[0];
        Scalar const a153 = a4*a59;
        Scalar const a154 = 168*a20;
        Scalar const a155 = 54*a152 - 189*a153 + a154*a59 - a60;
        Scalar const a156 = 252*a4;
        Scalar const a157 = 90*X[0];
        Scalar const a158 = a154 - a156 + a157;
        Scalar const a159 = a0*a4;
        Scalar const a160 = 24*a20;
        Scalar const a161 = a0*a160 + 30*a1 - 51*a159 - a69;
        Scalar const a162 = X[1]*X[1]*X[1]*X[1];
        Scalar const a163 = a157*a59;
        Scalar const a164 = 18*a152 - 18*a153 + a20*a73;
        Scalar const a165 = a154*a83 - a156*a83 + a157*a83;
        Scalar const a166 = a91*X[0];
        Scalar const a167 = 144*a4;
        Scalar const a168 = -153*a153 + a163 + 72*a62 - a76;
        Scalar const a169 = 39*a0;
        Scalar const a170 = 18*a1 - 18*a159 + a20*a67;
        Scalar const a171 = a105*X[0];
        Scalar const a172 = 9*a105;
        Scalar const a173 = a105*a4;
        Scalar const a174 = a105*a160 - a109 + 30*a171 - 51*a173;
        Scalar const a175 = a2*X[2];
        Scalar const a176 = 27*a2;
        Scalar const a177 = 36*a3;
        Scalar const a178 = a123*a4;
        Scalar const a179 = a115*a20 - a178 + 18*a3;
        Scalar const a180 = 18*a18;
        Scalar const a181 = 72*a18;
        Scalar const a182 = a180 - 18*a19 + 6*a21;
        Scalar const a183 = 36*X[1];
        Scalar const a184 = 12*a145;
        Scalar const a185 = a17*X[1];
        Scalar const a186 = X[0]*X[1];
        Scalar const a187 = a1*X[1];
        Scalar const a188 = a1*a30;
        Scalar const a189 = a75*X[0];
        Scalar const a190 = a4*a83;
        Scalar const a191 = a107*X[0];
        Scalar const a192 = a3*X[1];
        Scalar const a193 = 60*a17;
        Scalar const a194 = 36*a17;
        P[0] = a1;
        P[1] = -a3 + 2*a5;
        P[2] = a10*X[0] + a4*a6;
        P[3] = a11*a4 + (a13 + a16)*X[0];
        P[4] = a18 - 5*a19 + 5*a21;
        P[5] = a22*a23 + a4*(a22*a25 - 7*a22) + (2*a22 - a24)*X[0];
        P[6] = a26*a27 + a31*X[0] + a4*(-a26 + a28*X[1]);
        P[7] = a23*a32 + a4*(a32*a37 + a32*a9 - 7*a32) + (-a34 - a35 - a36)*X[0];
        P[8] = a21 + a4*(a38*X[2] + a39 + a40*X[1]) + (30*a42 + a44 + a48)*X[0];
        P[9] = a27*a49 + a4*(a50 + a52) + (a50*a53 + a56 + a58)*X[0];
        P[10] = a4*a61 + a59*a64 - a60*X[0] - 63*a62;
        P[11] = a20*(a66 - 84) + a4*(45 - 126*X[1]) + a64 + a65*X[0];
        P[12] = a20*(-17*a0 + 64*a68) + a4*(15*a0 - 108*a68 + 120*a70) + a63*a67 + (24*a0*X[1] - a69 - a71)*X[0];
        P[13] = a20*(a74 + 30*a75) + a4*(a76 - a78 + 135*a79) + (3.0/2.0)*a72 + a82*X[0];
        P[14] = a20*(56*a88 - a89 + 112*a90) + a4*(45*a83 - 42*a88 - a89*X[2]) + a64*a83 + (a85 + a87)*X[0];
        P[15] = a20*(-51*a59 + 144*a75 + 48*a91) + a4*(-243*a75 + 180*a79 - 81*a91 + a92 + a94*X[2]) + 18*a72 + (54*a59*X[1] + 18*a59*X[2] - a76 - a78*X[2] - a93)*X[0];
        P[16] = a0*a95 + a20*(26*a68 + a96 + 4*a97) + a4*(9*a0 - a100 + a101*X[2] + 99*a70 - a98) + (-a102*a68 + a103 + a104 + 252*a30*a97)*X[0];
        P[17] = a106*a63 + a20*(-17*a105 + 16*a107 + 48*a108) + a4*(15*a105 - 27*a107 - 81*a108 + 72*a111 + 12*a113 + a114*X[1]) + (6*a105*X[1] + 18*a105*X[2] - a108*a25 - a109 - a110 - a112)*X[0];
        P[18] = a2*a95 + a20*(a116 + a117*X[2] + a2*a25) + a4*(-a119 - a121 + 45*a122 + a123*a55 + a124*X[2] + 9*a2) + (252*a118*a55 - 288*a118*X[2] + 252*a122*X[2] + a125 + a126)*X[0];
        P[19] = a17*a95 + a20*(a128 + a43*X[1]) + a4*(a129*X[1] + a131 + a30*a40 + a46) + (a132*a47 + a133*X[1] + a135 + a136 - 144*a42)*X[0];
        P[20] = a68;
        P[21] = a137*X[1];
        P[22] = 3*a30*a6 + (a138 + a8)*X[1];
        P[23] = a11*a30 + (a139 + a16)*X[1];
        P[24] = a140*X[1];
        P[25] = a143*X[1] + a30*(18*a141 - 3*a22);
        P[26] = (10.0/3.0)*a26*a81 + a30*(a28*X[0] - a28) + (a144 + a26)*X[1];
        P[27] = a30*(6*a145 - a32) + (a145*a146 + a147 - a36)*X[1];
        P[28] = 5*a17*a81 + a30*(9*a18 - a40 + 15*a41) + (a148*X[2] + a149 + a44)*X[1];
        P[29] = a30*(a150 + a52) + (1.0/3.0)*a49*a81 + (a150*a53 + a151 + a56)*X[1];
        P[30] = a155*X[1];
        P[31] = a30*(a156 - 126*X[0] + 9) + (a158 - 6)*X[1];
        P[32] = a161*X[1] + a30*(-108*a1 + 96*a159 + a99) + a81*(-10*a0 + 80*a1);
        P[33] = (105.0/2.0)*a162*a59 + a30*(-a163 + a4*a92 + a92) + a81*(a163 - a77) + (a164 + a74)*X[1];
        P[34] = a30*(a4*a89 - 42*a83*X[0] + 3*a83) + (a165 + 336*a4*a90 + a87 - 168*a90*X[0])*X[1];
        P[35] = a30*(-243*a152 + 216*a153 + 360*a166 + a61 - a92*X[2]) + a81*(120*a152 - 15*a59) + (-162*a166 + a167*a91 + a168 + 18*a91)*X[1];
        P[36] = (63.0/2.0)*a0*a162 + a30*(a1*a132 - 78*a1 + a169*a4 + a169 - 72*a97) + a81*(-66*a0 + 66*a1 + 84*a97) + (-a1*a146 + a100*a4 + a103 + a170)*X[1];
        P[37] = a30*(a109 + a114*X[0] - 27*a171 - a172*X[2] + 24*a173) + a81*(-a105 + 8*a171) + (a108*a167 - 162*a108*X[0] + 18*a108 + 144*a111*X[0] - a112 + a174)*X[1];
        P[38] = (21.0/2.0)*a162*a2 + a30*(a102*a3 - 144*a175 + a176*a4 + a176 + 126*a2*a55 - 54*a3) + a81*(84*a175 - 30*a2 + 30*a3) + (a125 - a132*a3 + a177*a55 + a179 + 36*a5*X[2])*X[1];
        P[39] = (3.0/2.0)*a162*a17 + a30*(a131 - a180 + a181*X[2] + a4*a40) + a81*(a128 + a148) + (-a102*a18 + a132*a19 + a135 + 180*a18*a55 + a182)*X[1];
        P[40] = a97;
        P[41] = a137*X[2];
        P[42] = (a10 + a138)*X[2];
        P[43] = a12*a55 + (a13 + a139 + a14)*X[2];
        P[44] = a140*X[2];
        P[45] = (a141*a183 + a143 - a24)*X[2];
        P[46] = (a144 + a29*X[0] + a31)*X[2];
        P[47] = a55*(a184 + a34) + (a147 + a184*X[1] - a35)*X[2];
        P[48] = a55*(3*a18 + 15*a185 + a39) + (a149 + a18*a25 + a38 + a48)*X[2];
        P[49] = 2*a134*a49 + a55*(3*a150 + 3*a50 - a51) + (a151 + a49 + a57*X[0] + a58)*X[2];
        P[50] = a155*X[2];
        P[51] = (a158 - 252*a186 + 504*a4*X[1] + a65)*X[2];
        P[52] = (a161 - 216*a187 + 240*a188 + 192*a4*a68 + 24*a68 - a71)*X[2];
        P[53] = (a164 - 180*a189 + a4*a78 + a80*X[0] + a82)*X[2];
        P[54] = a55*(168*a190 + a84 - a89*X[0]) + (a165 - a186*a89 + a190*a66 + a85 + a86)*X[2];
        P[55] = a55*(-81*a152 + 72*a153 + a76 - a92*X[1] + a94*X[0]) + (a168 - 486*a189 + 432*a4*a75 + 54*a75 + 360*a79*X[0] - a93)*X[2];
        P[56] = a55*(-12*a1 - a101 + 72*a187 + a4*a67 + a67 + 126*a70) + (a104 + a170 - 156*a187 + 198*a188 + a4*a98 + a96)*X[2];
        P[57] = a134*(-a106 + 48*a171) + a55*(-81*a171 - a172*X[1] + a172 + 72*a173 + 72*a191) + (a106*X[1] + 48*a107*a4 - a110 + 24*a113*X[0] + a174 - 54*a191)*X[2];
        P[58] = a134*(-a117 + 84*a118 + 12*a3) + a55*(126*a122 + a123 - a124 - a177 + a178 + 144*a192) + (a116 + a126 + a179 - 108*a192 + 90*a3*a30 + 54*a5*X[1])*X[2];
        P[59] = a134*(60*a18 + a193*X[1] - a193) + 30*a17*X[2]*X[2]*X[2]*X[2] + a55*(a181*X[1] - a181 - 72*a185 + a194*a30 + a194*a4 + a194) + (a127 + a136 - a18*a183 + a180*a30 + a182 + a19*a25)*X[2];
        return P;
    }
};

template <>
class OrthonormalPolynomialBasis<3, 4>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 35;

    Vector<Size> eval(Vector<3> const& X) const 
    {
        Vector<Size> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = a1*X[0];
        Scalar const a3 = std::sqrt(5);
        Scalar const a4 = 2*a3;
        Scalar const a5 = 6*X[1];
        Scalar const a6 = std::sqrt(15);
        Scalar const a7 = 2*a6;
        Scalar const a8 = 4*X[2];
        Scalar const a9 = std::sqrt(14);
        Scalar const a10 = a9*X[0];
        Scalar const a11 = X[0]*X[0];
        Scalar const a12 = 15*a9;
        Scalar const a13 = std::sqrt(7);
        Scalar const a14 = 14*X[0];
        Scalar const a15 = 12*a11;
        Scalar const a16 = X[0]*X[1];
        Scalar const a17 = 36*a16;
        Scalar const a18 = std::sqrt(42);
        Scalar const a19 = 2*X[0];
        Scalar const a20 = 8*X[1];
        Scalar const a21 = a18*a20;
        Scalar const a22 = X[1]*X[1];
        Scalar const a23 = 10*a22;
        Scalar const a24 = std::sqrt(21);
        Scalar const a25 = 2*a24;
        Scalar const a26 = 24*X[2];
        Scalar const a27 = 3*a9;
        Scalar const a28 = 6*a9;
        Scalar const a29 = a28*X[0];
        Scalar const a30 = 18*X[1];
        Scalar const a31 = a30*a9;
        Scalar const a32 = a9*X[2];
        Scalar const a33 = a32*X[1];
        Scalar const a34 = std::sqrt(210);
        Scalar const a35 = a19*a34;
        Scalar const a36 = 2*X[1];
        Scalar const a37 = 6*a34;
        Scalar const a38 = a37*X[2];
        Scalar const a39 = X[2]*X[2];
        Scalar const a40 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a41 = a40*X[0];
        Scalar const a42 = a11*a40;
        Scalar const a43 = X[0]*X[0]*X[0];
        Scalar const a44 = 168*a43;
        Scalar const a45 = 90*X[0];
        Scalar const a46 = 252*a11;
        Scalar const a47 = a11*X[1];
        Scalar const a48 = 504*a47;
        Scalar const a49 = 30*a0;
        Scalar const a50 = 24*a0;
        Scalar const a51 = a0*a11;
        Scalar const a52 = a0*a16;
        Scalar const a53 = a0*a22;
        Scalar const a54 = 240*X[0];
        Scalar const a55 = a0*a47;
        Scalar const a56 = 6*a40;
        Scalar const a57 = a40*X[1];
        Scalar const a58 = 90*a57;
        Scalar const a59 = a22*a40;
        Scalar const a60 = X[1]*X[1]*X[1];
        Scalar const a61 = a41*X[1];
        Scalar const a62 = a22*a41;
        Scalar const a63 = a42*X[1];
        Scalar const a64 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a65 = 6*a64;
        Scalar const a66 = 12*X[2];
        Scalar const a67 = 84*a16;
        Scalar const a68 = 168*a64;
        Scalar const a69 = X[0]*X[2];
        Scalar const a70 = a11*X[2];
        Scalar const a71 = 18*X[2];
        Scalar const a72 = 162*X[2];
        Scalar const a73 = 144*X[2];
        Scalar const a74 = 6*a0;
        Scalar const a75 = 18*X[0];
        Scalar const a76 = a0*X[1];
        Scalar const a77 = 198*a53;
        Scalar const a78 = 252*X[2];
        Scalar const a79 = std::sqrt(30);
        Scalar const a80 = 3*a79;
        Scalar const a81 = a79*X[0];
        Scalar const a82 = a11*a79;
        Scalar const a83 = 24*a79;
        Scalar const a84 = a39*a79;
        Scalar const a85 = a16*a79;
        Scalar const a86 = a22*X[0];
        Scalar const a87 = 144*X[0];
        Scalar const a88 = 6*a1;
        Scalar const a89 = a1*X[1];
        Scalar const a90 = 36*a1;
        Scalar const a91 = a90*X[2];
        Scalar const a92 = 18*a11;
        Scalar const a93 = 90*a22;
        Scalar const a94 = 42*a60;
        Scalar const a95 = a2*X[1];
        Scalar const a96 = 288*X[2];
        Scalar const a97 = a1*a22;
        Scalar const a98 = 72*a32;
        Scalar const a99 = 18*a22;
        Scalar const a100 = 180*a39;
        Scalar const a101 = a100*a9;
        Scalar const a102 = X[2]*X[2]*X[2];
        Scalar const a103 = a10*a73;
        Scalar const a104 = std::sqrt(22);
        Scalar const a105 = a104*X[0];
        Scalar const a106 = a104*a11;
        Scalar const a107 = a104*a43;
        Scalar const a108 = X[0]*X[0]*X[0]*X[0];
        Scalar const a109 = a104*a108;
        Scalar const a110 = std::sqrt(11);
        Scalar const a111 = 50*X[0];
        Scalar const a112 = 264*a11;
        Scalar const a113 = 456*a43;
        Scalar const a114 = 240*a108;
        Scalar const a115 = a43*X[1];
        Scalar const a116 = std::sqrt(66);
        Scalar const a117 = a116*X[0];
        Scalar const a118 = a11*a116;
        Scalar const a119 = a116*a43;
        Scalar const a120 = a108*a116;
        Scalar const a121 = a116*a16;
        Scalar const a122 = a117*a22;
        Scalar const a123 = a119*X[1];
        Scalar const a124 = a118*a22;
        Scalar const a125 = a104*X[1];
        Scalar const a126 = 30*a125;
        Scalar const a127 = a104*a60;
        Scalar const a128 = a104*a16;
        Scalar const a129 = a105*a22;
        Scalar const a130 = a105*a60;
        Scalar const a131 = a104*a47;
        Scalar const a132 = a107*X[1];
        Scalar const a133 = a106*a22;
        Scalar const a134 = std::sqrt(110);
        Scalar const a135 = 4*a134;
        Scalar const a136 = a134*X[1];
        Scalar const a137 = 24*a136;
        Scalar const a138 = a108*a134;
        Scalar const a139 = 6*a11;
        Scalar const a140 = 126*a134;
        Scalar const a141 = a140*a22;
        Scalar const a142 = a134*a60;
        Scalar const a143 = 224*a142;
        Scalar const a144 = X[1]*X[1]*X[1]*X[1];
        Scalar const a145 = 72*a134;
        Scalar const a146 = a134*a86;
        Scalar const a147 = std::sqrt(33);
        Scalar const a148 = 2*a147;
        Scalar const a149 = a43*X[2];
        Scalar const a150 = 60*a105;
        Scalar const a151 = 6*X[2];
        Scalar const a152 = a104*a22;
        Scalar const a153 = a106*X[2];
        Scalar const a154 = a107*X[2];
        Scalar const a155 = a128*X[2];
        Scalar const a156 = a116*X[1];
        Scalar const a157 = a116*a22;
        Scalar const a158 = 48*X[2];
        Scalar const a159 = a116*a47;
        Scalar const a160 = 84*X[2];
        Scalar const a161 = 528*X[2];
        Scalar const a162 = 480*X[2];
        Scalar const a163 = std::sqrt(330);
        Scalar const a164 = 4*a163;
        Scalar const a165 = a163*X[1];
        Scalar const a166 = 22*a165;
        Scalar const a167 = a163*X[2];
        Scalar const a168 = 2*a167;
        Scalar const a169 = a108*a163;
        Scalar const a170 = a163*a22;
        Scalar const a171 = 105*a170;
        Scalar const a172 = 168*a163*a60;
        Scalar const a173 = 66*a163;
        Scalar const a174 = a163*X[0];
        Scalar const a175 = 168*a170;
        Scalar const a176 = 108*a163;
        Scalar const a177 = a163*a39;
        Scalar const a178 = 126*a47;
        Scalar const a179 = 270*a167;
        Scalar const a180 = 26*X[0];
        Scalar const a181 = 66*a11;
        Scalar const a182 = 62*a43;
        Scalar const a183 = 30*a22;
        Scalar const a184 = 12*a134;
        Scalar const a185 = a134*X[2];
        Scalar const a186 = a136*X[2];
        Scalar const a187 = a134*a39;
        Scalar const a188 = a187*X[0];
        Scalar const a189 = a134*a47;
        Scalar const a190 = a136*a39;
        Scalar const a191 = a185*a22;
        Scalar const a192 = a16*a185;
        Scalar const a193 = a16*a187;
        Scalar const a194 = 5*a104;
        Scalar const a195 = 30*a104;
        Scalar const a196 = 30*a106;
        Scalar const a197 = 510*X[2];
        Scalar const a198 = 480*a39;
        Scalar const a199 = 1560*X[2];
        Scalar const a200 = std::sqrt(154);
        Scalar const a201 = 2*a200;
        Scalar const a202 = a200*a26;
        Scalar const a203 = 20*a200;
        Scalar const a204 = a200*a22;
        Scalar const a205 = a200*a39;
        Scalar const a206 = 60*a205;
        Scalar const a207 = a102*a200;
        Scalar const a208 = a16*a200;
        Scalar const a209 = 600*a205;
        Scalar const a210 = std::sqrt(770);
        Scalar const a211 = 4*a210;
        Scalar const a212 = a210*X[1];
        Scalar const a213 = 12*a212;
        Scalar const a214 = a210*a66;
        Scalar const a215 = a183*a210;
        Scalar const a216 = a210*a60;
        Scalar const a217 = 28*a216;
        Scalar const a218 = a210*a39;
        Scalar const a219 = 30*a218;
        Scalar const a220 = a102*a210;
        Scalar const a221 = 20*a220;
        Scalar const a222 = 36*a210;
        Scalar const a223 = a210*X[2];
        Scalar const a224 = 228*a22;
        Scalar const a225 = 3*a134;
        Scalar const a226 = 12*a136;
        Scalar const a227 = 60*a185;
        Scalar const a228 = a134*a92;
        Scalar const a229 = 12*a142;
        Scalar const a230 = 270*a187;
        Scalar const a231 = 420*a102;
        Scalar const a232 = a134*a231;
        Scalar const a233 = 180*a185;
        P[0] = a0;
        P[1] = -a1 + 4*a2;
        P[2] = a3*a5 + a4*X[0] - a4;
        P[3] = a6*a8 + a7*X[0] + a7*X[1] - a7;
        P[4] = -10*a10 + a11*a12 + a9;
        P[5] = -a13*a14 + a13*a15 + a13*a17 - a13*a5 + 2*a13;
        P[6] = a11*a18 - a18*a19 + a18*a23 + a18 + a21*X[0] - a21;
        P[7] = -a14*a24 + a15*a24 + 12*a16*a24 + a24*a26*X[0] - a24*a8 - a25*X[1] + a25;
        P[8] = a10*a30 + a11*a27 + a12*a22 + a27 - a28*X[2] + a29*X[2] - a29 - a31 + 30*a33;
        P[9] = a11*a34 + a22*a34 - a34*a36 + a34*a5*X[2] + a34 + a35*X[1] - a35 + a37*a39 + a38*X[0] - a38;
        P[10] = a40*a44 - 3*a40 + 54*a41 - 189*a42;
        P[11] = -252*a16 + a30 + a44 + a45 - a46 + a48 - 6;
        P[12] = -3*a0 - a22*a49 + a43*a50 + a49*X[0] + a50*X[1] - 51*a51 - 216*a52 + a53*a54 + 192*a55;
        P[13] = 210*a40*a60 + 18*a41 - 18*a42 + a43*a56 - a56 + a58 - 270*a59 - 180*a61 + 270*a62 + 90*a63;
        P[14] = a44*a64 + a45*a64 - a46*a64 + a47*a68 + a64*a66 - a64*a67 + 336*a64*a70 + a65*X[1] - a65 - a68*a69;
        P[15] = 72*a40*a43 + a40*a45 + a40*a71 - 9*a40 - a41*a72 + a42*a73 - 153*a42 + 54*a57 - a58*X[2] - 45*a59 + 720*a61*X[2] - 486*a61 + 360*a62 + 432*a63;
        P[16] = a0*a15*X[2] + 126*a0*a60 + a0*a66 + a0*a75 + a43*a74 - a50*a69 - 18*a51 + a52*a73 - 156*a52 + a53*a78 + 78*a55 - a73*a76 - a74 + 78*a76 + a77*X[0] - a77;
        P[17] = -a22*a80 - a30*a79*X[2] + a43*a83 + 48*a47*a79 + a5*a79 + a71*a79 - a72*a81 + a73*a82 + a73*a85 - a80 + 30*a81 - 51*a82 + a83*a86 + a84*a87 - 18*a84 - 54*a85;
        P[18] = 54*a1*a47 - a1*a92 - a1*a93 + a1*a94 + a11*a91 + 36*a2*a39 - 72*a2*X[2] + 18*a2 + 252*a39*a89 - a39*a90 + a43*a88 + a45*a97 + a78*a97 - a88 - a89*a96 + 54*a89 + a91 + a95*a96 - 108*a95;
        P[19] = a10*a100 + a10*a99 - 36*a10*X[1] + 18*a10 + a101*X[1] - a101 + 120*a102*a9 + a103*X[1] - a103 + a11*a31 + a11*a98 + a22*a98 + a28*a43 + a28*a60 - a28 + a31 - 144*a33 - a9*a92 - a9*a99 + a98;
        P[20] = a104 - 28*a105 + 168*a106 - 336*a107 + 210*a109;
        P[21] = -a110*a111 + a110*a112 - a110*a113 + a110*a114 + 720*a110*a115 + 144*a110*a16 - 648*a110*a47 - a110*a5 + 2*a110;
        P[22] = -a116*a20 + a116*a23 - a116*a48 + a116 - 20*a117 + 82*a118 - 108*a119 + 45*a120 + 152*a121 - 180*a122 + 360*a123 + 450*a124;
        P[23] = a104*a93 + 2*a104 - 26*a105 + 66*a106 - 62*a107 + 20*a109 - a126 - 70*a127 + 360*a128 - 990*a129 + 700*a130 - 630*a131 + 300*a132 + 900*a133;
        P[24] = a11*a141 + a134*a139 + a134 - a135*a43 - a135*X[0] + a137*a43 - a137 + a138 + a140*a144 + a141 + a143*X[0] - a143 + a145*a16 - a145*a47 - 252*a146;
        P[25] = -a111*a147 + a112*a147 - a113*a147 + a114*a147 + 240*a115*a147 + 480*a147*a149 + 48*a147*a16 - 216*a147*a47 + 96*a147*a69 - 432*a147*a70 - a147*a8 - a148*X[1] + a148;
        P[26] = -a104*a151 - a104*a30 + 3*a104 + 114*a105*X[2] + 246*a106 - 324*a107 + 135*a109 + a126*X[2] + 342*a128 - 270*a129 + 1350*a131*X[2] - 1134*a131 + 810*a132 + 675*a133 - a150 + 15*a152 - 378*a153 + 270*a154 - 540*a155;
        P[27] = -a116*a8 - a116*a94 + 2*a116 + a117*a158 + 420*a117*a60 - 26*a117 - a118*a160 + 66*a118 + 40*a119*X[2] - 62*a119 + 20*a120 - a121*a161 + 312*a121 + 840*a122*X[2] - 726*a122 + 260*a123 + 660*a124 + a156*a158 - 26*a156 - a157*a160 + 66*a157 + a159*a162 - 546*a159;
        P[28] = a11*a171 + a139*a163 - a139*a167 + 84*a144*a163 + a151*a174 + a16*a173 + a163 - a164*a43 - a164*X[0] + 42*a165*X[2] + a166*a43 - a166 + 42*a167*a47 - a167*a67 + a168*a43 - a168 + a169 - 210*a170*X[0] + a171 + a172*X[0] + a172*X[2] - a172 - a173*a47 + a175*a69 - a175*X[2];
        P[29] = 82*a11*a163 - 378*a11*a167 + 45*a11*a170 + 270*a11*a177 - a151*a163 + 38*a16*a163 - 108*a16*a167 - a163*a178 - a163*a36 + a163 + 90*a165*a43 + a167*a5 + 114*a167*X[0] + 45*a169 - a170*a75 + a170 - 20*a174 - a176*a39*X[0] - a176*a43 + 6*a177 + a179*a43 + a179*a47;
        P[30] = 300*a11*a134*a22 + 120*a11*a187 + 216*a134*a16 - a134*a180 + a134*a181 - a134*a182 + a134*a183 - a134*a30 - a134*a66 + 2*a134 + 180*a136*a43 + 20*a138 + 140*a142*X[0] - 14*a142 - 330*a146 + a184*a39 + 120*a185*a43 - a185*a46 + 960*a185*a47 + 840*a185*a86 + a185*a87 + 96*a186 - 132*a188 - 378*a189 - 84*a190 - 84*a191 - 1056*a192 + 840*a193;
        P[31] = 180*a104*a144 + a104*a45*X[2] - 20*a105 - 20*a107 + a108*a194 + a125*a197 - a125*a198 - 90*a125 + 1080*a127*X[2] - 440*a127 + a128*a198 + 270*a128 + a129*a199 - 690*a129 + 440*a130 + a131*a197 - 270*a131 + 90*a132 + 345*a133 - a150*a39 - a152*a199 + 1080*a152*a39 + 345*a152 - 90*a153 + 30*a154 - 1020*a155 + a194 + a195*a39 - a195*X[2] + a196*a39 + a196;
        P[32] = a108*a203 + 60*a11*a204 + a11*a209 + 60*a115*a200 + 240*a149*a200 + a158*a200*X[1] + a16*a209 - a161*a208 + a162*a200*a47 - a178*a200 - a180*a200 + a181*a200 - a182*a200 - a200*a5 + 288*a200*a69 - 504*a200*a70 - a201*a60 + a201 - a202*a22 - a202 + a203*a60*X[0] + a204*a54*X[2] - 66*a204*X[0] + 6*a204 - 660*a205*X[0] - a206*X[1] + a206 + 400*a207*X[0] - 40*a207 + 72*a208;
        P[33] = a108*a210 + a11*a215 + a11*a219 + a139*a210 + 9*a144*a210 + 300*a16*a218 - 264*a16*a223 + a17*a210 + a210*a224*a69 - 60*a210*a86 + a210 - a211*a43 - a211*X[0] - 300*a212*a39 + 132*a212*X[2] + a213*a43 - a213 + a214*a43 - a214 + a215 + 108*a216*X[2] + a217*X[0] - a217 + 270*a218*a22 - 60*a218*X[0] + a219 + 180*a220*X[1] + a221*X[0] - a221 - a222*a47 + a222*a69 - a222*a70 - a223*a224 + 132*a223*a47;
        P[34] = a11*a230 - a11*a233 + a134*a17 + a134*a99 + 210*a134*X[2]*X[2]*X[2]*X[2] + a136*a231 + 3*a138 + 60*a142*X[2] + a144*a225 - 36*a146 - a184*a43 - a184*X[0] + 180*a186 - 540*a188 - 36*a189 - 540*a190 - 180*a191 - 360*a192 + 540*a193 + a22*a228 + a22*a230 + a225 + a226*a43 - a226 + a227*a43 - a227 + a228 + a229*X[0] - a229 + a230 + a232*X[0] - a232 + a233*a47 + a233*a86 + a233*X[0];
        return P;
    }
               
    Matrix<Dims, Size> derivatives(Vector<3> const& X) const
    {
        Matrix<Dims, Size> G;
        Scalar const a0 = std::sqrt(10);
        Scalar const a1 = std::sqrt(5);
        Scalar const a2 = std::sqrt(15);
        Scalar const a3 = 2*a2;
        Scalar const a4 = std::sqrt(14);
        Scalar const a5 = 30*a4;
        Scalar const a6 = std::sqrt(7);
        Scalar const a7 = a6*X[0];
        Scalar const a8 = 36*X[1];
        Scalar const a9 = std::sqrt(42);
        Scalar const a10 = 2*a9;
        Scalar const a11 = 8*a9;
        Scalar const a12 = std::sqrt(21);
        Scalar const a13 = 24*a12;
        Scalar const a14 = a13*X[0];
        Scalar const a15 = 12*a12;
        Scalar const a16 = 6*a4;
        Scalar const a17 = 18*a4;
        Scalar const a18 = a16*X[0] - a16;
        Scalar const a19 = a5*X[1];
        Scalar const a20 = std::sqrt(210);
        Scalar const a21 = 2*a20;
        Scalar const a22 = 6*a20;
        Scalar const a23 = a21*X[0] + a21*X[1] - a21 + a22*X[2];
        Scalar const a24 = 12*X[2];
        Scalar const a25 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a26 = 54*a25;
        Scalar const a27 = a25*X[0];
        Scalar const a28 = X[0]*X[0];
        Scalar const a29 = 504*a28;
        Scalar const a30 = 504*X[0];
        Scalar const a31 = 252*X[1];
        Scalar const a32 = X[0]*X[1];
        Scalar const a33 = 1008*a32;
        Scalar const a34 = 252*X[0];
        Scalar const a35 = std::sqrt(6);
        Scalar const a36 = a35*X[0];
        Scalar const a37 = 216*a35;
        Scalar const a38 = a28*a35;
        Scalar const a39 = X[1]*X[1];
        Scalar const a40 = a35*a39;
        Scalar const a41 = a32*a35;
        Scalar const a42 = 24*a35;
        Scalar const a43 = a35*X[1];
        Scalar const a44 = 18*a25;
        Scalar const a45 = a25*X[1];
        Scalar const a46 = a25*a39;
        Scalar const a47 = 180*a27;
        Scalar const a48 = 90*a25;
        Scalar const a49 = a27*X[1];
        Scalar const a50 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a51 = 84*a50;
        Scalar const a52 = 168*a50;
        Scalar const a53 = 336*a50;
        Scalar const a54 = X[0]*X[2];
        Scalar const a55 = 162*X[2];
        Scalar const a56 = a25*a28;
        Scalar const a57 = a27*X[2];
        Scalar const a58 = 720*X[2];
        Scalar const a59 = -a48*X[1] + 720*a49;
        Scalar const a60 = 18*a35;
        Scalar const a61 = a42*X[2];
        Scalar const a62 = 144*X[2];
        Scalar const a63 = 78*a35;
        Scalar const a64 = 504*X[2];
        Scalar const a65 = 12*a35;
        Scalar const a66 = std::sqrt(30);
        Scalar const a67 = a66*X[0];
        Scalar const a68 = a66*X[1];
        Scalar const a69 = a28*a66;
        Scalar const a70 = X[2]*X[2];
        Scalar const a71 = 144*a66;
        Scalar const a72 = a32*a66;
        Scalar const a73 = 288*a67*X[2];
        Scalar const a74 = 6*a66;
        Scalar const a75 = 18*a66;
        Scalar const a76 = 36*X[2];
        Scalar const a77 = 18*a0;
        Scalar const a78 = a0*X[1];
        Scalar const a79 = 36*a0;
        Scalar const a80 = a0*a39;
        Scalar const a81 = a0*a32;
        Scalar const a82 = 288*a78;
        Scalar const a83 = a0*X[2];
        Scalar const a84 = 72*a83;
        Scalar const a85 = a84*X[0] - a84;
        Scalar const a86 = 54*a0;
        Scalar const a87 = a0*X[0];
        Scalar const a88 = 288*a83;
        Scalar const a89 = a64*a78;
        Scalar const a90 = a4*X[0];
        Scalar const a91 = a4*a62;
        Scalar const a92 = a4*a70;
        Scalar const a93 = a17*a28 + a17*a39 + a17 - a4*a8 + a8*a90 - 36*a90 + a91*X[0] + a91*X[1] - a91 + 180*a92;
        Scalar const a94 = 72*a4;
        Scalar const a95 = 144*X[1];
        Scalar const a96 = 360*X[2];
        Scalar const a97 = a4*a96;
        Scalar const a98 = 144*a32;
        Scalar const a99 = std::sqrt(22);
        Scalar const a100 = a99*X[0];
        Scalar const a101 = a28*a99;
        Scalar const a102 = X[0]*X[0]*X[0];
        Scalar const a103 = a102*a99;
        Scalar const a104 = std::sqrt(11);
        Scalar const a105 = a104*X[0];
        Scalar const a106 = a104*a28;
        Scalar const a107 = a102*a104;
        Scalar const a108 = std::sqrt(66);
        Scalar const a109 = 20*a108;
        Scalar const a110 = a108*X[0];
        Scalar const a111 = 152*a108;
        Scalar const a112 = a108*a28;
        Scalar const a113 = 180*a108;
        Scalar const a114 = a110*a39;
        Scalar const a115 = a112*X[1];
        Scalar const a116 = a102*a108;
        Scalar const a117 = a108*a32;
        Scalar const a118 = a99*X[1];
        Scalar const a119 = a39*a99;
        Scalar const a120 = X[1]*X[1]*X[1];
        Scalar const a121 = a120*a99;
        Scalar const a122 = a32*a99;
        Scalar const a123 = a100*a39;
        Scalar const a124 = a101*X[1];
        Scalar const a125 = 30*a99;
        Scalar const a126 = -a125;
        Scalar const a127 = std::sqrt(110);
        Scalar const a128 = 4*a127;
        Scalar const a129 = 12*a127;
        Scalar const a130 = 72*a127;
        Scalar const a131 = a130*X[1];
        Scalar const a132 = a127*a39;
        Scalar const a133 = a120*a127;
        Scalar const a134 = 24*a127;
        Scalar const a135 = a127*a31;
        Scalar const a136 = 672*a132;
        Scalar const a137 = a127*X[1];
        Scalar const a138 = std::sqrt(33);
        Scalar const a139 = a138*X[0];
        Scalar const a140 = 48*a138;
        Scalar const a141 = 96*a138;
        Scalar const a142 = a138*a28;
        Scalar const a143 = a102*a138;
        Scalar const a144 = 60*a99;
        Scalar const a145 = 114*X[2];
        Scalar const a146 = 756*X[2];
        Scalar const a147 = 540*X[2];
        Scalar const a148 = a101*X[2];
        Scalar const a149 = a122*X[2];
        Scalar const a150 = -540*a122;
        Scalar const a151 = 1350*a124 + a125*X[1] + a150;
        Scalar const a152 = a108*a39;
        Scalar const a153 = a108*X[1];
        Scalar const a154 = 528*X[2];
        Scalar const a155 = 168*X[2];
        Scalar const a156 = a112*X[2];
        Scalar const a157 = 840*X[2];
        Scalar const a158 = a117*X[2];
        Scalar const a159 = 48*a108;
        Scalar const a160 = -26*a108 + a159*X[2];
        Scalar const a161 = std::sqrt(330);
        Scalar const a162 = 4*a161;
        Scalar const a163 = 12*a161;
        Scalar const a164 = 66*a161;
        Scalar const a165 = a164*X[1];
        Scalar const a166 = 6*a161;
        Scalar const a167 = a166*X[2];
        Scalar const a168 = a161*a39;
        Scalar const a169 = 210*a168;
        Scalar const a170 = a120*a161;
        Scalar const a171 = 168*a170;
        Scalar const a172 = a161*a32;
        Scalar const a173 = a161*X[0];
        Scalar const a174 = a161*X[1];
        Scalar const a175 = a174*X[2];
        Scalar const a176 = 168*a168;
        Scalar const a177 = 84*a172;
        Scalar const a178 = 22*a161;
        Scalar const a179 = 210*a174;
        Scalar const a180 = 42*a161;
        Scalar const a181 = a180*X[2];
        Scalar const a182 = 504*a168;
        Scalar const a183 = a173*X[2];
        Scalar const a184 = 2*a161;
        Scalar const a185 = -a184;
        Scalar const a186 = a180*X[1];
        Scalar const a187 = a161*a28;
        Scalar const a188 = a102*a161;
        Scalar const a189 = a161*a70;
        Scalar const a190 = a174*a28;
        Scalar const a191 = 270*a190;
        Scalar const a192 = a187*X[2];
        Scalar const a193 = 132*a127;
        Scalar const a194 = 216*a127;
        Scalar const a195 = a127*a28;
        Scalar const a196 = a102*a127;
        Scalar const a197 = a127*a32;
        Scalar const a198 = 1056*X[2];
        Scalar const a199 = a132*X[0];
        Scalar const a200 = a127*a70;
        Scalar const a201 = a200*X[0];
        Scalar const a202 = a195*X[1];
        Scalar const a203 = a197*X[2];
        Scalar const a204 = 60*a127;
        Scalar const a205 = 96*a127;
        Scalar const a206 = a127*X[0];
        Scalar const a207 = a195*X[2];
        Scalar const a208 = -a137*a155 + 1680*a203;
        Scalar const a209 = -a129;
        Scalar const a210 = 264*X[2];
        Scalar const a211 = 20*a99;
        Scalar const a212 = 90*a99;
        Scalar const a213 = a212*X[2];
        Scalar const a214 = a144*a28;
        Scalar const a215 = a144*a70;
        Scalar const a216 = a100*X[2];
        Scalar const a217 = a118*X[2];
        Scalar const a218 = 270*a101;
        Scalar const a219 = 480*a70;
        Scalar const a220 = 1560*a119;
        Scalar const a221 = 1020*a122;
        Scalar const a222 = 510*X[2];
        Scalar const a223 = a119*X[2];
        Scalar const a224 = std::sqrt(154);
        Scalar const a225 = a224*X[0];
        Scalar const a226 = a224*X[1];
        Scalar const a227 = a224*X[2];
        Scalar const a228 = a224*a28;
        Scalar const a229 = a102*a224;
        Scalar const a230 = a224*a39;
        Scalar const a231 = a224*a70;
        Scalar const a232 = X[2]*X[2]*X[2];
        Scalar const a233 = a225*X[2];
        Scalar const a234 = a230*X[0];
        Scalar const a235 = 1200*a231*X[0];
        Scalar const a236 = a228*X[1];
        Scalar const a237 = 600*a231;
        Scalar const a238 = a227*a32;
        Scalar const a239 = 6*a224;
        Scalar const a240 = a224*a32;
        Scalar const a241 = 48*a226;
        Scalar const a242 = a228*X[2];
        Scalar const a243 = 24*a224;
        Scalar const a244 = std::sqrt(770);
        Scalar const a245 = 4*a244;
        Scalar const a246 = 60*a244;
        Scalar const a247 = a246*a39;
        Scalar const a248 = 12*a244;
        Scalar const a249 = a232*a244;
        Scalar const a250 = a120*a244;
        Scalar const a251 = a244*a8;
        Scalar const a252 = a244*a76;
        Scalar const a253 = a244*X[1];
        Scalar const a254 = 72*a244;
        Scalar const a255 = a244*a39;
        Scalar const a256 = 228*a255;
        Scalar const a257 = a244*a70;
        Scalar const a258 = 300*a257;
        Scalar const a259 = a244*a32;
        Scalar const a260 = a246*a70;
        Scalar const a261 = a260*X[0] - a260;
        Scalar const a262 = 84*a255;
        Scalar const a263 = a246*X[1];
        Scalar const a264 = 132*a244;
        Scalar const a265 = a264*X[2];
        Scalar const a266 = a253*X[2];
        Scalar const a267 = 264*a244;
        Scalar const a268 = 120*a244;
        Scalar const a269 = a259*X[2];
        Scalar const a270 = 36*a244;
        Scalar const a271 = a102*a248 - a248 + 540*a257*X[1] - a270*a28 + a270*X[0];
        Scalar const a272 = a246*X[2];
        Scalar const a273 = a264*X[1];
        Scalar const a274 = 180*a127;
        Scalar const a275 = 36*a132;
        Scalar const a276 = 540*a200;
        Scalar const a277 = a127*a232;
        Scalar const a278 = 180*a195;
        Scalar const a279 = 180*a132;
        Scalar const a280 = a102*a129 + a120*a129 + a127*a8 - a130*a32 - a137*a96 + a195*a8 - 36*a195 + a197*a96 - a206*a96 + 36*a206 + a209 + a274*X[2] + a275*X[0] - a275 + a276*X[0] + a276*X[1] - a276 + 420*a277 + a278*X[2] + a279*X[2];
        Scalar const a281 = 1260*a200;
        Scalar const a282 = 1080*X[2];
        G[0] = 0;
        G[1] = 0;
        G[2] = 0;
        G[3] = 4*a0;
        G[4] = 0;
        G[5] = 0;
        G[6] = 2*a1;
        G[7] = 6*a1;
        G[8] = 0;
        G[9] = a3;
        G[10] = a3;
        G[11] = 4*a2;
        G[12] = -10*a4 + a5*X[0];
        G[13] = 0;
        G[14] = 0;
        G[15] = a6*a8 - 14*a6 + 24*a7;
        G[16] = -6*a6 + 36*a7;
        G[17] = 0;
        G[18] = a10*X[0] - a10 + a11*X[1];
        G[19] = a11*X[0] - a11 + 20*a9*X[1];
        G[20] = 0;
        G[21] = -14*a12 + a13*X[2] + a14 + a15*X[1];
        G[22] = -2*a12 + a15*X[0];
        G[23] = -4*a12 + a14;
        G[24] = a16*X[2] + a17*X[1] + a18;
        G[25] = a17*X[0] - a17 + a19 + a5*X[2];
        G[26] = a18 + a19;
        G[27] = a23;
        G[28] = a23;
        G[29] = a20*a24 + a22*X[0] + a22*X[1] - a22;
        G[30] = a25*a29 + a26 - 378*a27;
        G[31] = 0;
        G[32] = 0;
        G[33] = a29 - a30 - a31 + a33 + 90;
        G[34] = a29 - a34 + 18;
        G[35] = 0;
        G[36] = 30*a35 - 102*a36 - a37*X[1] + 72*a38 + 240*a40 + 384*a41;
        G[37] = -a37*X[0] + 192*a38 + 480*a41 + a42 - 60*a43;
        G[38] = 0;
        G[39] = -36*a27 + a28*a44 + a44 - 180*a45 + 270*a46 + a47*X[1];
        G[40] = a28*a48 - 540*a45 + 630*a46 - a47 + a48 + 540*a49;
        G[41] = 0;
        G[42] = a29*a50 - a30*a50 + a32*a53 + 672*a50*a54 + 90*a50 - a51*X[1] - a52*X[2];
        G[43] = a28*a52 + 6*a50 - a51*X[0];
        G[44] = a28*a53 + 12*a50 - a52*X[0];
        G[45] = -a25*a55 - 306*a27 + a45*a58 - 486*a45 + 360*a46 + a48 + 864*a49 + 216*a56 + 288*a57;
        G[46] = a26 - 486*a27 - a48*X[2] + 432*a56 + 720*a57 + a59;
        G[47] = -162*a27 + a44 + 144*a56 + a59;
        G[48] = a28*a60 - 36*a36 + 198*a40 + 156*a41 + a43*a62 - 156*a43 + a60 + a61*X[0] - a61;
        G[49] = a28*a63 - a35*a62 + a36*a62 - 156*a36 + 378*a40 + 396*a41 + a43*a64 - 396*a43 + a63;
        G[50] = a28*a65 + 252*a40 + 144*a41 - a42*X[0] - 144*a43 + a65;
        G[51] = 24*a39*a66 - a55*a66 + a62*a68 + 30*a66 - 102*a67 - 54*a68 + 72*a69 + a70*a71 + 96*a72 + a73;
        G[52] = a62*a67 - 54*a67 + 48*a69 + 48*a72 - a74*X[1] + a74 - a75*X[2];
        G[53] = a32*a71 - a66*a76 - 162*a67 + 144*a69 + a73 - a75*X[1] + a75;
        G[54] = a28*a77 + a70*a79 + a77 - 108*a78 - a79*X[0] + 90*a80 + 108*a81 + a82*X[2] + a85;
        G[55] = 252*a0*a70 + a28*a86 - 180*a78 + 126*a80 + 180*a81 + a86 - 108*a87 + a88*X[0] - a88 + a89;
        G[56] = a28*a79 + a79 + 252*a80 + 288*a81 - a82 + a85 - 72*a87 + a89;
        G[57] = a93;
        G[58] = a93;
        G[59] = a28*a94 + a39*a94 - a4*a95 + a4*a98 + a90*a96 - 144*a90 + 360*a92 + a94 + a97*X[1] - a97;
        G[60] = 336*a100 - 1008*a101 + 840*a103 - 28*a99;
        G[61] = 0;
        G[62] = 0;
        G[63] = -1296*a104*a32 + a104*a95 - 50*a104 + 528*a105 + 2160*a106*X[1] - 1368*a106 + 960*a107;
        G[64] = -6*a104 + 144*a105 - 648*a106 + 720*a107;
        G[65] = 0;
        G[66] = a102*a113 - a108*a33 - a109 + 164*a110 + a111*X[1] - 324*a112 - a113*a39 + 900*a114 + 1080*a115;
        G[67] = -a108*a29 - 8*a108 + a109*X[1] + a111*X[0] + 900*a115 + 360*a116 - 360*a117;
        G[68] = 0;
        G[69] = 132*a100 - 186*a101 + 80*a103 + 360*a118 - 990*a119 + 700*a121 - 1260*a122 + 1800*a123 + 900*a124 - 26*a99;
        G[70] = 360*a100 - 630*a101 + 300*a103 + 180*a118 - 210*a119 - 1980*a122 + 2100*a123 + 1800*a124 + a126;
        G[71] = 0;
        G[72] = a102*a128 - a127*a98 - a128 - a129*a28 + a129*X[0] + a131*a28 + a131 + a132*a34 - 252*a132 + 224*a133;
        G[73] = a102*a134 - a130*a28 + a130*X[0] + 504*a133 - a134 + a135*a28 + a135 + a136*X[0] - a136 - a137*a30;
        G[74] = 0;
        G[75] = -432*a138*a32 - 50*a138 - 864*a139*X[2] + 528*a139 + a140*X[1] + a141*X[2] + 720*a142*X[1] + 1440*a142*X[2] - 1368*a142 + 960*a143;
        G[76] = -2*a138 + a140*X[0] - 216*a142 + 240*a143;
        G[77] = -4*a138 + a141*X[0] - 432*a142 + 480*a143;
        G[78] = -a100*a146 + 492*a100 - 972*a101 + 540*a103 - a118*a147 + 342*a118 - 270*a119 - 2268*a122 + 1350*a123 + 2430*a124 - a144 + a145*a99 + 810*a148 + 2700*a149;
        G[79] = -a100*a147 + 342*a100 - 1134*a101 + 810*a103 + a125*X[2] + 1350*a148 + a151 - 18*a99;
        G[80] = 114*a100 - 378*a101 + 270*a103 + a151 - 6*a99;
        G[81] = 420*a108*a120 - a110*a155 + 132*a110 - 186*a112 + 1320*a114 + 780*a115 + 80*a116 - 1092*a117 + a152*a157 - 726*a152 - a153*a154 + 312*a153 + 120*a156 + 960*a158 + a160;
        G[82] = -a110*a154 + 312*a110 - 546*a112 + 1260*a114 + 1320*a115 + 260*a116 - 1452*a117 - 126*a152 - a153*a155 + 132*a153 + 480*a156 + 1680*a158 + a160;
        G[83] = -4*a108 + 48*a110 - 84*a112 + 840*a114 + 480*a115 + 40*a116 - 528*a117 - 84*a152 + a159*X[1];
        G[84] = a102*a162 - a162 - a163*a28 + a163*X[0] + a165*a28 + a165 + a167*a28 + a167 + a169*X[0] - a169 + a171 - 132*a172 - a173*a24 - 84*a175 + a176*X[2] + a177*X[2];
        G[85] = a102*a178 - a164*a28 + a164*X[0] + a168*a30 + 336*a170 + 336*a172*X[2] - 420*a172 - 336*a175 - a178 + a179*a28 + a179 + a181*a28 + a181 + a182*X[2] - a182 - 84*a183;
        G[86] = a102*a184 - a166*a28 + a166*X[0] + a171 + a176*X[0] - a176 - a177 + a185 + a186*a28 + a186;
        G[87] = a145*a161 - a146*a173 + a147*a172 - 20*a161 + 90*a168*X[0] - 18*a168 - a173*a31 + 164*a173 + 38*a174 - 108*a175 - 324*a187 + 180*a188 + 540*a189*X[0] - 108*a189 + a191 + 810*a192;
        G[88] = a167 - a173*a8 + 38*a173 - 108*a183 + a184*X[1] + a185 - 126*a187 + 90*a188 + 90*a190 + 270*a192;
        G[89] = a147*a187 + a161*a24 + a166*X[1] - a166 - 108*a172 + 114*a173 - 216*a183 - 378*a187 + 270*a188 + a191;
        G[90] = -a127*a30*X[2] + a127*a62 - 26*a127 + a132*a157 - 330*a132 + 140*a133 - a137*a198 - a193*a70 + a193*X[0] + a194*X[1] + a195*a96 - 186*a195 + 80*a196 - 756*a197 + 600*a199 + 840*a200*X[1] + 240*a201 + 540*a202 + 1920*a203;
        G[91] = -18*a127 - 42*a132 + a194*X[0] - 378*a195 + 180*a196 - 660*a197 - a198*a206 + 420*a199 - 84*a200 + 840*a201 + 600*a202 + a204*X[1] + a205*X[2] + 960*a207 + a208;
        G[92] = -84*a132 + a134*X[2] - 252*a195 + 120*a196 - 1056*a197 + 840*a199 + 960*a202 + a205*X[1] - a206*a210 + 144*a206 + 240*a207 + a208 + a209;
        G[93] = a102*a211 + a118*a219 + 270*a118 - 690*a119 + 440*a121 + 690*a123 + a144*X[0] + a150 - a211 + a213*a28 + a213 - a214 + a215*X[0] - a215 - 180*a216 - 1020*a217 + a218*X[1] + a220*X[2] + a221*X[2];
        G[94] = a100*a219 + 270*a100 + a101*a222 + a102*a212 + 2160*a118*a70 + 690*a118 - 1320*a119 + 720*a121 - 1380*a122 + 1320*a123 + 690*a124 + 3120*a149 - a212 - 1020*a216 - 3120*a217 - a218 - a219*a99 + a222*a99 + 3240*a223;
        G[95] = a102*a125 + 510*a118 + 1080*a121 + 1560*a123 + 510*a124 + a126 + a144*X[2] + 960*a149 - a212*a28 + a212*X[0] + a214*X[2] - 120*a216 - 960*a217 - a220 - a221 + 2160*a223;
        G[96] = 20*a120*a224 - a154*a226 + 400*a224*a232 - 26*a224 - a225*a31 + 132*a225 + 72*a226 + 288*a227 + a228*a58 - 186*a228 + 80*a229 + 240*a230*X[2] - 66*a230 - 660*a231 - 1008*a233 + 120*a234 + a235 + 180*a236 + a237*X[1] + 960*a238;
        G[97] = -a154*a225 + 72*a225 + 12*a226 + 48*a227 - 126*a228 + 60*a229 - 60*a231 + 60*a234 + 120*a236 + a237*X[0] + 480*a238 - a239*a39 - a239 - 132*a240 - a241*X[2] + 480*a242;
        G[98] = -a224*a29 + 288*a225 - 120*a226*X[2] + 120*a227 + 240*a229 - 120*a231 - 1320*a233 + 240*a234 + a235 + 480*a236 + 1200*a238 - 528*a240 + a241 + 1200*a242 - a243*a39 - a243;
        G[99] = a102*a245 - a210*a253 + a210*a259 - a245 + a247*X[0] - a247 - a248*a28 + a248*X[0] + 20*a249 + 28*a250 + a251*a28 + a251 + a252*a28 + a252 - a254*a32 - a254*a54 + a256*X[2] + a258*X[1] + a261;
        G[100] = 180*a249 + 36*a250 + 324*a255*X[2] + a258*X[0] - a258 + a262*X[0] - a262 + a263*a28 + a263 + a265*a28 + a265 - 456*a266 - a267*a54 - a268*a32 + 456*a269 + a271;
        G[101] = a147*a255 + 108*a250 + a256*X[0] - a256 + a261 - 600*a266 - a267*a32 - a268*a54 + 600*a269 + a271 + a272*a28 + a272 + a273*a28 + a273;
        G[102] = a280;
        G[103] = a280;
        G[104] = a102*a204 + a120*a204 + a127*a147 + a132*a147 - a137*a282 + a147*a195 + a197*a282 - 360*a197 - a204 - a206*a282 + 180*a206 + a274*X[1] + 840*a277 + a278*X[1] - a278 + a279*X[0] - a279 + a281*X[0] + a281*X[1] - a281;
        return G;
    }
               
    Matrix<Size, Dims> antiderivatives(Vector<3> const& X) const
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = a0*X[0];
        Scalar const a2 = std::sqrt(10);
        Scalar const a3 = a2*X[0];
        Scalar const a4 = X[0]*X[0];
        Scalar const a5 = a2*a4;
        Scalar const a6 = std::sqrt(5);
        Scalar const a7 = 2*a6;
        Scalar const a8 = -a7;
        Scalar const a9 = 6*X[1];
        Scalar const a10 = a6*a9 + a8;
        Scalar const a11 = std::sqrt(15);
        Scalar const a12 = 2*a11;
        Scalar const a13 = a12*X[1];
        Scalar const a14 = -a12;
        Scalar const a15 = 4*X[2];
        Scalar const a16 = a11*a15 + a14;
        Scalar const a17 = std::sqrt(14);
        Scalar const a18 = a17*X[0];
        Scalar const a19 = a17*a4;
        Scalar const a20 = X[0]*X[0]*X[0];
        Scalar const a21 = a17*a20;
        Scalar const a22 = std::sqrt(7);
        Scalar const a23 = 4*a20;
        Scalar const a24 = a22*a9;
        Scalar const a25 = 18*X[1];
        Scalar const a26 = std::sqrt(42);
        Scalar const a27 = (1.0/3.0)*a20;
        Scalar const a28 = 4*a26;
        Scalar const a29 = 8*X[1];
        Scalar const a30 = a26*a29;
        Scalar const a31 = X[1]*X[1];
        Scalar const a32 = 10*a26*a31 + a26 - a30;
        Scalar const a33 = std::sqrt(21);
        Scalar const a34 = 2*a33;
        Scalar const a35 = -a34;
        Scalar const a36 = a34*X[1];
        Scalar const a37 = a15*a33;
        Scalar const a38 = 12*X[2];
        Scalar const a39 = 3*a17;
        Scalar const a40 = -a39;
        Scalar const a41 = 9*a17;
        Scalar const a42 = a17*X[2];
        Scalar const a43 = a42*X[1];
        Scalar const a44 = 6*a17;
        Scalar const a45 = a39 - a44*X[2];
        Scalar const a46 = a17*a25;
        Scalar const a47 = -a46;
        Scalar const a48 = a17*a31;
        Scalar const a49 = a47 + 15*a48;
        Scalar const a50 = std::sqrt(210);
        Scalar const a51 = a50*X[1];
        Scalar const a52 = 3*a50;
        Scalar const a53 = -a50 + a52*X[2];
        Scalar const a54 = 6*X[2];
        Scalar const a55 = 6*a50;
        Scalar const a56 = X[2]*X[2];
        Scalar const a57 = a50 + a55*a56 - a55*X[2];
        Scalar const a58 = 2*a51;
        Scalar const a59 = a31*a50 - a58;
        Scalar const a60 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a61 = 3*a60;
        Scalar const a62 = 27*a60;
        Scalar const a63 = a20*a60;
        Scalar const a64 = X[0]*X[0]*X[0]*X[0];
        Scalar const a65 = 42*a64;
        Scalar const a66 = a25 - 6;
        Scalar const a67 = 126*X[1];
        Scalar const a68 = 168*X[1];
        Scalar const a69 = 6*a0;
        Scalar const a70 = a0*X[1];
        Scalar const a71 = 3*a0;
        Scalar const a72 = a0*a31;
        Scalar const a73 = 30*a72;
        Scalar const a74 = a60*a64;
        Scalar const a75 = 6*a60;
        Scalar const a76 = -a75;
        Scalar const a77 = a60*X[1];
        Scalar const a78 = 9*a60;
        Scalar const a79 = 90*a60;
        Scalar const a80 = a79*X[1];
        Scalar const a81 = a31*a60;
        Scalar const a82 = 270*a81;
        Scalar const a83 = X[1]*X[1]*X[1];
        Scalar const a84 = 210*a83;
        Scalar const a85 = a60*a84 + a76 + a80 - a82;
        Scalar const a86 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a87 = 6*a86;
        Scalar const a88 = a87*X[1];
        Scalar const a89 = -a87;
        Scalar const a90 = a38*a86 + a89;
        Scalar const a91 = a86*X[1];
        Scalar const a92 = 84*a86;
        Scalar const a93 = a86*X[2];
        Scalar const a94 = a60*X[2];
        Scalar const a95 = 45*a60;
        Scalar const a96 = a31*a95;
        Scalar const a97 = 360*a77;
        Scalar const a98 = (3.0/2.0)*a64;
        Scalar const a99 = -a69;
        Scalar const a100 = a0*X[2];
        Scalar const a101 = 78*a70;
        Scalar const a102 = 12*a0;
        Scalar const a103 = a102*X[2];
        Scalar const a104 = 72*a70;
        Scalar const a105 = 144*X[2];
        Scalar const a106 = a103 + a99;
        Scalar const a107 = 126*a0*a83 + a101 - 198*a72;
        Scalar const a108 = std::sqrt(30);
        Scalar const a109 = 6*a108;
        Scalar const a110 = a108*X[1];
        Scalar const a111 = a108*X[2];
        Scalar const a112 = 3*a108;
        Scalar const a113 = a112*a31;
        Scalar const a114 = a108*a56;
        Scalar const a115 = 18*a114;
        Scalar const a116 = a108*a31;
        Scalar const a117 = 72*a111;
        Scalar const a118 = 6*a2;
        Scalar const a119 = -a118;
        Scalar const a120 = 12*a2;
        Scalar const a121 = a2*X[1];
        Scalar const a122 = 54*a121;
        Scalar const a123 = 36*a2;
        Scalar const a124 = a123*X[2];
        Scalar const a125 = a2*a31;
        Scalar const a126 = 18*a2;
        Scalar const a127 = 144*a121;
        Scalar const a128 = a119 - a123*a56 + a124;
        Scalar const a129 = 42*a83;
        Scalar const a130 = a122 - 90*a125 + a129*a2;
        Scalar const a131 = -a44;
        Scalar const a132 = a131 + 24*a42;
        Scalar const a133 = 72*a42;
        Scalar const a134 = a17*a56;
        Scalar const a135 = -a133 + 90*a134 + a41;
        Scalar const a136 = 72*X[2];
        Scalar const a137 = 180*a134;
        Scalar const a138 = X[2]*X[2]*X[2];
        Scalar const a139 = a131 + a133 - a137 + 120*a138*a17;
        Scalar const a140 = a44*a83 + a46 - 18*a48;
        Scalar const a141 = std::sqrt(22);
        Scalar const a142 = a141*X[0];
        Scalar const a143 = a141*a4;
        Scalar const a144 = a141*a20;
        Scalar const a145 = a141*a64;
        Scalar const a146 = std::pow(X[0], 5);
        Scalar const a147 = a141*a146;
        Scalar const a148 = std::sqrt(11);
        Scalar const a149 = 48*a146;
        Scalar const a150 = a148*a9;
        Scalar const a151 = a148*X[1];
        Scalar const a152 = std::sqrt(66);
        Scalar const a153 = 9*a146;
        Scalar const a154 = a152*X[1];
        Scalar const a155 = 10*a152;
        Scalar const a156 = -a152*a29 + a152 + a155*a31;
        Scalar const a157 = a152*a31;
        Scalar const a158 = a141*X[1];
        Scalar const a159 = a141*a31;
        Scalar const a160 = 30*a141;
        Scalar const a161 = a160*X[1];
        Scalar const a162 = a141*a83;
        Scalar const a163 = 70*a162;
        Scalar const a164 = std::sqrt(110);
        Scalar const a165 = (1.0/5.0)*a146;
        Scalar const a166 = 6*a164;
        Scalar const a167 = 2*a164;
        Scalar const a168 = a164*X[1];
        Scalar const a169 = 24*a168;
        Scalar const a170 = -a169;
        Scalar const a171 = 42*a164;
        Scalar const a172 = a171*a31;
        Scalar const a173 = -a167;
        Scalar const a174 = 36*a168;
        Scalar const a175 = 126*a164;
        Scalar const a176 = a175*a31;
        Scalar const a177 = a164*a83;
        Scalar const a178 = 224*a177;
        Scalar const a179 = X[1]*X[1]*X[1]*X[1];
        Scalar const a180 = a164 + a170 + a175*a179 + a176 - a178;
        Scalar const a181 = std::sqrt(33);
        Scalar const a182 = 2*a181;
        Scalar const a183 = -a182;
        Scalar const a184 = a182*X[1];
        Scalar const a185 = a15*a181;
        Scalar const a186 = a181*X[1];
        Scalar const a187 = a181*X[2];
        Scalar const a188 = a141*X[2];
        Scalar const a189 = 3*a141;
        Scalar const a190 = -6*a188 + a189;
        Scalar const a191 = 15*a141;
        Scalar const a192 = -a141*a25 + a191*a31;
        Scalar const a193 = a158*X[2];
        Scalar const a194 = 4*a152;
        Scalar const a195 = a154*X[2];
        Scalar const a196 = a152*X[2];
        Scalar const a197 = 22*a152 - 28*a196;
        Scalar const a198 = 2*a152;
        Scalar const a199 = -a198;
        Scalar const a200 = 26*a154;
        Scalar const a201 = a194*X[2];
        Scalar const a202 = a129*a152;
        Scalar const a203 = a157*X[2];
        Scalar const a204 = -13*a152 + 24*a196;
        Scalar const a205 = std::sqrt(330);
        Scalar const a206 = -a205;
        Scalar const a207 = a205*X[1];
        Scalar const a208 = a205*X[2];
        Scalar const a209 = 2*a205;
        Scalar const a210 = a205*a31;
        Scalar const a211 = a207*X[2];
        Scalar const a212 = -22*a207;
        Scalar const a213 = a209*X[2];
        Scalar const a214 = -a213;
        Scalar const a215 = a212 + a214;
        Scalar const a216 = 3*a205;
        Scalar const a217 = a216*X[2];
        Scalar const a218 = 105*a210;
        Scalar const a219 = a205*a83;
        Scalar const a220 = 84*a219;
        Scalar const a221 = 42*a205;
        Scalar const a222 = a221*X[1];
        Scalar const a223 = a222*X[2];
        Scalar const a224 = 84*a210;
        Scalar const a225 = 168*a219;
        Scalar const a226 = 84*a179*a205 + a205 + a218 - a225;
        Scalar const a227 = 6*a208;
        Scalar const a228 = a205*a56;
        Scalar const a229 = a205 - a227 + 6*a228;
        Scalar const a230 = -a209*X[1] + a210;
        Scalar const a231 = 4*a164;
        Scalar const a232 = 30*a164;
        Scalar const a233 = a164*X[2];
        Scalar const a234 = a164*a31;
        Scalar const a235 = a164*a56;
        Scalar const a236 = a168*X[2];
        Scalar const a237 = a164*a25;
        Scalar const a238 = 12*a164;
        Scalar const a239 = a238*X[2];
        Scalar const a240 = 14*a177;
        Scalar const a241 = a168*a56;
        Scalar const a242 = a234*X[2];
        Scalar const a243 = 5*a141;
        Scalar const a244 = 10*a141;
        Scalar const a245 = 115*a141;
        Scalar const a246 = -90*a158;
        Scalar const a247 = a160*X[2];
        Scalar const a248 = -a247;
        Scalar const a249 = a246 + a248;
        Scalar const a250 = 45*a141;
        Scalar const a251 = 345*a159;
        Scalar const a252 = a160*a56;
        Scalar const a253 = 510*a193;
        Scalar const a254 = a158*a56;
        Scalar const a255 = 780*a159;
        Scalar const a256 = a243 + a252;
        Scalar const a257 = 180*a141*a179 - 440*a162 + a251;
        Scalar const a258 = std::sqrt(154);
        Scalar const a259 = a258*X[1];
        Scalar const a260 = a258*X[2];
        Scalar const a261 = 22*a258;
        Scalar const a262 = 20*a258;
        Scalar const a263 = a258*a56;
        Scalar const a264 = a259*X[2];
        Scalar const a265 = a258*a9;
        Scalar const a266 = 24*a260;
        Scalar const a267 = 2*a258;
        Scalar const a268 = a267*a83;
        Scalar const a269 = a138*a258;
        Scalar const a270 = 40*a269;
        Scalar const a271 = 60*a263;
        Scalar const a272 = a258*a31;
        Scalar const a273 = a272*X[2];
        Scalar const a274 = 10*a258;
        Scalar const a275 = 300*a263;
        Scalar const a276 = std::sqrt(770);
        Scalar const a277 = a276*X[1];
        Scalar const a278 = a276*X[2];
        Scalar const a279 = 2*a276;
        Scalar const a280 = 10*a276;
        Scalar const a281 = a278*X[1];
        Scalar const a282 = 12*a277;
        Scalar const a283 = -a282;
        Scalar const a284 = a276*a38;
        Scalar const a285 = -a284;
        Scalar const a286 = a283 + a285;
        Scalar const a287 = 30*a276;
        Scalar const a288 = a287*a31;
        Scalar const a289 = a276*a83;
        Scalar const a290 = a287*a56;
        Scalar const a291 = 132*a281;
        Scalar const a292 = a277*a56;
        Scalar const a293 = a278*a31;
        Scalar const a294 = a276*a56;
        Scalar const a295 = 270*a31;
        Scalar const a296 = a138*a276;
        Scalar const a297 = 20*a296;
        Scalar const a298 = a276 + a290 - a297;
        Scalar const a299 = 28*a289;
        Scalar const a300 = 9*a179*a276 + a288 - a299;
        Scalar const a301 = (3.0/5.0)*a164;
        Scalar const a302 = 3*a164;
        Scalar const a303 = 15*a233 - a302;
        Scalar const a304 = 60*a233;
        Scalar const a305 = -a304;
        Scalar const a306 = a238*X[1];
        Scalar const a307 = -a306;
        Scalar const a308 = a305 + a307;
        Scalar const a309 = 90*a164;
        Scalar const a310 = a166 + a309*a56;
        Scalar const a311 = 18*a234;
        Scalar const a312 = 180*a168;
        Scalar const a313 = a312*X[2];
        Scalar const a314 = a309*X[2];
        Scalar const a315 = -a166;
        Scalar const a316 = 270*a235;
        Scalar const a317 = a138*a164;
        Scalar const a318 = a314 + a315 - a316 + 210*a317;
        Scalar const a319 = 420*a317;
        Scalar const a320 = X[2]*X[2]*X[2]*X[2];
        Scalar const a321 = 210*a164*a320 + a302 + a316 - a319;
        Scalar const a322 = a238*a83;
        Scalar const a323 = a179*a302 + a311 - a322;
        Scalar const a324 = -a2 + 4*a3;
        Scalar const a325 = a7*X[0];
        Scalar const a326 = a12*X[0];
        Scalar const a327 = a17 - 10*a18 + 15*a19;
        Scalar const a328 = a22*X[0];
        Scalar const a329 = 12*a4;
        Scalar const a330 = a22*a329 + 2*a22 - 14*a328;
        Scalar const a331 = a26*a4 - 2*a26*X[0];
        Scalar const a332 = a33*X[0];
        Scalar const a333 = 24*X[2];
        Scalar const a334 = a329*a33 - 14*a332 + a34;
        Scalar const a335 = 6*a18;
        Scalar const a336 = -a335 + a39*a4;
        Scalar const a337 = a50*X[0];
        Scalar const a338 = -2*a337 + a4*a50;
        Scalar const a339 = a60*X[0];
        Scalar const a340 = a4*a60;
        Scalar const a341 = 168*a20;
        Scalar const a342 = 54*a339 - 189*a340 + a341*a60 - a61;
        Scalar const a343 = 252*a4;
        Scalar const a344 = 90*X[0];
        Scalar const a345 = a341 - a343 + a344;
        Scalar const a346 = a0*a4;
        Scalar const a347 = 24*a20;
        Scalar const a348 = a0*a347 + 30*a1 - 51*a346 - a71;
        Scalar const a349 = a344*a60;
        Scalar const a350 = a20*a75 + 18*a339 - 18*a340;
        Scalar const a351 = a341*a86 - a343*a86 + a344*a86;
        Scalar const a352 = a94*X[0];
        Scalar const a353 = 144*a4;
        Scalar const a354 = -153*a340 + a349 + 72*a63 - a78;
        Scalar const a355 = 39*a0;
        Scalar const a356 = 18*a1 + a20*a69 - 18*a346;
        Scalar const a357 = a108*X[0];
        Scalar const a358 = 9*a108;
        Scalar const a359 = a108*a4;
        Scalar const a360 = 144*X[0];
        Scalar const a361 = a108*a347 - a112 + 30*a357 - 51*a359;
        Scalar const a362 = a2*X[2];
        Scalar const a363 = 27*a2;
        Scalar const a364 = 36*a3;
        Scalar const a365 = a126*a4;
        Scalar const a366 = a118*a20 + 18*a3 - a365;
        Scalar const a367 = 18*a18;
        Scalar const a368 = 72*a18;
        Scalar const a369 = -18*a19 + 6*a21 + a367;
        Scalar const a370 = a141 - 28*a142 + 168*a143 - 336*a144 + 210*a145;
        Scalar const a371 = a148*X[0];
        Scalar const a372 = a148*a4;
        Scalar const a373 = a148*a20;
        Scalar const a374 = 240*a64;
        Scalar const a375 = a148*a374 + 2*a148 - 50*a371 + 264*a372 - 456*a373;
        Scalar const a376 = a152*X[0];
        Scalar const a377 = a152*a4;
        Scalar const a378 = a152*a20;
        Scalar const a379 = a152*a64;
        Scalar const a380 = -20*a376 + 82*a377 - 108*a378 + 45*a379;
        Scalar const a381 = -a191;
        Scalar const a382 = 2*a141 - 26*a142 + 66*a143 - 62*a144 + 20*a145;
        Scalar const a383 = std::pow(X[1], 5);
        Scalar const a384 = 56*a164;
        Scalar const a385 = a164*X[0];
        Scalar const a386 = 36*a164;
        Scalar const a387 = a20*a238;
        Scalar const a388 = a164*a64;
        Scalar const a389 = a166*a4;
        Scalar const a390 = -a20*a231 - a231*X[0] + a388 + a389;
        Scalar const a391 = a181*X[0];
        Scalar const a392 = a181*a4;
        Scalar const a393 = a181*a20;
        Scalar const a394 = 432*a4;
        Scalar const a395 = a181*a374 + a182 - 50*a391 + 264*a392 - 456*a393;
        Scalar const a396 = 90*a142;
        Scalar const a397 = a142*X[2];
        Scalar const a398 = a143*X[2];
        Scalar const a399 = 270*X[2];
        Scalar const a400 = 60*a142;
        Scalar const a401 = 246*a143 - 324*a144 + 135*a145 - a400;
        Scalar const a402 = a196*X[0];
        Scalar const a403 = a196*a4;
        Scalar const a404 = a198 - 26*a376 + 66*a377 - 62*a378 + 20*a379;
        Scalar const a405 = a221*X[0];
        Scalar const a406 = 35*a205;
        Scalar const a407 = a205*X[0];
        Scalar const a408 = 56*a208;
        Scalar const a409 = 11*a205;
        Scalar const a410 = 33*a205;
        Scalar const a411 = 21*a208;
        Scalar const a412 = a205*a64;
        Scalar const a413 = a20*a205;
        Scalar const a414 = a205*a4;
        Scalar const a415 = -4*a407 + a412 - 4*a413 + 6*a414;
        Scalar const a416 = a208*X[0];
        Scalar const a417 = a208*a4;
        Scalar const a418 = -20*a407 + 45*a412 - 108*a413 + 82*a414;
        Scalar const a419 = a164*a4;
        Scalar const a420 = a233*X[0];
        Scalar const a421 = a235*X[0];
        Scalar const a422 = a233*a4;
        Scalar const a423 = 62*a20;
        Scalar const a424 = -a164*a423 + a167 - 26*a385 + 20*a388 + 66*a419;
        Scalar const a425 = a141*a56;
        Scalar const a426 = -20*a142;
        Scalar const a427 = -20*a144 + a160*a4 + a243*a64 + a426;
        Scalar const a428 = a258*X[0];
        Scalar const a429 = a260*X[0];
        Scalar const a430 = a258*a4;
        Scalar const a431 = a20*a258;
        Scalar const a432 = 240*a260;
        Scalar const a433 = 504*a4;
        Scalar const a434 = -a258*a423 + a262*a64 + a267 - 26*a428 + 66*a430;
        Scalar const a435 = 7*a276;
        Scalar const a436 = 76*a278;
        Scalar const a437 = a276*X[0];
        Scalar const a438 = a280*a4 + a280 - 20*a437;
        Scalar const a439 = 150*a294;
        Scalar const a440 = 66*a278;
        Scalar const a441 = a278*X[0];
        Scalar const a442 = 6*a276;
        Scalar const a443 = 18*a276;
        Scalar const a444 = a20*a442 - a4*a443 - a442 + a443*X[0];
        Scalar const a445 = 36*a4;
        Scalar const a446 = -a23*a276 + a276*a64 + a4*a442 - 4*a437;
        Scalar const a447 = -a238*X[0];
        Scalar const a448 = a305 + a447;
        Scalar const a449 = 18*a419;
        Scalar const a450 = 180*a420;
        Scalar const a451 = a302*a64 - a387 + a449;
        Scalar const a452 = 36*X[1];
        Scalar const a453 = 12*a332;
        Scalar const a454 = a17*X[1];
        Scalar const a455 = X[0]*X[1];
        Scalar const a456 = a1*X[1];
        Scalar const a457 = a1*a31;
        Scalar const a458 = a77*X[0];
        Scalar const a459 = a4*a86;
        Scalar const a460 = a110*X[0];
        Scalar const a461 = a3*X[1];
        Scalar const a462 = 60*a17;
        Scalar const a463 = 36*a17;
        Scalar const a464 = a157*X[0];
        Scalar const a465 = a154*X[0];
        Scalar const a466 = a154*a20;
        Scalar const a467 = a157*a4;
        Scalar const a468 = a142*a31;
        Scalar const a469 = a143*X[1];
        Scalar const a470 = a144*X[1];
        Scalar const a471 = a142*X[1];
        Scalar const a472 = a142*a83;
        Scalar const a473 = a143*a31;
        Scalar const a474 = a234*X[0];
        Scalar const a475 = 72*a168;
        Scalar const a476 = 270*a471;
        Scalar const a477 = a154*a4;
        Scalar const a478 = 21*a207;
        Scalar const a479 = a210*X[0];
        Scalar const a480 = 66*a207;
        Scalar const a481 = a413*X[1];
        Scalar const a482 = a207*X[0];
        Scalar const a483 = a168*X[0];
        Scalar const a484 = 60*a20;
        Scalar const a485 = a168*a4;
        Scalar const a486 = a259*X[0];
        Scalar const a487 = 12*a258;
        Scalar const a488 = a272*X[0];
        Scalar const a489 = 5*a276;
        Scalar const a490 = 100*a277;
        Scalar const a491 = a276*a31;
        Scalar const a492 = 66*a277;
        Scalar const a493 = a277*X[0];
        Scalar const a494 = a31*a437;
        Scalar const a495 = 105*a164;
        Scalar const a496 = a309*a4;
        Scalar const a497 = a309*a31;
        Scalar const a498 = a312*X[0];
        P[0] = a1;
        P[1] = -a3 + 2*a5;
        P[2] = a10*X[0] + a4*a6;
        P[3] = a11*a4 + (a13 + a16)*X[0];
        P[4] = a18 - 5*a19 + 5*a21;
        P[5] = a22*a23 + a4*(a22*a25 - 7*a22) + (2*a22 - a24)*X[0];
        P[6] = a26*a27 + a32*X[0] + a4*(-a26 + a28*X[1]);
        P[7] = a23*a33 + a4*(a33*a38 + a33*a9 - 7*a33) + (-a35 - a36 - a37)*X[0];
        P[8] = a21 + a4*(a39*X[2] + a40 + a41*X[1]) + (30*a43 + a45 + a49)*X[0];
        P[9] = a27*a50 + a4*(a51 + a53) + (a51*a54 + a57 + a59)*X[0];
        P[10] = a4*a62 + a60*a65 - a61*X[0] - 63*a63;
        P[11] = a20*(a68 - 84) + a4*(45 - a67) + a65 + a66*X[0];
        P[12] = a20*(-17*a0 + 64*a70) + a4*(15*a0 - 108*a70 + 120*a72) + a64*a69 + (24*a0*X[1] - a71 - a73)*X[0];
        P[13] = a20*(a76 + 30*a77) + a4*(a78 - a80 + 135*a81) + (3.0/2.0)*a74 + a85*X[0];
        P[14] = a20*(56*a91 - a92 + 112*a93) + a4*(45*a86 - 42*a91 - a92*X[2]) + a65*a86 + (a88 + a90)*X[0];
        P[15] = a20*(-51*a60 + 144*a77 + 48*a94) + a4*(-243*a77 + 180*a81 - 81*a94 + a95 + a97*X[2]) + 18*a74 + (54*a60*X[1] + 18*a60*X[2] - a78 - a80*X[2] - a96)*X[0];
        P[16] = a0*a98 + a20*(4*a100 + 26*a70 + a99) + a4*(9*a0 - a101 - a103 + a104*X[2] + 99*a72) + (252*a100*a31 - a105*a70 + a106 + a107)*X[0];
        P[17] = a109*a64 + a20*(-17*a108 + 16*a110 + 48*a111) + a4*(15*a108 - 27*a110 - 81*a111 + 72*a114 + 12*a116 + a117*X[1]) + (6*a108*X[1] + 18*a108*X[2] - a111*a25 - a112 - a113 - a115)*X[0];
        P[18] = a2*a98 + a20*(a119 + a120*X[2] + a2*a25) + a4*(-a122 - a124 + 45*a125 + a126*a56 + a127*X[2] + 9*a2) + (252*a121*a56 - 288*a121*X[2] + 252*a125*X[2] + a128 + a130)*X[0];
        P[19] = a17*a98 + a20*(a132 + a44*X[1]) + a4*(a133*X[1] + a135 + a31*a41 + a47) + (a136*a48 + a137*X[1] + a139 + a140 - 144*a43)*X[0];
        P[20] = a142 - 14*a143 + 56*a144 - 84*a145 + 42*a147;
        P[21] = a148*a149 + a20*(88*a148 - 216*a151) + a4*(-25*a148 + 72*a151) + a64*(-114*a148 + 180*a151) + (2*a148 - a150)*X[0];
        P[22] = a152*a153 + a156*X[0] + a20*(-a152*a68 + (82.0/3.0)*a152 + 150*a157) + a4*(76*a152*X[1] - a155 - 90*a157) + a64*(-27*a152 + 90*a154);
        P[23] = 4*a147 + a20*(22*a141 - 210*a158 + 300*a159) + a4*(-13*a141 + 180*a158 - 495*a159 + 350*a162) + a64*(-31.0/2.0*a141 + 75*a158) + (90*a141*a31 + 2*a141 - a161 - a163)*X[0];
        P[24] = a164*a165 + a180*X[0] + a20*(a167 + a170 + a172) + a4*(a173 + a174 - a176 + 112*a177) + a64*(-a164 + a166*X[1]);
        P[25] = a149*a181 + a20*(88*a181 - 72*a186 - 144*a187) + a4*(-25*a181 + 24*a186 + 48*a187) + a64*(-114*a181 + 60*a186 + 120*a187) + (-a183 - a184 - a185)*X[0];
        P[26] = 27*a147 + a20*(82*a141 - 378*a158 + 225*a159 - 126*a188 + 450*a193) + a4*(171*a141*X[1] + 57*a141*X[2] - 135*a159 - a160 - 270*a193) + a64*(-81*a141 + (405.0/2.0)*a158 + (135.0/2.0)*a188) + (a161*X[2] + a190 + a192)*X[0];
        P[27] = a146*a194 + a20*(-182*a154 + 220*a157 + 160*a195 + a197) + a4*(a152*a84 + 156*a154 - 363*a157 - 264*a195 + 420*a203 + a204) + a64*(-31.0/2.0*a152 + 65*a154 + a155*X[2]) + (66*a152*a31 + 48*a152*X[1]*X[2] - a199 - a200 - a201 - a202 - 84*a203)*X[0];
        P[28] = a165*a205 + a20*(a209 + 35*a210 + 14*a211 + a215) + a4*(33*a207 - a209 + a217 - a218 + a220 - a223 + a224*X[2]) + a64*(a206 + (11.0/2.0)*a207 + (1.0/2.0)*a208) + (-168*a210*X[2] + a215 + a223 + a225*X[2] + a226)*X[0];
        P[29] = a153*a205 + a20*((82.0/3.0)*a205 - 126*a208 + 15*a210 + 90*a211 - a222 + 90*a228) + a4*(19*a205*X[1] + 57*a205*X[2] - 10*a205 - 9*a210 - 54*a211 - 54*a228) + a64*(-27*a205 + (45.0/2.0)*a207 + (135.0/2.0)*a208) + (a208*a9 + a229 + a230)*X[0];
        P[30] = a146*a231 + a20*(-a164*a67 + 22*a164 - 84*a233 + 100*a234 + 40*a235 + 320*a236) + a4*(-13*a164 + 108*a168 + 70*a177 + 72*a233 - 165*a234 - 66*a235 - 528*a236 + 420*a241 + 420*a242) + a64*(-31.0/2.0*a164 + 45*a168 + a232*X[2]) + (30*a164*a31 + 12*a164*a56 + 96*a164*X[1]*X[2] - a173 - a237 - a239 - a240 - 84*a241 - 84*a242)*X[0];
        P[31] = a147 + a20*(170*a193 + a244*a56 + a244 + a245*a31 + a249) + a4*(135*a158 + 220*a162 - a244 + a250*X[2] - a251 - a252 - a253 + 240*a254 + a255*X[2]) + a64*((45.0/2.0)*a158 + (15.0/2.0)*a188 - a243) + (1080*a159*a56 - 1560*a159*X[2] + 1080*a162*X[2] + a249 + a253 - 480*a254 + a256 + a257)*X[0];
        P[32] = 4*a146*a258 + a20*(-42*a259 - 168*a260 + a261 + a262*a31 + 200*a263 + 160*a264) + a4*(-13*a258 + 36*a259 + 144*a260 - 330*a263 - 264*a264 + 200*a269 - 33*a272 + 120*a273 + a274*a83 + a275*X[1]) + a64*(-31.0/2.0*a258 + 15*a259 + 60*a260) + (6*a258*a31 + 60*a258*a56 + 48*a258*X[1]*X[2] + 2*a258 - a265 - a266 - a268 - a270 - a271*X[1] - 24*a273)*X[0];
        P[33] = a165*a276 + a20*(a279 + a280*a31 + a280*a56 + 44*a281 + a286) + a4*(a138*a280 + a25*a276 + 18*a278 - a279 - a288 + 14*a289 - a290 - a291 + 150*a292 + 114*a293) + a64*(-a276 + 3*a277 + 3*a278) + (180*a138*a277 + a286 + 108*a289*X[2] + a291 - 300*a292 - 228*a293 + a294*a295 + a298 + a300)*X[0];
        P[34] = a146*a301 + a20*(a166*a31 + 60*a236 + a308 + a310) + a4*(a166*a83 + a237 + 270*a241 + a31*a314 - a311 - a313 + a318) + a64*(a302*X[1] + a303) + (420*a138*a168 + 60*a177*X[2] + 270*a234*a56 - 540*a241 - 180*a242 + a308 + a313 + a321 + a323)*X[0];
        P[35] = a70;
        P[36] = a324*X[1];
        P[37] = 3*a31*a6 + (a325 + a8)*X[1];
        P[38] = a11*a31 + (a16 + a326)*X[1];
        P[39] = a327*X[1];
        P[40] = a31*(-3*a22 + 18*a328) + a330*X[1];
        P[41] = (10.0/3.0)*a26*a83 + a31*(a28*X[0] - a28) + (a26 + a331)*X[1];
        P[42] = a31*(-a33 + 6*a332) + (a332*a333 + a334 - a37)*X[1];
        P[43] = 5*a17*a83 + a31*(9*a18 - a41 + 15*a42) + (a335*X[2] + a336 + a45)*X[1];
        P[44] = a31*(a337 + a53) + (1.0/3.0)*a50*a83 + (a337*a54 + a338 + a57)*X[1];
        P[45] = a342*X[1];
        P[46] = a31*(a343 - 126*X[0] + 9) + (a345 - 6)*X[1];
        P[47] = a31*(-108*a1 + a102 + 96*a346) + a348*X[1] + a83*(-10*a0 + 80*a1);
        P[48] = (105.0/2.0)*a179*a60 + a31*(-a349 + a4*a95 + a95) + a83*(a349 - a79) + (a350 + a76)*X[1];
        P[49] = a31*(a4*a92 - 42*a86*X[0] + 3*a86) + (a351 + 336*a4*a93 + a90 - 168*a93*X[0])*X[1];
        P[50] = a31*(-243*a339 + 216*a340 + 360*a352 + a62 - a95*X[2]) + a83*(120*a339 - 15*a60) + (-162*a352 + a353*a94 + a354 + 18*a94)*X[1];
        P[51] = (63.0/2.0)*a0*a179 + a31*(a1*a136 - 78*a1 - 72*a100 + a355*a4 + a355) + a83*(-66*a0 + 66*a1 + 84*a100) + (-a1*a333 + a103*a4 + a106 + a356)*X[1];
        P[52] = a31*(a112 + a117*X[0] - 27*a357 - a358*X[2] + 24*a359) + a83*(-a108 + 8*a357) + (a111*a353 - 162*a111*X[0] + 18*a111 + a114*a360 - a115 + a361)*X[1];
        P[53] = (21.0/2.0)*a179*a2 + a31*(a105*a3 + 126*a2*a56 - 54*a3 - 144*a362 + a363*a4 + a363) + a83*(-30*a2 + 30*a3 + 84*a362) + (a128 - a136*a3 + a364*a56 + a366 + 36*a5*X[2])*X[1];
        P[54] = (3.0/2.0)*a17*a179 + a31*(a135 - a367 + a368*X[2] + a4*a41) + a83*(a132 + a335) + (-a105*a18 + a136*a19 + a139 + 180*a18*a56 + a369)*X[1];
        P[55] = a370*X[1];
        P[56] = a31*(-3*a148 + 72*a371 - 324*a372 + 360*a373) + a375*X[1];
        P[57] = a31*(-a152*a343 - a194 + 76*a376 + 180*a378) + a83*((10.0/3.0)*a152 - 60*a376 + 150*a377) + (a152 + a380)*X[1];
        P[58] = a179*(-35.0/2.0*a141 + 175*a142) + a31*(180*a142 - 315*a143 + 150*a144 + a381) + a382*X[1] + a83*(-330*a142 + 300*a143 + a160);
        P[59] = (126.0/5.0)*a164*a383 + a179*(a384*X[0] - a384) + a31*(-a238 - a386*a4 + a386*X[0] + a387) + a83*(a171*a4 + a171 - 84*a385) + (a164 + a390)*X[1];
        P[60] = a31*(-a181 + 24*a391 - 108*a392 + 120*a393) + (-a185 + 480*a187*a20 - a187*a394 + 96*a187*X[0] + a395)*X[1];
        P[61] = a31*(-9*a141 + 171*a142 - 567*a143 + 405*a144 + a191*X[2] - 270*a397 + 675*a398) + a83*(225*a143 + a243 - a396) + (a144*a399 + a190 + 114*a397 - 378*a398 + a401)*X[1];
        P[62] = a179*(-21.0/2.0*a152 + 105*a376) + a31*(a204 + 156*a376 - 273*a377 + 130*a378 - 264*a402 + 240*a403) + a83*(a197 - 242*a376 + 220*a377 + 280*a402) + (40*a196*a20 - a201 + 48*a402 - 84*a403 + a404)*X[1];
        P[63] = a179*(a221*X[2] - a221 + a405) + (84.0/5.0)*a205*a383 + a31*(a20*a409 - a4*a410 + a4*a411 - a405*X[2] - a409 + a410*X[0] + a411) + a83*(a4*a406 + a406 - 70*a407 + a408*X[0] - a408) + (a20*a213 + a205 + a214 - a227*a4 + a227*X[0] + a415)*X[1];
        P[64] = a31*(a206 + a217 + 19*a407 + 45*a413 - 63*a414 - 54*a416 + 135*a417) + a83*((1.0/3.0)*a205 - 6*a407 + 15*a414) + (270*a228*a4 - 108*a228*X[0] + a229 + a399*a413 + 114*a416 - 378*a417 + a418)*X[1];
        P[65] = a179*(-7.0/2.0*a164 + 35*a385) + a31*(-9*a164 - a171*a56 + a20*a309 + 48*a233 + 108*a385 - 189*a419 - 528*a420 + 420*a421 + 480*a422) + a83*(10*a164 - 28*a233 - 110*a385 + 100*a419 + 280*a420) + (120*a20*a233 - a233*a343 + a233*a360 + 120*a235*a4 + a238*a56 - a239 - 132*a421 + a424)*X[1];
        P[66] = 36*a141*a383 + a179*(-110*a141 + 110*a142 + 270*a188) + a31*(240*a142*a56 + 135*a142 - 135*a143 + 255*a188 + a20*a250 - a250 - 510*a397 + 255*a398 - 240*a425) + a83*(-230*a142 - 520*a188 + a245*a4 + a245 + 520*a397 + 360*a425) + (a20*a247 + a248 + a252*a4 + a256 + a396*X[2] - 90*a398 - a400*a56 + a427)*X[1];
        P[67] = a179*(-1.0/2.0*a258 + 5*a428) + a31*(-3*a258 - 30*a263 + a266 + a275*X[0] + a4*a432 + 36*a428 - 264*a429 - 63*a430 + 30*a431) + a83*(-8*a260 - a261*X[0] + a262*a4 + a267 + 80*a429) + (a20*a432 - a260*a433 + 600*a263*a4 - 660*a263*X[0] - a266 + 400*a269*X[0] - a270 + a271 + 288*a429 + a434)*X[1];
        P[68] = a179*(27*a278 + a435*X[0] - a435) + (9.0/5.0)*a276*a383 + a31*(90*a296 + a4*a440 + a439*X[0] - a439 + a440 - 132*a441 + a444) + a83*(90*a294 + a436*X[0] - a436 + a438) + (a20*a284 - a278*a445 + a285 + a290*a4 - 60*a294*X[0] + a297*X[0] + a298 + 36*a441 + a446)*X[1];
        P[69] = a179*(a302*X[0] + a303) + a301*a383 + a31*(a166*a20 + a314*a4 + a316*X[0] + a318 + 18*a385 - a449 - a450) + a83*(a304*X[0] + a310 + a389 + a448) + (a20*a304 + a316*a4 + a319*X[0] + a321 - 540*a421 - 180*a422 + a448 + a450 + a451)*X[1];
        P[70] = a100;
        P[71] = a324*X[2];
        P[72] = (a10 + a325)*X[2];
        P[73] = a12*a56 + (a13 + a14 + a326)*X[2];
        P[74] = a327*X[2];
        P[75] = (-a24 + a328*a452 + a330)*X[2];
        P[76] = (a30*X[0] + a32 + a331)*X[2];
        P[77] = a56*(a35 + a453) + (a334 - a36 + a453*X[1])*X[2];
        P[78] = a56*(3*a18 + a40 + 15*a454) + (a18*a25 + a336 + a39 + a49)*X[2];
        P[79] = 2*a138*a50 + a56*(3*a337 + 3*a51 - a52) + (a338 + a50 + a58*X[0] + a59)*X[2];
        P[80] = a342*X[2];
        P[81] = (a345 + a433*X[1] - 252*a455 + a66)*X[2];
        P[82] = (a348 + 192*a4*a70 - 216*a456 + 240*a457 + 24*a70 - a73)*X[2];
        P[83] = (a350 + a4*a80 - 180*a458 + a82*X[0] + a85)*X[2];
        P[84] = a56*(168*a459 + a87 - a92*X[0]) + (a351 - a455*a92 + a459*a68 + a88 + a89)*X[2];
        P[85] = a56*(-81*a339 + 72*a340 + a78 - a95*X[1] + a97*X[0]) + (a354 + a394*a77 - 486*a458 + 54*a77 + 360*a81*X[0] - a96)*X[2];
        P[86] = a56*(-12*a1 - a104 + a4*a69 + 72*a456 + a69 + 126*a72) + (a101*a4 + a107 + a356 - 156*a456 + 198*a457 + a99)*X[2];
        P[87] = a138*(-a109 + 48*a357) + a56*(-81*a357 - a358*X[1] + a358 + 72*a359 + 72*a460) + (a109*X[1] + 48*a110*a4 - a113 + 24*a116*X[0] + a361 - 54*a460)*X[2];
        P[88] = a138*(-a120 + 84*a121 + 12*a3) + a56*(126*a125 + a126 - a127 - a364 + a365 + 144*a461) + (a119 + a130 + 90*a3*a31 + a366 - 108*a461 + 54*a5*X[1])*X[2];
        P[89] = a138*(60*a18 + a462*X[1] - a462) + 30*a17*a320 + a56*(a31*a463 + a368*X[1] - a368 + a4*a463 - 72*a454 + a463) + (a131 + a140 - a18*a452 + a19*a25 + a31*a367 + a369)*X[2];
        P[90] = a370*X[2];
        P[91] = (-a150 + 720*a151*a20 + a151*a360 - 648*a151*a4 + a375)*X[2];
        P[92] = (-a154*a433 + a156 + a380 - 180*a464 + 152*a465 + 360*a466 + 450*a467)*X[2];
        P[93] = (90*a159 - a161 - a163 + a382 - 990*a468 - 630*a469 + 300*a470 + 360*a471 + 700*a472 + 900*a473)*X[2];
        P[94] = (a169*a20 + a176*a4 + a178*X[0] + a180 + a390 - a4*a475 - 252*a474 + a475*X[0])*X[2];
        P[95] = a56*(a183 + 48*a391 - 216*a392 + 240*a393) + (-a184 + 240*a186*a20 - 216*a186*a4 + 48*a186*X[0] + a395)*X[2];
        P[96] = a56*(57*a142 - 189*a143 + 135*a144 - a189 + a191*X[1] + 675*a469 - a476) + (-a142*a295 + a189 + a192 + a401 - 1134*a469 + 810*a470 + 342*a471 + 675*a473)*X[2];
        P[97] = a56*(24*a154 - 42*a157 + a199 + 24*a376 - 42*a377 + 20*a378 + 420*a464 - 264*a465 + 240*a477) + (66*a157 - a200 - a202 + 420*a376*a83 + a404 - 726*a464 + 312*a465 + 260*a466 + 660*a467 - 546*a477)*X[2];
        P[98] = a56*(a206 - a216*a4 + a216*X[0] + a220 - a222*X[0] + a224*X[0] - a224 + a4*a478 + a413 + a478) + (a212 + a218*a4 + a225*X[0] + a226 - a4*a480 + a415 - 210*a479 + a480*X[0] + 22*a481)*X[2];
        P[99] = a138*(a209 - 36*a407 + 90*a414) + a56*(135*a207*a4 + a216*X[1] - a216 + 57*a407 + 135*a413 - 189*a414 - 54*a482) + (a205 + 45*a210*a4 + a230 - a414*a67 + a418 - 18*a479 + 90*a481 + 38*a482)*X[2];
        P[100] = a138*(-28*a168 + a231 - 44*a385 + 40*a419 + 280*a483) + a56*(a164*a484 + 48*a168 - a172 - a175*a4 + a315 + 72*a385 + 420*a474 - 528*a483 + 480*a485) + (140*a177*X[0] + a20*a312 + a232*a31 + 300*a234*a4 - a237 - a240 + a424 - 330*a474 + 216*a483 - 378*a485)*X[2];
        P[101] = a138*(-160*a158 + 360*a159 + a244*a4 + a244 + a426 + 160*a471) + a56*(45*a142 + 255*a158 + 540*a162 + a191*a20 - a250*a4 - a255 + a381 + 780*a468 + 255*a469 - 510*a471) + (a243 + a246 + a257 + a427 - 690*a468 - 270*a469 + 90*a470 + 440*a472 + 345*a473 + a476)*X[2];
        P[102] = a138*(-a262*X[1] + a262 - 220*a428 + 200*a430 + 200*a486) + a320*(-a274 + 100*a428) + a56*(-a258*a343 + 240*a259*a4 + 24*a259 - a31*a487 + 144*a428 + 120*a431 - 264*a486 - a487 + 120*a488) + (a259*a484 + a262*a83*X[0] - a265 - a268 + 60*a272*a4 + 6*a272 - a430*a67 + a434 + 72*a486 - 66*a488)*X[2];
        P[103] = a138*(a438 + a490*X[0] - a490 + 90*a491) + a320*(45*a277 + a489*X[0] - a489) + a56*(54*a289 + a4*a492 + a444 - 114*a491 + a492 - 132*a493 + 114*a494) + (a20*a282 + a276 - a277*a445 + a283 + a288*a4 + a299*X[0] + a300 + a446 + 36*a493 - 60*a494)*X[2];
        P[104] = a138*(a309 - a312 - 180*a385 + a496 + a497 + a498) + a171*std::pow(X[2], 5) + a320*(a495*X[0] + a495*X[1] - a495) + a56*(a164*a344 + a20*a232 + a232*a83 - a232 + a234*a344 + a309*X[1] + a496*X[1] - a496 - a497 - a498) + (-a174*a4 + a174*X[0] + a20*a306 + a302 + a307 + a311*a4 + a322*X[0] + a323 + a447 + a451 - 36*a474)*X[2];
        return P;
    }
};

/**
 * Divergence free polynomial basis on reference line
 */

template <>
class DivergenceFreePolynomialBasis<1, 1>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 1;

    Matrix<Size, Dims> eval(Vector<1> const& X) const 
    {
        Matrix<Size, Dims> P;
        P[0] = 1;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<1, 2>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 1;

    Matrix<Size, Dims> eval(Vector<1> const& X) const 
    {
        Matrix<Size, Dims> P;
        P[0] = 1;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<1, 3>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 1;

    Matrix<Size, Dims> eval(Vector<1> const& X) const 
    {
        Matrix<Size, Dims> P;
        P[0] = 1;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<1, 4>
{
  public:
    inline static constexpr std::size_t Dims = 1;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 1;

    Matrix<Size, Dims> eval(Vector<1> const& X) const 
    {
        Matrix<Size, Dims> P;
        P[0] = 1;
        return P;
    }
};

/**
 * Divergence free polynomial basis on reference triangle
 */

template <>
class DivergenceFreePolynomialBasis<2, 1>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 5;

    Matrix<Size, Dims> eval(Vector<2> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = std::numbers::sqrt3_v<Scalar>;
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = a0;
        P[4] = 0;
        P[5] = 6*X[0] - 2;
        P[6] = a1*(4*X[1] - 4.0/3.0);
        P[7] = 0;
        P[8] = a1*(4.0/3.0 - 4*X[0]);
        P[9] = 2*a1*(X[0] + 2*X[1] - 1);
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<2, 2>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 9;

    Matrix<Size, Dims> eval(Vector<2> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = X[0]*X[0];
        Scalar const a4 = 30*a3;
        Scalar const a5 = -6*X[1];
        Scalar const a6 = X[0]*X[1];
        Scalar const a7 = std::sqrt(30);
        Scalar const a8 = X[1]*X[1];
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = a0;
        P[4] = 0;
        P[5] = a1 - 2;
        P[6] = a2*(4*X[1] - 4.0/3.0);
        P[7] = 0;
        P[8] = a2*(4.0/3.0 - 4*X[0]);
        P[9] = 2*a2*(X[0] + 2*X[1] - 1);
        P[10] = 0;
        P[11] = std::sqrt(6)*(10*a3 - 8*X[0] + 1);
        P[12] = (1.0/2.0)*a0*(-a4 + 12*X[0] + 1);
        P[13] = a0*(15*a3 + a5 + 30*a6 - 18*X[0] + 3);
        P[14] = (3.0/5.0)*a7*(10*a8 - 8*X[1] + 1);
        P[15] = 0;
        P[16] = (1.0/10.0)*a7*(-a4 - 120*a6 + 60*X[0] + 24*X[1] - 13);
        P[17] = a7*(a1*X[1] + a3 + a5 + 6*a8 - 2*X[0] + 1);
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<2, 3>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 14;

    Matrix<Size, Dims> eval(Vector<2> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = 4*X[1];
        Scalar const a4 = 4*X[0];
        Scalar const a5 = std::sqrt(6);
        Scalar const a6 = X[0]*X[0];
        Scalar const a7 = 30*a6;
        Scalar const a8 = -a7;
        Scalar const a9 = 18*X[0];
        Scalar const a10 = -6*X[1];
        Scalar const a11 = 30*X[0];
        Scalar const a12 = std::sqrt(30);
        Scalar const a13 = X[1]*X[1];
        Scalar const a14 = X[0]*X[1];
        Scalar const a15 = X[0]*X[0]*X[0];
        Scalar const a16 = a6*X[1];
        Scalar const a17 = 84*a16 - 2;
        Scalar const a18 = std::sqrt(10);
        Scalar const a19 = 210*a15;
        Scalar const a20 = 12*X[1];
        Scalar const a21 = a13*X[0];
        Scalar const a22 = std::sqrt(14);
        Scalar const a23 = X[1]*X[1]*X[1];
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = a0;
        P[4] = 0;
        P[5] = a1 - 2;
        P[6] = a2*(a3 - 4.0/3.0);
        P[7] = 0;
        P[8] = a2*(4.0/3.0 - a4);
        P[9] = 2*a2*(X[0] + 2*X[1] - 1);
        P[10] = 0;
        P[11] = a5*(10*a6 - 8*X[0] + 1);
        P[12] = (1.0/2.0)*a0*(a8 + 12*X[0] + 1);
        P[13] = a0*(a10 + a11*X[1] + 15*a6 - a9 + 3);
        P[14] = (3.0/5.0)*a12*(10*a13 - 8*X[1] + 1);
        P[15] = 0;
        P[16] = (1.0/10.0)*a12*(-120*a14 - a7 + 60*X[0] + 24*X[1] - 13);
        P[17] = a12*(a1*X[1] + a10 + 6*a13 + a6 - 2*X[0] + 1);
        P[18] = 0;
        P[19] = a0*(a11 + 70*a15 - 90*a6 - 2);
        P[20] = a5*(-28*a15 - a4 + 24*a6 + 2.0/15.0);
        P[21] = a5*(-48*a14 + 42*a15 + a17 + a3 - 66*a6 + 26*X[0]);
        P[22] = (2.0/15.0)*a18*(-a10 - a19 - 630*a6*X[1] + 360*a6 + 180*X[0]*X[1] - 90*X[0] - 5);
        P[23] = a18*(-12*a13 - 96*a14 + 14*a15 + a17 + a20 + 84*a21 + a8 + a9);
        P[24] = (8.0/7.0)*a22*(-45*a13 + 35*a23 + 15*X[1] - 1);
        P[25] = 0;
        P[26] = (4.0/105.0)*a22*(450*a13 - 1575*a16 - a19 - 3150*a21 + 630*a6 + 3150*X[0]*X[1] - 630*X[0] - 465*X[1] + 101);
        P[27] = 2*a22*(a11*a13 - 30*a13 + a15 + a20*a6 + a20 + 20*a23 - 3*a6 - 24*X[0]*X[1] + 3*X[0] - 1);
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<2, 4>
{
  public:
    inline static constexpr std::size_t Dims = 2;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 20;

    Matrix<Size, Dims> eval(Vector<2> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6*X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = 4*X[1];
        Scalar const a4 = 4*X[0];
        Scalar const a5 = 2*X[1];
        Scalar const a6 = std::sqrt(6);
        Scalar const a7 = X[0]*X[0];
        Scalar const a8 = 12*X[0];
        Scalar const a9 = 30*a7;
        Scalar const a10 = -a9;
        Scalar const a11 = 18*X[0];
        Scalar const a12 = -6*X[1];
        Scalar const a13 = 30*X[0];
        Scalar const a14 = std::sqrt(30);
        Scalar const a15 = X[1]*X[1];
        Scalar const a16 = 24*X[1];
        Scalar const a17 = -a16;
        Scalar const a18 = X[0]*X[1];
        Scalar const a19 = 90*a7;
        Scalar const a20 = X[0]*X[0]*X[0];
        Scalar const a21 = 28*a20;
        Scalar const a22 = a7*X[1];
        Scalar const a23 = 84*a22 - 2;
        Scalar const a24 = std::sqrt(10);
        Scalar const a25 = 90*X[0];
        Scalar const a26 = 210*a20;
        Scalar const a27 = 630*a7;
        Scalar const a28 = 12*X[1];
        Scalar const a29 = a15*X[0];
        Scalar const a30 = std::sqrt(14);
        Scalar const a31 = X[1]*X[1]*X[1];
        Scalar const a32 = -a27;
        Scalar const a33 = -450*a15;
        Scalar const a34 = 30*a15;
        Scalar const a35 = 20*a31;
        Scalar const a36 = X[0]*X[0]*X[0]*X[0];
        Scalar const a37 = 60*X[0];
        Scalar const a38 = 1260*a36;
        Scalar const a39 = 168*a20;
        Scalar const a40 = 11340*a36;
        Scalar const a41 = a20*X[1];
        Scalar const a42 = a15*a7;
        Scalar const a43 = std::sqrt(70);
        Scalar const a44 = -5040*X[0];
        Scalar const a45 = a31*X[0];
        Scalar const a46 = X[1]*X[1]*X[1]*X[1];
        Scalar const a47 = 20*X[1];
        Scalar const a48 = 140*a31;
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = a0;
        P[4] = 0;
        P[5] = a1 - 2;
        P[6] = a2*(a3 - 4.0/3.0);
        P[7] = 0;
        P[8] = a2*(4.0/3.0 - a4);
        P[9] = 2*a2*(a5 + X[0] - 1);
        P[10] = 0;
        P[11] = a6*(10*a7 - 8*X[0] + 1);
        P[12] = (1.0/2.0)*a0*(a10 + a8 + 1);
        P[13] = a0*(-a11 + a12 + a13*X[1] + 15*a7 + 3);
        P[14] = (3.0/5.0)*a14*(10*a15 - 8*X[1] + 1);
        P[15] = 0;
        P[16] = (1.0/10.0)*a14*(-a17 - 120*a18 - a9 + 60*X[0] - 13);
        P[17] = a14*(a1*X[1] + a12 + 6*a15 + a7 - 2*X[0] + 1);
        P[18] = 0;
        P[19] = a0*(a13 - a19 + 70*a20 - 2);
        P[20] = a6*(-a21 - a4 + 24*a7 + 2.0/15.0);
        P[21] = a6*(-48*a18 + 42*a20 + a23 + a3 - 66*a7 + 26*X[0]);
        P[22] = (2.0/15.0)*a24*(-a12 - a25 - a26 - a27*X[1] + 360*a7 + 180*X[0]*X[1] - 5);
        P[23] = a24*(a10 + a11 - 12*a15 - 96*a18 + 14*a20 + a23 + a28 + 84*a29);
        P[24] = (8.0/7.0)*a30*(-45*a15 + 35*a31 + 15*X[1] - 1);
        P[25] = 0;
        P[26] = (4.0/105.0)*a30*(-1575*a22 - a26 - 3150*a29 - a32 - a33 + 3150*X[0]*X[1] - 630*X[0] - 465*X[1] + 101);
        P[27] = 2*a30*(a13*a15 - a16*X[0] + a20 + a28*a7 + a28 - a34 + a35 - 3*a7 + 3*X[0] - 1);
        P[28] = 0;
        P[29] = a24*(-224*a20 + 126*a36 + 126*a7 - 24*X[0] + 1);
        P[30] = (1.0/30.0)*a14*(1680*a20 + a32 + a37 - a38 + 1);
        P[31] = a14*(42*a18 - 168*a22 + 84*a36 + a39*X[1] - a39 - a5 + 105*a7 - 22*X[0] + 1);
        P[32] = (1.0/42.0)*a0*(-a17 - 2520*a18 + 21840*a20 - a40 - 30240*a41 + 20160*a7*X[1] - 10710*a7 + 1260*X[0] - 29);
        P[33] = a0*(510*a18 - 440*a20 - 1560*a22 - a25 - 480*a29 + a34 + 180*a36 + 1080*a41 + 1080*a42 + 345*a7 - 30*X[1] + 5);
        P[34] = (1.0/420.0)*a43*(25200*a15*X[0] - 25200*a18 + 31920*a20 - a33 - a40 - 75600*a41 - 113400*a42 - a44 + 126000*a7*X[1] - 27720*a7 - 600*X[1] + 209);
        P[35] = a43*(132*a18 - a21 - 228*a22 - a28 - 300*a29 + a34 - a35 + 9*a36 + 108*a41 + 270*a42 + 180*a45 - a8 + a9 + 1);
        P[36] = (5.0/3.0)*a24*(126*a15 + a17 - 224*a31 + 126*a46 + 1);
        P[37] = 0;
        P[38] = (1.0/84.0)*a24*(105840*a15*X[0] - 11970*a15 - 45360*a18 + 5040*a20 + 7840*a31 - a38 - 15120*a41 - 52920*a42 - a44 - 70560*a45 + 45360*a7*X[1] - 7560*a7 + 5304*X[1] - 641);
        P[39] = 3*a24*(a15*a19 + 90*a15 + a20*a47 - 4*a20 - 60*a22 - 180*a29 + a36 + a37*X[1] - a4 + 70*a46 - a47 + a48*X[0] - a48 + 6*a7 + 1);
        return P;
    }
};

/**
 * Divergence free polynomial basis on reference tetrahedron
 */

template <>
class DivergenceFreePolynomialBasis<3, 1>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 1;
    inline static constexpr std::size_t Size = 11;

    Matrix<Size, Dims> eval(Vector<3> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = 4*X[0] - 1;
        Scalar const a2 = std::sqrt(10)*a1;
        Scalar const a3 = std::sqrt(5);
        Scalar const a4 = X[0] - 1;
        Scalar const a5 = 2*a3*(a4 + 3*X[1]);
        Scalar const a6 = std::sqrt(15);
        Scalar const a7 = -a1*a6;
        Scalar const a8 = 2*a6*(a4 + X[1] + 2*X[2]);
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = a0;
        P[5] = 0;
        P[6] = 0;
        P[7] = 0;
        P[8] = a0;
        P[9] = 0;
        P[10] = a2;
        P[11] = 0;
        P[12] = 0;
        P[13] = 0;
        P[14] = a2;
        P[15] = a3*(6*X[1] - 3.0/2.0);
        P[16] = 0;
        P[17] = 0;
        P[18] = a3*(3.0/2.0 - 6*X[0]);
        P[19] = a5;
        P[20] = 0;
        P[21] = 0;
        P[22] = 0;
        P[23] = a5;
        P[24] = (1.0/2.0)*a6*(4*X[1] + 8*X[2] - 3);
        P[25] = 0;
        P[26] = 0;
        P[27] = (1.0/2.0)*a7;
        P[28] = a8;
        P[29] = 0;
        P[30] = a7;
        P[31] = 0;
        P[32] = a8;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<3, 2>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 2;
    inline static constexpr std::size_t Size = 26;

    Matrix<Size, Dims> eval(Vector<3> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = 4*X[0] - 1;
        Scalar const a2 = std::sqrt(10)*a1;
        Scalar const a3 = std::sqrt(5);
        Scalar const a4 = 6*X[1];
        Scalar const a5 = 6*X[0];
        Scalar const a6 = X[0] - 1;
        Scalar const a7 = 2*a3*(a6 + 3*X[1]);
        Scalar const a8 = std::sqrt(15);
        Scalar const a9 = -a1*a8;
        Scalar const a10 = 2*X[2];
        Scalar const a11 = 2*a8*(a10 + a6 + X[1]);
        Scalar const a12 = std::sqrt(14);
        Scalar const a13 = X[0]*X[0];
        Scalar const a14 = a12*(15*a13 - 10*X[0] + 1);
        Scalar const a15 = std::sqrt(7);
        Scalar const a16 = -a4;
        Scalar const a17 = X[0]*X[1];
        Scalar const a18 = a15*(12*a13 + a16 + 36*a17 - 14*X[0] + 2);
        Scalar const a19 = std::sqrt(42);
        Scalar const a20 = 10*X[1];
        Scalar const a21 = X[1]*X[1];
        Scalar const a22 = 15*a21;
        Scalar const a23 = 600*a17 - 100*X[1];
        Scalar const a24 = 8*X[1];
        Scalar const a25 = 2*X[0];
        Scalar const a26 = a13 - a25;
        Scalar const a27 = a26 + 1;
        Scalar const a28 = a19*(10*a21 + a24*X[0] - a24 + a27);
        Scalar const a29 = std::sqrt(21);
        Scalar const a30 = 60*a13;
        Scalar const a31 = a29*(-a30 + 20*X[0] + 1);
        Scalar const a32 = X[0]*X[2];
        Scalar const a33 = -a10 + a4*X[0];
        Scalar const a34 = 12*a32 + a33 - X[1];
        Scalar const a35 = 2*a29*(6*a13 + a34 - 7*X[0] + 1);
        Scalar const a36 = X[1]*X[2];
        Scalar const a37 = 3*a12*(a10*X[0] + a16 + a20*X[2] + 5*a21 + a27 + a33);
        Scalar const a38 = std::sqrt(210);
        Scalar const a39 = X[2]*X[2];
        Scalar const a40 = a25*X[1] + a5*X[2];
        Scalar const a41 = a38*(a21 + a27 + 6*a39 + a4*X[2] + a40 - 2*X[1] - 6*X[2]);
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = a0;
        P[5] = 0;
        P[6] = 0;
        P[7] = 0;
        P[8] = a0;
        P[9] = 0;
        P[10] = a2;
        P[11] = 0;
        P[12] = 0;
        P[13] = 0;
        P[14] = a2;
        P[15] = a3*(a4 - 3.0/2.0);
        P[16] = 0;
        P[17] = 0;
        P[18] = a3*(3.0/2.0 - a5);
        P[19] = a7;
        P[20] = 0;
        P[21] = 0;
        P[22] = 0;
        P[23] = a7;
        P[24] = (1.0/2.0)*a8*(4*X[1] + 8*X[2] - 3);
        P[25] = 0;
        P[26] = 0;
        P[27] = (1.0/2.0)*a9;
        P[28] = a11;
        P[29] = 0;
        P[30] = a9;
        P[31] = 0;
        P[32] = a11;
        P[33] = 0;
        P[34] = a14;
        P[35] = 0;
        P[36] = 0;
        P[37] = 0;
        P[38] = a14;
        P[39] = a15*(-18*a13 + a5 + 3.0/10.0);
        P[40] = a18;
        P[41] = 0;
        P[42] = 0;
        P[43] = 0;
        P[44] = a18;
        P[45] = (2.0/3.0)*a19*(-a20 + a22 + 1);
        P[46] = 0;
        P[47] = 0;
        P[48] = (1.0/30.0)*a19*(-120*a13 - a23 + 240*X[0] - 43);
        P[49] = a28;
        P[50] = 0;
        P[51] = 0;
        P[52] = 0;
        P[53] = a28;
        P[54] = (1.0/10.0)*a31;
        P[55] = a35;
        P[56] = 0;
        P[57] = (1.0/5.0)*a31;
        P[58] = 0;
        P[59] = a35;
        P[60] = a12*(a22 + 30*a36 - 15*X[1] - 5*X[2] + 2);
        P[61] = 0;
        P[62] = 0;
        P[63] = (1.0/10.0)*a12*(-90*a13 - 300*a17 - 300*a32 + 180*X[0] + 50*X[1] + 50*X[2] - 31);
        P[64] = a37;
        P[65] = 0;
        P[66] = (1.0/20.0)*a12*(-a23 - a30 + 120*X[0] - 19);
        P[67] = 0;
        P[68] = a37;
        P[69] = (1.0/3.0)*a38*(3*a21 + 18*a36 + 18*a39 - 5*X[1] - 15*X[2] + 2);
        P[70] = 0;
        P[71] = 0;
        P[72] = a38*(-a26 - a40 + (1.0/3.0)*X[1] + X[2] - 1.0/3.0);
        P[73] = a41;
        P[74] = 0;
        P[75] = a38*(-3*a13 - a34 + a5 - 21.0/20.0);
        P[76] = 0;
        P[77] = a41;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<3, 3>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 3;
    inline static constexpr std::size_t Size = 50;

    Matrix<Size, Dims> eval(Vector<3> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = 4*X[0];
        Scalar const a3 = a2 - 1;
        Scalar const a4 = a1*a3;
        Scalar const a5 = std::sqrt(5);
        Scalar const a6 = 6*X[1];
        Scalar const a7 = 6*X[0];
        Scalar const a8 = 3*X[1];
        Scalar const a9 = X[0] - 1;
        Scalar const a10 = 2*a5*(a8 + a9);
        Scalar const a11 = std::sqrt(15);
        Scalar const a12 = -a11*a3;
        Scalar const a13 = 2*X[2];
        Scalar const a14 = a13 + X[1];
        Scalar const a15 = 2*a11*(a14 + a9);
        Scalar const a16 = std::sqrt(14);
        Scalar const a17 = 10*X[0];
        Scalar const a18 = X[0]*X[0];
        Scalar const a19 = 15*a18;
        Scalar const a20 = a16*(-a17 + a19 + 1);
        Scalar const a21 = std::sqrt(7);
        Scalar const a22 = 14*X[0];
        Scalar const a23 = -a6;
        Scalar const a24 = 12*a18;
        Scalar const a25 = X[0]*X[1];
        Scalar const a26 = a21*(-a22 + a23 + a24 + 36*a25 + 2);
        Scalar const a27 = std::sqrt(42);
        Scalar const a28 = 10*X[1];
        Scalar const a29 = X[1]*X[1];
        Scalar const a30 = 15*a29;
        Scalar const a31 = -100*X[1];
        Scalar const a32 = 600*a25 + a31;
        Scalar const a33 = 8*X[1];
        Scalar const a34 = 2*X[0];
        Scalar const a35 = a18 - a34;
        Scalar const a36 = a35 + 1;
        Scalar const a37 = a27*(10*a29 + a33*X[0] - a33 + a36);
        Scalar const a38 = std::sqrt(21);
        Scalar const a39 = 60*a18;
        Scalar const a40 = a38*(-a39 + 20*X[0] + 1);
        Scalar const a41 = 6*a18;
        Scalar const a42 = -X[1];
        Scalar const a43 = a42 + 1;
        Scalar const a44 = 12*X[2];
        Scalar const a45 = a44*X[0];
        Scalar const a46 = a6*X[0];
        Scalar const a47 = -a13 + a46;
        Scalar const a48 = a45 + a47;
        Scalar const a49 = 2*a38*(a41 + a43 + a48 - 7*X[0]);
        Scalar const a50 = 15*X[1];
        Scalar const a51 = X[1]*X[2];
        Scalar const a52 = X[0]*X[2];
        Scalar const a53 = -50*X[1] - 50*X[2];
        Scalar const a54 = 3*a16*(a13*X[0] + a23 + a28*X[2] + 5*a29 + a36 + a47);
        Scalar const a55 = std::sqrt(210);
        Scalar const a56 = 3*a29;
        Scalar const a57 = X[2]*X[2];
        Scalar const a58 = 18*X[1];
        Scalar const a59 = a34*X[1] + a7*X[2];
        Scalar const a60 = 6*X[2];
        Scalar const a61 = 2*X[1];
        Scalar const a62 = 6*a57;
        Scalar const a63 = a6*X[2];
        Scalar const a64 = a55*(a29 + a36 + a59 - a60 - a61 + a62 + a63);
        Scalar const a65 = 3*a18;
        Scalar const a66 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a67 = 54*X[0];
        Scalar const a68 = -189*a18;
        Scalar const a69 = X[0]*X[0]*X[0];
        Scalar const a70 = 168*a69;
        Scalar const a71 = a66*(a67 + a68 + a70 - 3);
        Scalar const a72 = 18*X[0];
        Scalar const a73 = 90*X[0];
        Scalar const a74 = 504*a18;
        Scalar const a75 = -252*a18 - 252*a25 + a58 + a70 + a73 + a74*X[1] - 6;
        Scalar const a76 = 1680*X[0];
        Scalar const a77 = a18*X[1];
        Scalar const a78 = a31 + 16800*a77 - 4200*X[0]*X[1];
        Scalar const a79 = (1.0/70.0)*a0;
        Scalar const a80 = 30*X[0];
        Scalar const a81 = 24*X[1];
        Scalar const a82 = 240*X[0];
        Scalar const a83 = a0*(-51*a18 - 216*a25 + a29*a82 - 30*a29 + 24*a69 + 192*a77 + a80 + a81 - 3);
        Scalar const a84 = X[1]*X[1]*X[1];
        Scalar const a85 = 56*a84;
        Scalar const a86 = a29*X[0];
        Scalar const a87 = 840*X[0];
        Scalar const a88 = -840*a18 + 280*a69 + a87;
        Scalar const a89 = -735*a29 + 5880*a86 + a88;
        Scalar const a90 = 45*a29;
        Scalar const a91 = -a90;
        Scalar const a92 = -a65 + a69 + 3*X[0] - 1;
        Scalar const a93 = 6*a66*(a19*X[1] + a50 - a80*X[1] + 35*a84 + a90*X[0] + a91 + a92);
        Scalar const a94 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a95 = 560*a69;
        Scalar const a96 = a94*(420*a18 - a95 - 60*X[0] + 1);
        Scalar const a97 = a18*X[2];
        Scalar const a98 = 6*a94*(a14 - 42*a18 - a22*X[1] - 28*a52 + 28*a69 + 28*a77 + 56*a97 + 15*X[0] - 1);
        Scalar const a99 = 54*X[1];
        Scalar const a100 = a25*X[2];
        Scalar const a101 = a66*(720*a100 - 153*a18 - 486*a25 - 90*a51 - 162*a52 + 72*a69 + a73 + 432*a77 + 360*a86 + a91 + 144*a97 + a99 + 18*X[2] - 9);
        Scalar const a102 = -3780*a18 + 2240*a69 + a87;
        Scalar const a103 = -168*a51;
        Scalar const a104 = a29*X[2];
        Scalar const a105 = 336*a104 - 5;
        Scalar const a106 = 33*a29;
        Scalar const a107 = 13*X[1];
        Scalar const a108 = 42*a104 + a92;
        Scalar const a109 = -a81*X[2];
        Scalar const a110 = a109 + a52*a81;
        Scalar const a111 = 6*a0*(a106*X[0] - a106 + a107*a18 + a107 + a108 + a110 + a13*a18 + a13 - a2*X[2] - 26*a25 + 21*a84);
        Scalar const a112 = std::sqrt(30);
        Scalar const a113 = 8*a69;
        Scalar const a114 = a57*X[0];
        Scalar const a115 = 48*a25*X[2] + a60 - a62 - a72*X[1];
        Scalar const a116 = 3*a112*(a113 + 48*a114 + a115 + a17 - 17*a18 - a29 + a61 - a63 - a67*X[2] + 16*a77 + 8*a86 + 48*a97 - 1);
        Scalar const a117 = 3360*a77 - 1680*X[0]*X[2];
        Scalar const a118 = 42*a57;
        Scalar const a119 = a57*X[1];
        Scalar const a120 = 840*a18;
        Scalar const a121 = 9*X[1];
        Scalar const a122 = 6*a1*(a108 + a115 + a118*X[1] + a121*a18 + a121 + a30*X[0] - a30 + a41*X[2] - a45 - 48*a51 + a57*a7 + 7*a84);
        Scalar const a123 = X[2]*X[2]*X[2];
        Scalar const a124 = (3.0/4.0)*a16;
        Scalar const a125 = 24*X[0];
        Scalar const a126 = 30*a57;
        Scalar const a127 = -a126 - a56;
        Scalar const a128 = 6*a16*(a110 + 20*a123 - a125*X[2] + a126*X[1] + a127 + a18*a8 + a24*X[2] + a29*a44 + a44 - a46 + a56*X[0] + a57*a80 + a8 + a84 + a92);
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = a0;
        P[5] = 0;
        P[6] = 0;
        P[7] = 0;
        P[8] = a0;
        P[9] = 0;
        P[10] = a4;
        P[11] = 0;
        P[12] = 0;
        P[13] = 0;
        P[14] = a4;
        P[15] = a5*(a6 - 3.0/2.0);
        P[16] = 0;
        P[17] = 0;
        P[18] = a5*(3.0/2.0 - a7);
        P[19] = a10;
        P[20] = 0;
        P[21] = 0;
        P[22] = 0;
        P[23] = a10;
        P[24] = (1.0/2.0)*a11*(4*X[1] + 8*X[2] - 3);
        P[25] = 0;
        P[26] = 0;
        P[27] = (1.0/2.0)*a12;
        P[28] = a15;
        P[29] = 0;
        P[30] = a12;
        P[31] = 0;
        P[32] = a15;
        P[33] = 0;
        P[34] = a20;
        P[35] = 0;
        P[36] = 0;
        P[37] = 0;
        P[38] = a20;
        P[39] = a21*(-18*a18 + a7 + 3.0/10.0);
        P[40] = a26;
        P[41] = 0;
        P[42] = 0;
        P[43] = 0;
        P[44] = a26;
        P[45] = (2.0/3.0)*a27*(-a28 + a30 + 1);
        P[46] = 0;
        P[47] = 0;
        P[48] = (1.0/30.0)*a27*(-120*a18 - a32 + 240*X[0] - 43);
        P[49] = a37;
        P[50] = 0;
        P[51] = 0;
        P[52] = 0;
        P[53] = a37;
        P[54] = (1.0/10.0)*a40;
        P[55] = a49;
        P[56] = 0;
        P[57] = (1.0/5.0)*a40;
        P[58] = 0;
        P[59] = a49;
        P[60] = a16*(a30 - a50 + 30*a51 - 5*X[2] + 2);
        P[61] = 0;
        P[62] = 0;
        P[63] = (1.0/10.0)*a16*(-90*a18 - 300*a25 - 300*a52 - a53 + 180*X[0] - 31);
        P[64] = a54;
        P[65] = 0;
        P[66] = (1.0/20.0)*a16*(-a32 - a39 + 120*X[0] - 19);
        P[67] = 0;
        P[68] = a54;
        P[69] = (1.0/3.0)*a55*(a56 + 18*a57 + a58*X[2] - 5*X[1] - 15*X[2] + 2);
        P[70] = 0;
        P[71] = 0;
        P[72] = a55*(-a35 - a59 + (1.0/3.0)*X[1] + X[2] - 1.0/3.0);
        P[73] = a64;
        P[74] = 0;
        P[75] = a55*(-a42 - a48 - a65 + a7 - 21.0/20.0);
        P[76] = 0;
        P[77] = a64;
        P[78] = 0;
        P[79] = a71;
        P[80] = 0;
        P[81] = 0;
        P[82] = 0;
        P[83] = a71;
        P[84] = 126*a18 - a70 - a72 + 3.0/10.0;
        P[85] = a75;
        P[86] = 0;
        P[87] = 0;
        P[88] = 0;
        P[89] = a75;
        P[90] = a79*(7560*a18 - 4480*a69 - a76 - a78 - 67);
        P[91] = a83;
        P[92] = 0;
        P[93] = 0;
        P[94] = 0;
        P[95] = a83;
        P[96] = (15.0/4.0)*a66*(-63*a29 + a58 + a85 - 1);
        P[97] = 0;
        P[98] = 0;
        P[99] = (3.0/28.0)*a66*(-2520*a77 - a89 + 5040*X[0]*X[1] - 650*X[1] + 117);
        P[100] = a93;
        P[101] = 0;
        P[102] = 0;
        P[103] = 0;
        P[104] = a93;
        P[105] = (1.0/10.0)*a96;
        P[106] = a98;
        P[107] = 0;
        P[108] = (1.0/5.0)*a96;
        P[109] = 0;
        P[110] = a98;
        P[111] = (3.0/70.0)*a66*(5670*a18 - a53 - 3360*a69 - 8400*a77 - 8400*a97 + 2100*X[0]*X[1] + 2100*X[0]*X[2] - 1260*X[0] - 39);
        P[112] = a101;
        P[113] = 0;
        P[114] = (3.0/140.0)*a66*(-a102 - a78 - 11);
        P[115] = 0;
        P[116] = a101;
        P[117] = (3.0/4.0)*a0*(a103 + a105 - 231*a29 + a44 + 168*a84 + 78*X[1]);
        P[118] = 0;
        P[119] = 0;
        P[120] = (1.0/28.0)*a0*(-14112*a100 + 2184*a18 + 1323*a29 - 728*a69 - 5544*a77 - 10584*a86 - 2016*a97 + 11088*X[0]*X[1] + 4032*X[0]*X[2] - 2184*X[0] + 1764*X[1]*X[2] - 1416*X[1] - 534*X[2] + 295);
        P[121] = a111;
        P[122] = 0;
        P[123] = a79*(2205*a29 - 5040*a77 - 17640*a86 - a88 + 10080*X[0]*X[1] - 1230*X[1] + 101);
        P[124] = 0;
        P[125] = a111;
        P[126] = (1.0/7.0)*a112*(-a43 - a68 - 112*a69 - a74*X[2] - 168*a77 + 42*X[0]*X[1] + 126*X[0]*X[2] - 42*X[0] + 3*X[2]);
        P[127] = a116;
        P[128] = 0;
        P[129] = (3.0/140.0)*a112*(-a102 - a117 - 6720*a97 + 840*X[0]*X[1] + 20*X[1] + 40*X[2] - 29);
        P[130] = 0;
        P[131] = a116;
        P[132] = (3.0/4.0)*a1*(a105 - a118 + 336*a119 - 105*a29 - 336*a51 + a85 + a99 + 36*X[2]);
        P[133] = 0;
        P[134] = 0;
        P[135] = (3.0/28.0)*a1*(-4704*a100 - 2352*a114 - a120*X[1] + 504*a18 + 147*a29 + 294*a57 - a70 - 1176*a86 - 1344*a97 + 1680*X[0]*X[1] + 2688*X[0]*X[2] - 504*X[0] + 588*X[1]*X[2] - 212*X[1] - 342*X[2] + 65);
        P[136] = a122;
        P[137] = 0;
        P[138] = (3.0/70.0)*a1*(-11760*a100 - a117 - a120*X[2] - a89 + 6720*X[0]*X[1] + 1470*X[1]*X[2] - 855*X[1] - 205*X[2] + 104);
        P[139] = 0;
        P[140] = a122;
        P[141] = a124*(a103 + 96*a104 + 240*a119 + 160*a123 - 21*a29 - 210*a57 + a58 + 8*a84 + 72*X[2] - 5);
        P[142] = 0;
        P[143] = 0;
        P[144] = a124*(-192*a100 - a109 - a113 - a125*a29 - a125 - a127 - a18*a81 + 24*a18 - a57*a82 - a6 - 96*a97 + 48*X[0]*X[1] + 192*X[0]*X[2] - 24*X[2] + 3);
        P[145] = a128;
        P[146] = 0;
        P[147] = (3.0/70.0)*a16*(-8400*a100 - 8400*a114 - 1680*a18*X[1] + 1680*a18 - a29*a76 + 210*a29 + 1050*a57 - a76 - a95 - 4200*a97 + 3360*X[0]*X[1] + 8400*X[0]*X[2] + 1050*X[1]*X[2] - 435*X[1] - 1075*X[2] + 227);
        P[148] = 0;
        P[149] = a128;
        return P;
    }
};

template <>
class DivergenceFreePolynomialBasis<3, 4>
{
  public:
    inline static constexpr std::size_t Dims = 3;
    inline static constexpr std::size_t Order = 4;
    inline static constexpr std::size_t Size = 85;

    Matrix<Size, Dims> eval(Vector<3> const& X) const 
    {
        Matrix<Size, Dims> P;
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = std::sqrt(10);
        Scalar const a2 = 4*X[0];
        Scalar const a3 = a2 - 1;
        Scalar const a4 = a1*a3;
        Scalar const a5 = std::sqrt(5);
        Scalar const a6 = 6*X[1];
        Scalar const a7 = 6*X[0];
        Scalar const a8 = 3*X[1];
        Scalar const a9 = X[0] - 1;
        Scalar const a10 = 2*a5*(a8 + a9);
        Scalar const a11 = std::sqrt(15);
        Scalar const a12 = 4*X[1];
        Scalar const a13 = 8*X[2];
        Scalar const a14 = -a11*a3;
        Scalar const a15 = 2*X[2];
        Scalar const a16 = a15 + X[1];
        Scalar const a17 = 2*a11*(a16 + a9);
        Scalar const a18 = std::sqrt(14);
        Scalar const a19 = 10*X[0];
        Scalar const a20 = X[0]*X[0];
        Scalar const a21 = 15*a20;
        Scalar const a22 = a18*(-a19 + a21 + 1);
        Scalar const a23 = std::sqrt(7);
        Scalar const a24 = 18*a20;
        Scalar const a25 = 14*X[0];
        Scalar const a26 = 12*a20;
        Scalar const a27 = X[0]*X[1];
        Scalar const a28 = 36*a27;
        Scalar const a29 = -a6;
        Scalar const a30 = a29 + 2;
        Scalar const a31 = a23*(-a25 + a26 + a28 + a30);
        Scalar const a32 = std::sqrt(42);
        Scalar const a33 = 10*X[1];
        Scalar const a34 = X[1]*X[1];
        Scalar const a35 = 15*a34;
        Scalar const a36 = 120*a20;
        Scalar const a37 = -100*X[1];
        Scalar const a38 = 600*a27 + a37;
        Scalar const a39 = 8*X[1];
        Scalar const a40 = 2*X[0];
        Scalar const a41 = a20 - a40;
        Scalar const a42 = a41 + 1;
        Scalar const a43 = 10*a34 - a39;
        Scalar const a44 = a32*(a39*X[0] + a42 + a43);
        Scalar const a45 = std::sqrt(21);
        Scalar const a46 = 20*X[0];
        Scalar const a47 = 60*a20;
        Scalar const a48 = a45*(a46 - a47 + 1);
        Scalar const a49 = 6*a20;
        Scalar const a50 = -X[1];
        Scalar const a51 = a50 + 1;
        Scalar const a52 = 12*X[2];
        Scalar const a53 = a52*X[0];
        Scalar const a54 = -a15;
        Scalar const a55 = a6*X[0];
        Scalar const a56 = a54 + a55;
        Scalar const a57 = a53 + a56;
        Scalar const a58 = 2*a45*(a49 + a51 + a57 - 7*X[0]);
        Scalar const a59 = 15*X[1];
        Scalar const a60 = 30*X[1];
        Scalar const a61 = a35 + a60*X[2];
        Scalar const a62 = 90*a20;
        Scalar const a63 = 300*a27;
        Scalar const a64 = X[0]*X[2];
        Scalar const a65 = 300*a64;
        Scalar const a66 = -50*X[1] - 50*X[2];
        Scalar const a67 = 3*a18*(a15*X[0] + a29 + a33*X[2] + 5*a34 + a42 + a56);
        Scalar const a68 = std::sqrt(210);
        Scalar const a69 = 3*a34;
        Scalar const a70 = X[2]*X[2];
        Scalar const a71 = 18*X[1];
        Scalar const a72 = a7*X[2];
        Scalar const a73 = a40*X[1] + a72;
        Scalar const a74 = 2*X[1];
        Scalar const a75 = a6*X[2];
        Scalar const a76 = 6*X[2];
        Scalar const a77 = -a76;
        Scalar const a78 = 6*a70;
        Scalar const a79 = a77 + a78;
        Scalar const a80 = a34 - a74 + a75 + a79;
        Scalar const a81 = a68*(a42 + a73 + a80);
        Scalar const a82 = 3*a20;
        Scalar const a83 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a84 = 54*X[0];
        Scalar const a85 = -189*a20;
        Scalar const a86 = X[0]*X[0]*X[0];
        Scalar const a87 = 168*a86;
        Scalar const a88 = a83*(a84 + a85 + a87 - 3);
        Scalar const a89 = 18*X[0];
        Scalar const a90 = 90*X[0];
        Scalar const a91 = 252*a20;
        Scalar const a92 = 504*a20;
        Scalar const a93 = a92*X[1];
        Scalar const a94 = -252*a27 + a71 + a87 + a90 - a91 + a93 - 6;
        Scalar const a95 = 1680*X[0];
        Scalar const a96 = a20*X[1];
        Scalar const a97 = a37 + 16800*a96 - 4200*X[0]*X[1];
        Scalar const a98 = (1.0/70.0)*a0;
        Scalar const a99 = 30*X[0];
        Scalar const a100 = 24*X[1];
        Scalar const a101 = 24*a86;
        Scalar const a102 = 30*a34;
        Scalar const a103 = 216*a27;
        Scalar const a104 = 240*X[0];
        Scalar const a105 = a0*(a100 + a101 - a102 - a103 + a104*a34 - 51*a20 + 192*a96 + a99 - 3);
        Scalar const a106 = X[1]*X[1]*X[1];
        Scalar const a107 = 56*a106;
        Scalar const a108 = -735*a34;
        Scalar const a109 = a34*X[0];
        Scalar const a110 = 840*X[0];
        Scalar const a111 = a110 - 840*a20 + 280*a86;
        Scalar const a112 = a108 + 5880*a109 + a111;
        Scalar const a113 = 45*a34;
        Scalar const a114 = -a113;
        Scalar const a115 = -a82 + a86 + 3*X[0] - 1;
        Scalar const a116 = 6*a83*(35*a106 + a113*X[0] + a114 + a115 + a21*X[1] + a59 - a99*X[1]);
        Scalar const a117 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a118 = 60*X[0];
        Scalar const a119 = 560*a86;
        Scalar const a120 = a117*(-a118 - a119 + 420*a20 + 1);
        Scalar const a121 = 42*a20;
        Scalar const a122 = 28*X[0];
        Scalar const a123 = 28*X[1];
        Scalar const a124 = a20*X[2];
        Scalar const a125 = 6*a117*(-a121 - a122*X[2] + a123*a20 + 56*a124 + a16 - a25*X[1] + 28*a86 + 15*X[0] - 1);
        Scalar const a126 = 54*X[1];
        Scalar const a127 = X[1]*X[2];
        Scalar const a128 = 360*X[0];
        Scalar const a129 = 432*a20;
        Scalar const a130 = a27*X[2];
        Scalar const a131 = a83*(a114 + 144*a124 + a126 - 90*a127 + a128*a34 + a129*X[1] + 720*a130 - 153*a20 - 486*a27 - 162*a64 + 72*a86 + a90 + 18*X[2] - 9);
        Scalar const a132 = a110 - 3780*a20 + 2240*a86;
        Scalar const a133 = 168*a106;
        Scalar const a134 = 168*a127;
        Scalar const a135 = -a134;
        Scalar const a136 = a34*X[2];
        Scalar const a137 = 336*a136 - 5;
        Scalar const a138 = 10584*a109;
        Scalar const a139 = 33*a34;
        Scalar const a140 = 13*X[1];
        Scalar const a141 = 26*X[0];
        Scalar const a142 = a139*X[0];
        Scalar const a143 = a115 + 42*a136;
        Scalar const a144 = a100*X[2];
        Scalar const a145 = -a144;
        Scalar const a146 = a100*a64 + a145;
        Scalar const a147 = 6*a0*(21*a106 - a139 + a140*a20 + a140 - a141*X[1] + a142 + a143 + a146 + a15*a20 + a15 - a2*X[2]);
        Scalar const a148 = std::sqrt(30);
        Scalar const a149 = 168*a20;
        Scalar const a150 = a51 - 3*X[2];
        Scalar const a151 = 8*a86;
        Scalar const a152 = a70*X[0];
        Scalar const a153 = -a34 + a74 - 1;
        Scalar const a154 = 48*a27*X[2] + a76 - a78 - a89*X[1];
        Scalar const a155 = 3*a148*(8*a109 + 48*a124 + a151 + 48*a152 + a153 + a154 + a19 - 17*a20 - a75 - a84*X[2] + 16*a96);
        Scalar const a156 = -20*X[1] - 40*X[2];
        Scalar const a157 = 3360*a96 - 1680*X[0]*X[2];
        Scalar const a158 = 105*a34;
        Scalar const a159 = 42*a70;
        Scalar const a160 = 36*X[2];
        Scalar const a161 = a70*X[1];
        Scalar const a162 = 342*X[2];
        Scalar const a163 = 504*X[0];
        Scalar const a164 = 840*a20;
        Scalar const a165 = -588*X[1]*X[2];
        Scalar const a166 = a165 - 147*a34 - 294*a70;
        Scalar const a167 = 9*X[1];
        Scalar const a168 = 48*a127;
        Scalar const a169 = a49*X[2];
        Scalar const a170 = 6*a1*(7*a106 + a143 + a154 + a159*X[1] + a167*a20 + a167 - a168 + a169 + a35*X[0] - a35 - a53 + a7*a70);
        Scalar const a171 = -1470*X[1]*X[2];
        Scalar const a172 = X[2]*X[2]*X[2];
        Scalar const a173 = (3.0/4.0)*a18;
        Scalar const a174 = 24*X[0];
        Scalar const a175 = 30*a70;
        Scalar const a176 = -a175 - a69;
        Scalar const a177 = 20*a172;
        Scalar const a178 = a34*a52;
        Scalar const a179 = a175*X[1];
        Scalar const a180 = 6*a18*(a106 + a115 + a146 - a174*X[2] + a176 + a177 + a178 + a179 + a20*a8 + a26*X[2] + a52 - a55 + a69*X[0] + a70*a99 + a8);
        Scalar const a181 = 1680*a20;
        Scalar const a182 = -a181;
        Scalar const a183 = 8400*a64;
        Scalar const a184 = 3360*X[0]*X[1];
        Scalar const a185 = 8400*a27;
        Scalar const a186 = -210*a34 - 1050*a70 - 1050*X[1]*X[2];
        Scalar const a187 = std::sqrt(22);
        Scalar const a188 = X[0]*X[0]*X[0]*X[0];
        Scalar const a189 = a187*(-a122 + a149 + 210*a188 - 336*a86 + 1);
        Scalar const a190 = std::sqrt(11);
        Scalar const a191 = 72*a20;
        Scalar const a192 = 180*a188;
        Scalar const a193 = 144*a27;
        Scalar const a194 = a86*X[1];
        Scalar const a195 = a190*(240*a188 + a193 + 720*a194 + 264*a20 + a30 - 456*a86 - 648*a96 - 50*X[0]);
        Scalar const a196 = std::sqrt(66);
        Scalar const a197 = 252000*a194 - 151200*a20*X[1] + 16800*a27 + a37;
        Scalar const a198 = 180*X[0];
        Scalar const a199 = a20*a34;
        Scalar const a200 = -a46;
        Scalar const a201 = 45*a188 + 82*a20 + a200 - 108*a86 + 1;
        Scalar const a202 = a196*(-a198*a34 + 450*a199 + a201 + 152*a27 + a43 + 360*a86*X[1] - a93);
        Scalar const a203 = 45360*a20;
        Scalar const a204 = a108 + 264600*a199 - 52920*a34*X[0];
        Scalar const a205 = 18900*a188 + a203 + a204 - 52920*a86 - 7560*X[0];
        Scalar const a206 = 90*a34;
        Scalar const a207 = a106*X[0];
        Scalar const a208 = 66*a20;
        Scalar const a209 = -a141 + 20*a188 + a208 - 62*a86 + 2;
        Scalar const a210 = a187*(-70*a106 - 990*a109 + a128*X[1] + 300*a194 + 900*a199 + a206 + 700*a207 + a209 - a60 - 630*a96);
        Scalar const a211 = 168*a34;
        Scalar const a212 = X[1]*X[1]*X[1]*X[1];
        Scalar const a213 = std::sqrt(110);
        Scalar const a214 = (3.0/5.0)*a213;
        Scalar const a215 = 90720*a20;
        Scalar const a216 = 224*a106;
        Scalar const a217 = 126*a20;
        Scalar const a218 = a188 - a2 + a49 - 4*a86 + 1;
        Scalar const a219 = a213*(-a100 + a101*X[1] - 252*a109 - a191*X[1] + 126*a212 + a216*X[0] - a216 + a217*a34 + a218 + 126*a34 + 72*X[0]*X[1]);
        Scalar const a220 = std::sqrt(33);
        Scalar const a221 = 140*X[0];
        Scalar const a222 = a220*(a182 - 4200*a188 + a221 + 5040*a86 + 1);
        Scalar const a223 = 132*a20;
        Scalar const a224 = 48*a64;
        Scalar const a225 = a86*X[2];
        Scalar const a226 = 2*a220*(a100*X[0] - 216*a124 + 120*a188 + 120*a194 + a223 + a224 + 240*a225 + a51 + a54 - 228*a86 - 108*a96 - 25*X[0]);
        Scalar const a227 = -105840*a86;
        Scalar const a228 = -a71;
        Scalar const a229 = a96*X[2];
        Scalar const a230 = -378*a124 + 270*a225 + 114*a64;
        Scalar const a231 = a187*(-270*a109 - a118 - 540*a130 + 135*a188 + 810*a194 + 675*a199 + 246*a20 + a228 + 1350*a229 + a230 + 342*a27 + a61 + a77 - 324*a86 - 1134*a96 + 3);
        Scalar const a232 = 37800*a188;
        Scalar const a233 = 31920*a20 + a232 - 70560*a86 - 3360*X[0];
        Scalar const a234 = 110880*a194;
        Scalar const a235 = 211680*a96*X[2] - 42336*X[0]*X[1]*X[2];
        Scalar const a236 = -42*a106;
        Scalar const a237 = a110*a136 - 84*a136 + a209;
        Scalar const a238 = a196*(-726*a109 - 84*a124 - 528*a130 + a168 + 260*a194 + 660*a199 + 420*a207 + a224 + 480*a229 + a236 + a237 + 312*a27 + 66*a34 + 40*a86*X[2] - 546*a96 - 26*X[1] - 4*X[2]);
        Scalar const a239 = 15120*a20;
        Scalar const a240 = 30240*X[0]*X[1];
        Scalar const a241 = a239 + a240;
        Scalar const a242 = 100800*a194 - 166320*a20*X[1];
        Scalar const a243 = std::sqrt(330);
        Scalar const a244 = a106*X[2];
        Scalar const a245 = a34*a64;
        Scalar const a246 = -84672*a106 + 846720*a207;
        Scalar const a247 = 22*X[1];
        Scalar const a248 = a243*(a121*a127 + 42*a127 - 84*a130 + a133*X[0] + a133*X[2] - a133 + a15*a86 + a158*a20 + a158 - a169 - a208*X[1] + a211*a64 - a211*X[2] + 84*a212 + a218 + a247*a86 - a247 + 66*a27 - 210*a34*X[0] + a54 + a72);
        Scalar const a249 = 211680*a27;
        Scalar const a250 = 2520*a188 - 10080*a86 - 10080*X[0];
        Scalar const a251 = a20*a70;
        Scalar const a252 = a243*(a113*a20 - 108*a130 - 108*a152 + 90*a194 + a201 - a217*X[1] + 270*a229 + a230 + 270*a251 + 38*a27 - a34*a89 + a80);
        Scalar const a253 = -60480*a20*X[2];
        Scalar const a254 = -30240*a20*X[1];
        Scalar const a255 = 50400*a194;
        Scalar const a256 = 15120*X[0];
        Scalar const a257 = 105840*a251;
        Scalar const a258 = 12*a70;
        Scalar const a259 = -a52;
        Scalar const a260 = a102 + a259;
        Scalar const a261 = 120*a225 + 144*a64 - a91*X[2];
        Scalar const a262 = a213*(a103 + a106*a221 - 14*a106 - 330*a109 + a110*a70*X[1] + 96*a127 - 1056*a130 - 132*a152 - 84*a161 + 180*a194 + 300*a199 + a228 + 960*a229 + a237 + a258 + a260 + a261 + a36*a70 - 378*a96);
        Scalar const a263 = 60480*X[0];
        Scalar const a264 = -332640*a20*X[1] + 50400*a225;
        Scalar const a265 = 63*X[1];
        Scalar const a266 = a34*a70;
        Scalar const a267 = a27*a70;
        Scalar const a268 = 88*a106;
        Scalar const a269 = 69*a34;
        Scalar const a270 = 5*a187*(-138*a109 - a126*a20 + 102*a127 - 204*a130 - 312*a136 - 96*a161 + a20*a269 + 36*a212 + a218 + a228 + 102*a229 - a24*X[2] + 216*a244 + 312*a245 - a258*X[0] + 216*a266 + 96*a267 + a268*X[0] - a268 + a269 + a49*a70 + a71*a86 + a76*a86 + a79 + a84*X[1] + a89*X[2]);
        Scalar const a271 = 7560*a188 + a203 - 30240*a86 - 30240*X[0];
        Scalar const a272 = std::sqrt(154);
        Scalar const a273 = a172*X[0];
        Scalar const a274 = 120*X[0];
        Scalar const a275 = a102*a20;
        Scalar const a276 = -264*a130 + a175 - a177 + a275 + a28 + a63*a70;
        Scalar const a277 = 2*a272*(a106*a19 - a106 + a136*a274 - a142 + a144 - 330*a152 - a178 - a179 + 10*a188 - a20*a265 + 33*a20 + 240*a229 + 300*a251 + a259 + a261 + 200*a273 + a276 + a60*a86 + a69 - a8 - 31*a86 - 13*X[0] + 1);
        Scalar const a278 = std::sqrt(770);
        Scalar const a279 = a172*X[1];
        Scalar const a280 = 450*a266 - 14*X[1];
        Scalar const a281 = -a118*a34;
        Scalar const a282 = 12*X[1];
        Scalar const a283 = a278*(a106*a122 + 108*a106*X[2] - 28*a106 - a118*a70 + a127*a223 + 132*a127 - 228*a136 - a160*a20 + a160*X[0] - 300*a161 + a172*a46 + a175*a20 + 9*a212 + a218 + 228*a245 + a260 + 270*a266 + a276 + 180*a279 + a281 + a282*a86 - a282 + a52*a86 - 36*a96);
        Scalar const a284 = X[2]*X[2]*X[2]*X[2];
        Scalar const a285 = 60*a127 + 6*a34 + 90*a70;
        Scalar const a286 = 140*a172;
        Scalar const a287 = 20*X[2];
        Scalar const a288 = 3*a213*(a106*a2 + a106*a287 - 4*a106 - 12*a109 + a118*a136 + a118*X[2] + a12*a86 - a12 - a127*a274 + a127*a47 - 60*a136 + a161*a198 - 180*a161 + a172*a221 - a198*a70 + a206*a70 + a212 + a218 - a26*X[1] + a282*X[0] + 70*a284 + a285 + a286*X[1] - a286 - a287 + a34*a49 - a47*X[2] + a62*a70 + 20*a86*X[2]);
        Scalar const a289 = 90720*a64;
        P[0] = a0;
        P[1] = 0;
        P[2] = 0;
        P[3] = 0;
        P[4] = a0;
        P[5] = 0;
        P[6] = 0;
        P[7] = 0;
        P[8] = a0;
        P[9] = 0;
        P[10] = a4;
        P[11] = 0;
        P[12] = 0;
        P[13] = 0;
        P[14] = a4;
        P[15] = a5*(a6 - 3.0/2.0);
        P[16] = 0;
        P[17] = 0;
        P[18] = a5*(3.0/2.0 - a7);
        P[19] = a10;
        P[20] = 0;
        P[21] = 0;
        P[22] = 0;
        P[23] = a10;
        P[24] = (1.0/2.0)*a11*(a12 + a13 - 3);
        P[25] = 0;
        P[26] = 0;
        P[27] = (1.0/2.0)*a14;
        P[28] = a17;
        P[29] = 0;
        P[30] = a14;
        P[31] = 0;
        P[32] = a17;
        P[33] = 0;
        P[34] = a22;
        P[35] = 0;
        P[36] = 0;
        P[37] = 0;
        P[38] = a22;
        P[39] = a23*(-a24 + a7 + 3.0/10.0);
        P[40] = a31;
        P[41] = 0;
        P[42] = 0;
        P[43] = 0;
        P[44] = a31;
        P[45] = (2.0/3.0)*a32*(-a33 + a35 + 1);
        P[46] = 0;
        P[47] = 0;
        P[48] = (1.0/30.0)*a32*(-a36 - a38 + 240*X[0] - 43);
        P[49] = a44;
        P[50] = 0;
        P[51] = 0;
        P[52] = 0;
        P[53] = a44;
        P[54] = (1.0/10.0)*a48;
        P[55] = a58;
        P[56] = 0;
        P[57] = (1.0/5.0)*a48;
        P[58] = 0;
        P[59] = a58;
        P[60] = a18*(-a59 + a61 - 5*X[2] + 2);
        P[61] = 0;
        P[62] = 0;
        P[63] = (1.0/10.0)*a18*(-a62 - a63 - a65 - a66 + 180*X[0] - 31);
        P[64] = a67;
        P[65] = 0;
        P[66] = (1.0/20.0)*a18*(-a38 - a47 + 120*X[0] - 19);
        P[67] = 0;
        P[68] = a67;
        P[69] = (1.0/3.0)*a68*(a69 + 18*a70 + a71*X[2] - 5*X[1] - 15*X[2] + 2);
        P[70] = 0;
        P[71] = 0;
        P[72] = a68*(-a41 - a73 + (1.0/3.0)*X[1] + X[2] - 1.0/3.0);
        P[73] = a81;
        P[74] = 0;
        P[75] = a68*(-a50 - a57 + a7 - a82 - 21.0/20.0);
        P[76] = 0;
        P[77] = a81;
        P[78] = 0;
        P[79] = a88;
        P[80] = 0;
        P[81] = 0;
        P[82] = 0;
        P[83] = a88;
        P[84] = 126*a20 - a87 - a89 + 3.0/10.0;
        P[85] = a94;
        P[86] = 0;
        P[87] = 0;
        P[88] = 0;
        P[89] = a94;
        P[90] = a98*(7560*a20 - 4480*a86 - a95 - a97 - 67);
        P[91] = a105;
        P[92] = 0;
        P[93] = 0;
        P[94] = 0;
        P[95] = a105;
        P[96] = (15.0/4.0)*a83*(a107 - 63*a34 + a71 - 1);
        P[97] = 0;
        P[98] = 0;
        P[99] = (3.0/28.0)*a83*(-a112 - 2520*a96 + 5040*X[0]*X[1] - 650*X[1] + 117);
        P[100] = a116;
        P[101] = 0;
        P[102] = 0;
        P[103] = 0;
        P[104] = a116;
        P[105] = (1.0/10.0)*a120;
        P[106] = a125;
        P[107] = 0;
        P[108] = (1.0/5.0)*a120;
        P[109] = 0;
        P[110] = a125;
        P[111] = (3.0/70.0)*a83*(-8400*a124 + 5670*a20 - a66 - 3360*a86 - 8400*a96 + 2100*X[0]*X[1] + 2100*X[0]*X[2] - 1260*X[0] - 39);
        P[112] = a131;
        P[113] = 0;
        P[114] = (3.0/140.0)*a83*(-a132 - a97 - 11);
        P[115] = 0;
        P[116] = a131;
        P[117] = (3.0/4.0)*a0*(a133 + a135 + a137 - 231*a34 + a52 + 78*X[1]);
        P[118] = 0;
        P[119] = 0;
        P[120] = (1.0/28.0)*a0*(-2016*a124 - 14112*a130 - a138 + 2184*a20 + 1323*a34 - 728*a86 - 5544*a96 + 11088*X[0]*X[1] + 4032*X[0]*X[2] - 2184*X[0] + 1764*X[1]*X[2] - 1416*X[1] - 534*X[2] + 295);
        P[121] = a147;
        P[122] = 0;
        P[123] = a98*(-17640*a109 - a111 + 2205*a34 - 5040*a96 + 10080*X[0]*X[1] - 1230*X[1] + 101);
        P[124] = 0;
        P[125] = a147;
        P[126] = (1.0/7.0)*a148*(-a149*X[1] - a150 - a85 - 112*a86 - a92*X[2] + 42*X[0]*X[1] + 126*X[0]*X[2] - 42*X[0]);
        P[127] = a155;
        P[128] = 0;
        P[129] = (3.0/140.0)*a148*(-6720*a124 - a132 - a156 - a157 + 840*X[0]*X[1] - 29);
        P[130] = 0;
        P[131] = a155;
        P[132] = (3.0/4.0)*a1*(a107 + a126 - 336*a127 + a137 - a158 - a159 + a160 + 336*a161);
        P[133] = 0;
        P[134] = 0;
        P[135] = (3.0/28.0)*a1*(-1176*a109 - 1344*a124 - 4704*a130 - 2352*a152 - a162 - a163 - a164*X[1] - a166 + 504*a20 - a87 + 1680*X[0]*X[1] + 2688*X[0]*X[2] - 212*X[1] + 65);
        P[136] = a170;
        P[137] = 0;
        P[138] = (3.0/70.0)*a1*(-a112 - 11760*a130 - a157 - a164*X[2] - a171 + 6720*X[0]*X[1] - 855*X[1] - 205*X[2] + 104);
        P[139] = 0;
        P[140] = a170;
        P[141] = a173*(8*a106 + a135 + 96*a136 + 240*a161 + 160*a172 - 21*a34 - 210*a70 + a71 + 72*X[2] - 5);
        P[142] = 0;
        P[143] = 0;
        P[144] = a173*(-a100*a20 - a104*a70 - 96*a124 - 192*a130 - a145 - a151 - a174*a34 - a174 - a176 + 24*a20 - a6 + 48*X[0]*X[1] + 192*X[0]*X[2] - 24*X[2] + 3);
        P[145] = a180;
        P[146] = 0;
        P[147] = (3.0/70.0)*a18*(-a119 - 4200*a124 - 8400*a152 - a181*X[1] - a182 + a183 + a184 - a185*X[2] - a186 - a34*a95 - a95 - 435*X[1] - 1075*X[2] + 227);
        P[148] = 0;
        P[149] = a180;
        P[150] = 0;
        P[151] = a189;
        P[152] = 0;
        P[153] = 0;
        P[154] = 0;
        P[155] = a189;
        P[156] = a190*(-a191 - a192 + a7 + 216*a86 + 3.0/70.0);
        P[157] = a195;
        P[158] = 0;
        P[159] = 0;
        P[160] = 0;
        P[161] = a195;
        P[162] = (1.0/840.0)*a196*(-75600*a188 - a197 - 63840*a20 + 141120*a86 + 6720*X[0] - 97);
        P[163] = a202;
        P[164] = 0;
        P[165] = 0;
        P[166] = 0;
        P[167] = a202;
        P[168] = (1.0/252.0)*a187*(-151200*a194 + 249480*a20*X[1] - a205 - 45360*a27 - 850*X[1] + 247);
        P[169] = a210;
        P[170] = 0;
        P[171] = 0;
        P[172] = 0;
        P[173] = a210;
        P[174] = a214*(-336*a106 - a123 + a211 + 210*a212 + 1);
        P[175] = 0;
        P[176] = 0;
        P[177] = (1.0/2520.0)*a213*(127008*a106 - 15120*a188 - 211680*a194 - 846720*a199 + 635040*a20*X[1] - 1270080*a207 - a215 - 635040*a27 + 1693440*a34*X[0] - 172284*a34 + 60480*a86 + 60480*X[0] + 66724*X[1] - 6883);
        P[178] = a219;
        P[179] = 0;
        P[180] = 0;
        P[181] = 0;
        P[182] = a219;
        P[183] = (1.0/70.0)*a222;
        P[184] = a226;
        P[185] = 0;
        P[186] = (1.0/35.0)*a222;
        P[187] = 0;
        P[188] = a226;
        P[189] = (1.0/280.0)*a187*(-a183 - a185 - 56700*a188 - 126000*a194 + 75600*a20*X[1] + 75600*a20*X[2] - 47880*a20 - 126000*a225 - a227 - a66 + 5040*X[0] - 49);
        P[190] = a231;
        P[191] = 0;
        P[192] = (1.0/560.0)*a187*(-a197 - a233 - 1);
        P[193] = 0;
        P[194] = a231;
        P[195] = (1.0/252.0)*a196*(-a165 - 16380*a188 - 158760*a199 + 182952*a20*X[1] + 66528*a20*X[2] - 39312*a20 - 40320*a225 - a234 - a235 - 33264*a27 + 31752*a34*X[0] + 441*a34 - 12096*a64 + 45864*a86 + 6552*X[0] - 572*X[1] - 278*X[2] + 175);
        P[196] = a238;
        P[197] = 0;
        P[198] = (1.0/630.0)*a196*(-6300*a188 - a204 - a241 - a242 + 17640*a86 + 2520*X[0] - 310*X[1] + 22);
        P[199] = 0;
        P[200] = a238;
        P[201] = (1.0/5.0)*a243*(-756*a106 + a134 - 756*a136 + 420*a212 + 840*a244 + 420*a34 - 77*X[1] - 7*X[2] + 3);
        P[202] = 0;
        P[203] = 0;
        P[204] = (1.0/2520.0)*a243*(-87024*a127 - 13860*a188 - 176400*a194 - 635040*a199 + 529200*a20*X[1] + 105840*a20*X[2] - 83160*a20 - 35280*a225 - 423360*a229 - 1270080*a245 - a246 - 529200*a27 + 1270080*a34*X[0] + 127008*a34*X[2] - 128772*a34 - 105840*a64 + 55440*a86 + 846720*X[0]*X[1]*X[2] + 55440*X[0] + 55118*X[1] + 11606*X[2] - 6163);
        P[205] = a248;
        P[206] = 0;
        P[207] = (1.0/5040.0)*a243*(-423360*a199 + 211680*a20*X[1] - a239 - a246 - a249 - a250 + 846720*a34*X[0] - 83496*a34 - 70560*a86*X[1] + 20636*X[1] - 971);
        P[208] = 0;
        P[209] = a248;
        P[210] = (1.0/84.0)*a243*(-a150 - a163*X[2] - 1890*a188 - 2520*a194 + 1512*a20*X[1] + 4536*a20*X[2] - 1596*a20 - 7560*a225 + 3528*a86 - 168*X[0]*X[1] + 168*X[0]);
        P[211] = a252;
        P[212] = 0;
        P[213] = (1.0/560.0)*a243*(-a156 - a184 - 100800*a225 - a233 - a253 - a254 - a255 - 6720*X[0]*X[2] - 39);
        P[214] = 0;
        P[215] = a252;
        P[216] = (1.0/252.0)*a213*(a138 - a166 - 11340*a188 - 52920*a199 + 83160*a20*X[1] + 133056*a20*X[2] - 27216*a20 - 80640*a225 - a235 - a255 - a256*X[1] - a257 - 24192*a64 + 21168*a70*X[0] + 31752*a86 + 4536*X[0] - 232*X[1] - 402*X[2] + 85);
        P[217] = a262;
        P[218] = 0;
        P[219] = (1.0/630.0)*a213*(-a171 - 201600*a194 + 83160*a20*X[2] - a205 - a256*X[2] - a263*X[1] - a264 - 529200*a96*X[2] + 105840*X[0]*X[1]*X[2] - 1005*X[1] - 155*X[2] + 89);
        P[220] = 0;
        P[221] = a262;
        P[222] = a187*(-396*a106 + 408*a127 - 1404*a136 - 432*a161 + 180*a212 + 1080*a244 - a265 + 1080*a266 + 276*a34 + 24*a70 - 21*X[2] + 3);
        P[223] = 0;
        P[224] = 0;
        P[225] = (1.0/36.0)*a187*(2592*a106 - 11400*a127 - 810*a188 - 8280*a194 - 23760*a199 + 24840*a20*X[1] + 18360*a20*X[2] - 4860*a20 - 25920*a207 - 6120*a225 - 56160*a229 - 116640*a245 - 8640*a251 - 77760*a267 - 24840*a27 + 47520*a34*X[0] + 11664*a34*X[2] - 4794*a34 - 18360*a64 + 17280*a70*X[0] + 7776*a70*X[1] - 1812*a70 + 3240*a86 + 112320*X[0]*X[1]*X[2] + 3240*X[0] + 2549*X[1] + 1947*X[2] - 347);
        P[226] = a270;
        P[227] = 0;
        P[228] = (1.0/1008.0)*a187*(108864*a106 - 95424*a127 - 171360*a194 - 786240*a199 + 514080*a20*X[1] - 1088640*a207 - 20160*a225 - 483840*a229 - 2177280*a245 - a253 - a263*X[2] - 514080*a27 - a271 + 1572480*a34*X[0] + 217728*a34*X[2] - 158088*a34 + 967680*X[0]*X[1]*X[2] + 51204*X[1] + 5896*X[2] - 2983);
        P[229] = 0;
        P[230] = a270;
        P[231] = (1.0/12.0)*a272*(-a129 - a13 - a153 - a192 - a193 - 480*a194 - 360*a199 + 792*a20*X[1] + 3168*a20*X[2] - 1920*a225 - 2880*a229 - 3600*a251 + 72*a34*X[0] - 576*a64 + 720*a70*X[0] + 10*a70 + 504*a86 + 576*X[0]*X[1]*X[2] + 72*X[0] + 8*X[1]*X[2]);
        P[232] = a277;
        P[233] = 0;
        P[234] = (1.0/630.0)*a272*(-a186 - 75600*a199 + 415800*a20*X[2] - a215 - 252000*a225 - a227 - 378000*a229 - a232 - a240 - a242 - 378000*a251 + 15120*a34*X[0] - 75600*a64 + 75600*a70*X[0] + 75600*X[0]*X[1]*X[2] + 15120*X[0] - 585*X[1] - 1325*X[2] + 407);
        P[235] = 0;
        P[236] = a277;
        P[237] = (3.0/5.0)*a278*(176*a127 - 450*a161 - a162*a34 - 30*a172 + 15*a212 + a236 + 180*a244 + 300*a279 + a280 + 40*a34 + 40*a70 - 14*X[2] + 1);
        P[238] = 0;
        P[239] = 0;
        P[240] = (1.0/30.0)*a278*(108*a106 - 1376*a127 + 540*a172 - 90*a188 - 600*a194 - 1260*a199 + 1800*a20*X[1] + 3960*a20*X[2] - 540*a20 - 1080*a207 - 1320*a225 - 6840*a229 - 9720*a245 - 4500*a251 - 16200*a267 - 1800*a27 - 5400*a273 + 2520*a34*X[0] + 972*a34*X[2] - 253*a34 - 3960*a64 + 9000*a70*X[0] + 1620*a70*X[1] - 910*a70 + 360*a86 + 13680*X[0]*X[1]*X[2] + 360*X[0] + 182*X[1] + 404*X[2] - 37);
        P[241] = a283;
        P[242] = 0;
        P[243] = (1.0/2520.0)*a278*(27216*a106 - 152880*a127 - 287280*a199 + 151200*a20*X[2] - 272160*a207 - 756000*a229 - a234 - 1360800*a245 - 75600*a251 - a264 - 1360800*a267 - 332640*a27 - a271 + 574560*a34*X[0] + 136080*a34*X[2] - 58548*a34 - 151200*a64 + 151200*a70*X[0] + 136080*a70*X[1] - 14910*a70 + 1512000*X[0]*X[1]*X[2] + 34488*X[1] + 15080*X[2] - 3083);
        P[244] = 0;
        P[245] = a283;
        P[246] = a214*(-18*a106 + 240*a127 - 270*a136 - 810*a161 - 630*a172 + 5*a212 + 100*a244 + 700*a279 + a280 + 350*a284 + 24*a34 + 360*a70 - 70*X[2] + 3);
        P[247] = 0;
        P[248] = 0;
        P[249] = a214*(-a106*a46 + 2*a106 - a118*X[1] + 70*a172 - 5*a188 + 60*a20*X[1] + 300*a20*X[2] - 30*a20 - a200 - 100*a225 - 300*a229 - 450*a251 - 900*a267 - 700*a273 - a275 - a281 - a285 - a30 - a34*a65 + 30*a34*X[2] - a65 + 900*a70*X[0] + 90*a70*X[1] - 20*a86*X[1] + 20*a86 + 600*X[0]*X[1]*X[2] + 30*X[2]);
        P[250] = a288;
        P[251] = 0;
        P[252] = (1.0/168.0)*a213*(-10080*a106*X[0] + 1008*a106 - a127*a215 - 18480*a127 + 14112*a172 + 90720*a20*X[2] - a239*a34 - a241 - a249*a70 - a250 - a254 - a257 - 141120*a273 - a289*a34 - a289 + 30240*a34*X[0] + 9072*a34*X[2] - 3108*a34 + 211680*a70*X[0] + 21168*a70*X[1] - 21462*a70 - 10080*a86*X[1] - 30240*a86*X[2] + 181440*X[0]*X[1]*X[2] + 3208*X[1] + 9432*X[2] - 1111);
        P[253] = 0;
        P[254] = a288;
        return P;
    }
};

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_POLYNOMIAL_BASIS_H
