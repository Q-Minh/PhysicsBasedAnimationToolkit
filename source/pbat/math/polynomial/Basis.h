/**
 * @file Basis.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Basis polynomials in \f$ d=1,2,3 \f$ dimensions and orders \f$ p=1,2,3,4 \f$.
 *
 * All the polynomials defined are based on expressions computed symbolically in the script
 * polynomial_basis.py (or equivalently polynomial_basis.ipynb).
 *
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_POLYNOMIAL_BASIS_H
#define PBAT_MATH_POLYNOMIAL_BASIS_H

#include "pbat/Aliases.h"

#include <cmath>
#include <numbers>

namespace pbat {
namespace math {
namespace polynomial {

namespace detail {

template <int Dims, int Order>
class MonomialBasis;

template <int Dims, int Order>
class OrthonormalBasis;

template <int Dims, int Order>
class DivergenceFreeBasis;

} // namespace detail

/**
 * @brief Polynomial basis \f$ \left\{ \Pi_{i=1}^{d} \mathbf{X}_i^{p_i} \; \text{s.t.} \; 0 \leq
 * \sum_{i=1}^d p_i \leq p \right\} \f$ in dimensions \f$ d \f$ and order \f$ p \f$.
 *
 * See [Monomial basis](https://en.wikipedia.org/wiki/Monomial_basis).
 *
 * @tparam Dims Spatial dimensions
 * @tparam Order Polynomial order
 */
template <int Dims, int Order>
struct MonomialBasis : detail::MonomialBasis<Dims, Order>
{
  public:
    using BaseType                             = detail::MonomialBasis<Dims, Order>; ///< Base type
    inline static constexpr std::size_t kDims  = BaseType::kDims;  ///< Spatial dimensions
    inline static constexpr std::size_t kOrder = BaseType::kOrder; ///< Polynomial order
    inline static constexpr std::size_t kSize  = BaseType::kSize;  ///< Number of basis functions
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::eval(X);
    }
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::derivatives(X);
    }
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::antiderivatives(X);
    }
};

/**
 * @brief Orthonormal polynomial basis \f$ \left\{ P_i(X) \right\} \f$ in dimensions \f$ d \f$ and
 * order \f$ p
 * \f$.
 *
 * The basis is orthonormal with respect to the inner product
 * \f[
 *      \langle f, g \rangle = \int_{\Omega^d} f(X) g(X) \, d\Omega^d
 * \f]
 * where \f$ \Omega^d \f$ is the reference simplex in dimensions \f$ d \f$, e.g.
 * - the line segment \f$ 0,1 \f$ in 1D,
 * - the triangle \f$
 * \begin{pmatrix} 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 1 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 1 \end{pmatrix}
 * \f$ in 2D, and
 * - the tetrahedron \f$
 * \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 0 \\ 1\end{pmatrix}
 * \f$ in 3D.
 *
 * In other words,
 * \f[
 *     \langle P_i, P_j \rangle = \delta_{ij}
 * \f]
 * where \f$ \delta_{ij} \f$ is the Kronecker delta.
 *
 * See [Orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials).
 *
 * @tparam Dims Spatial dimensions
 * @tparam Order Polynomial order
 */
template <int Dims, int Order>
struct OrthonormalBasis : detail::OrthonormalBasis<Dims, Order>
{
  public:
    using BaseType = typename detail::OrthonormalBasis<Dims, Order>; ///< Base type
    inline static constexpr std::size_t kDims  = BaseType::kDims;    ///< Spatial dimensions
    inline static constexpr std::size_t kOrder = BaseType::kOrder;   ///< Polynomial order
    inline static constexpr std::size_t kSize  = BaseType::kSize;    ///< Number of basis functions
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::eval(X);
    }
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::derivatives(X);
    }
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::antiderivatives(X);
    }
};

/**
 * @brief Divergence-free polynomial basis \f$ \left\{ \mathbf{P}_i(X) \; \text{s.t.} \; \nabla_X
 * \mathbf{P}_i = 0 \right\} \f$ in dimensions \f$ d \f$ and order \f$ p \f$.
 *
 * The basis satisfies \f$ \nabla_X \cdot \mathbf{P}_i = 0 \f$ on the reference simplex in
 * dimensions \f$ d \f$, e.g.
 * - the line segment \f$ 0,1 \f$ in 1D,
 * - the triangle \f$
 * \begin{pmatrix} 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 1 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 1 \end{pmatrix}
 * \f$ in 2D, and
 * - the tetrahedron \f$
 * \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix},
 * \begin{pmatrix} 0 \\ 0 \\ 1\end{pmatrix}
 * \f$ in 3D.
 *
 * See \cite muller2013highly
 *
 * @tparam Dims Spatial dimensions
 * @tparam Order Polynomial order
 */
template <int Dims, int Order>
struct DivergenceFreeBasis : detail::DivergenceFreeBasis<Dims, Order>
{
  public:
    using BaseType = typename detail::DivergenceFreeBasis<Dims, Order>; ///< Base type
    inline static constexpr std::size_t kDims  = BaseType::kDims;       ///< Spatial dimensions
    inline static constexpr std::size_t kOrder = BaseType::kOrder;      ///< Polynomial order
    inline static constexpr std::size_t kSize  = BaseType::kSize; ///< Number of basis functions
    /**
     * @brief
     *
     * @param X
     * @return
     */
    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        return BaseType::eval(X);
    }
};

namespace detail {

/**
 * Monomial basis in 1D
 */

template <>
class MonomialBasis<1, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 2;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 1;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = X[0];
        P[1]      = (1.0 / 2.0) * ((X[0]) * (X[0]));
        return Pm;
    }
};

template <>
class MonomialBasis<1, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 3;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = ((X[0]) * (X[0]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 1;
        G[2]      = 2 * X[0];
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = X[0];
        P[1]      = (1.0 / 2.0) * ((X[0]) * (X[0]));
        P[2]      = (1.0 / 3.0) * ((X[0]) * (X[0]) * (X[0]));
        return Pm;
    }
};

template <>
class MonomialBasis<1, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 4;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = ((X[0]) * (X[0]));
        P[3] = ((X[0]) * (X[0]) * (X[0]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 1;
        G[2]      = 2 * X[0];
        G[3]      = 3 * ((X[0]) * (X[0]));
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = X[0];
        P[1]      = (1.0 / 2.0) * ((X[0]) * (X[0]));
        P[2]      = (1.0 / 3.0) * ((X[0]) * (X[0]) * (X[0]));
        P[3]      = (1.0 / 4.0) * ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        return Pm;
    }
};

template <>
class MonomialBasis<1, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 5;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = ((X[0]) * (X[0]));
        P[3] = ((X[0]) * (X[0]) * (X[0]));
        P[4] = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 1;
        G[2]      = 2 * X[0];
        G[3]      = 3 * ((X[0]) * (X[0]));
        G[4]      = 4 * ((X[0]) * (X[0]) * (X[0]));
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = X[0];
        P[1]      = (1.0 / 2.0) * ((X[0]) * (X[0]));
        P[2]      = (1.0 / 3.0) * ((X[0]) * (X[0]) * (X[0]));
        P[3]      = (1.0 / 4.0) * ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[4]      = (1.0 / 5.0) * std::pow(X[0], 5);
        return Pm;
    }
};

/**
 * Monomial basis in 2D
 */

template <>
class MonomialBasis<2, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 3;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[1];
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 0;
        G[2]      = 1;
        G[3]      = 0;
        G[4]      = 0;
        G[5]      = 1;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = X[0] * X[1];
        P[0]            = X[0];
        P[1]            = (1.0 / 2.0) * ((X[0]) * (X[0]));
        P[2]            = a0;
        P[3]            = X[1];
        P[4]            = a0;
        P[5]            = (1.0 / 2.0) * ((X[1]) * (X[1]));
        return Pm;
    }
};

template <>
class MonomialBasis<2, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 6;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = ((X[0]) * (X[0]));
        P[3] = X[1];
        P[4] = X[0] * X[1];
        P[5] = ((X[1]) * (X[1]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 0;
        G[2]      = 1;
        G[3]      = 0;
        G[4]      = 2 * X[0];
        G[5]      = 0;
        G[6]      = 0;
        G[7]      = 1;
        G[8]      = X[1];
        G[9]      = X[0];
        G[10]     = 0;
        G[11]     = 2 * X[1];
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = (1.0 / 2.0) * a0;
        Scalar const a2 = X[0] * X[1];
        Scalar const a3 = ((X[1]) * (X[1]));
        Scalar const a4 = a3 * X[0];
        P[0]            = X[0];
        P[1]            = a1;
        P[2]            = (1.0 / 3.0) * ((X[0]) * (X[0]) * (X[0]));
        P[3]            = a2;
        P[4]            = a1 * X[1];
        P[5]            = a4;
        P[6]            = X[1];
        P[7]            = a2;
        P[8]            = a0 * X[1];
        P[9]            = (1.0 / 2.0) * a3;
        P[10]           = (1.0 / 2.0) * a4;
        P[11]           = (1.0 / 3.0) * ((X[1]) * (X[1]) * (X[1]));
        return Pm;
    }
};

template <>
class MonomialBasis<2, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 10;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = ((X[1]) * (X[1]));
        P[0]            = 1;
        P[1]            = X[0];
        P[2]            = a0;
        P[3]            = ((X[0]) * (X[0]) * (X[0]));
        P[4]            = X[1];
        P[5]            = X[0] * X[1];
        P[6]            = a0 * X[1];
        P[7]            = a1;
        P[8]            = a1 * X[0];
        P[9]            = ((X[1]) * (X[1]) * (X[1]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = 2 * X[0];
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = a0 * X[1];
        Scalar const a3 = ((X[1]) * (X[1]));
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 1;
        G[3]            = 0;
        G[4]            = a0;
        G[5]            = 0;
        G[6]            = 3 * a1;
        G[7]            = 0;
        G[8]            = 0;
        G[9]            = 1;
        G[10]           = X[1];
        G[11]           = X[0];
        G[12]           = a2;
        G[13]           = a1;
        G[14]           = 0;
        G[15]           = 2 * X[1];
        G[16]           = a3;
        G[17]           = a2;
        G[18]           = 0;
        G[19]           = 3 * a3;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = (1.0 / 2.0) * a0;
        Scalar const a2 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a3 = (1.0 / 3.0) * a2;
        Scalar const a4 = X[0] * X[1];
        Scalar const a5 = ((X[1]) * (X[1]));
        Scalar const a6 = a5 * X[0];
        Scalar const a7 = a1 * a5;
        Scalar const a8 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a9 = a8 * X[0];
        P[0]            = X[0];
        P[1]            = a1;
        P[2]            = a3;
        P[3]            = (1.0 / 4.0) * ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[4]            = a4;
        P[5]            = a1 * X[1];
        P[6]            = a3 * X[1];
        P[7]            = a6;
        P[8]            = a7;
        P[9]            = a9;
        P[10]           = X[1];
        P[11]           = a4;
        P[12]           = a0 * X[1];
        P[13]           = a2 * X[1];
        P[14]           = (1.0 / 2.0) * a5;
        P[15]           = (1.0 / 2.0) * a6;
        P[16]           = a7;
        P[17]           = (1.0 / 3.0) * a8;
        P[18]           = (1.0 / 3.0) * a9;
        P[19]           = (1.0 / 4.0) * ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        return Pm;
    }
};

template <>
class MonomialBasis<2, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 15;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a2 = ((X[1]) * (X[1]));
        Scalar const a3 = ((X[1]) * (X[1]) * (X[1]));
        P[0]            = 1;
        P[1]            = X[0];
        P[2]            = a0;
        P[3]            = a1;
        P[4]            = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[5]            = X[1];
        P[6]            = X[0] * X[1];
        P[7]            = a0 * X[1];
        P[8]            = a1 * X[1];
        P[9]            = a2;
        P[10]           = a2 * X[0];
        P[11]           = a0 * a2;
        P[12]           = a3;
        P[13]           = a3 * X[0];
        P[14]           = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = 2 * X[0];
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = 3 * a1;
        Scalar const a3 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a4 = a0 * X[1];
        Scalar const a5 = 2 * X[1];
        Scalar const a6 = ((X[1]) * (X[1]));
        Scalar const a7 = 3 * a6;
        Scalar const a8 = ((X[1]) * (X[1]) * (X[1]));
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 1;
        G[3]            = 0;
        G[4]            = a0;
        G[5]            = 0;
        G[6]            = a2;
        G[7]            = 0;
        G[8]            = 4 * a3;
        G[9]            = 0;
        G[10]           = 0;
        G[11]           = 1;
        G[12]           = X[1];
        G[13]           = X[0];
        G[14]           = a4;
        G[15]           = a1;
        G[16]           = a2 * X[1];
        G[17]           = a3;
        G[18]           = 0;
        G[19]           = a5;
        G[20]           = a6;
        G[21]           = a4;
        G[22]           = a0 * a6;
        G[23]           = a1 * a5;
        G[24]           = 0;
        G[25]           = a7;
        G[26]           = a8;
        G[27]           = a7 * X[0];
        G[28]           = 0;
        G[29]           = 4 * a8;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = ((X[0]) * (X[0]));
        Scalar const a1  = (1.0 / 2.0) * a0;
        Scalar const a2  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a3  = (1.0 / 3.0) * a2;
        Scalar const a4  = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a5  = (1.0 / 4.0) * a4;
        Scalar const a6  = X[0] * X[1];
        Scalar const a7  = ((X[1]) * (X[1]));
        Scalar const a8  = a7 * X[0];
        Scalar const a9  = a1 * a7;
        Scalar const a10 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a11 = a10 * X[0];
        Scalar const a12 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a13 = a12 * X[0];
        Scalar const a14 = (1.0 / 2.0) * a7;
        Scalar const a15 = (1.0 / 3.0) * a10;
        P[0]             = X[0];
        P[1]             = a1;
        P[2]             = a3;
        P[3]             = a5;
        P[4]             = (1.0 / 5.0) * std::pow(X[0], 5);
        P[5]             = a6;
        P[6]             = a1 * X[1];
        P[7]             = a3 * X[1];
        P[8]             = a5 * X[1];
        P[9]             = a8;
        P[10]            = a9;
        P[11]            = a3 * a7;
        P[12]            = a11;
        P[13]            = a1 * a10;
        P[14]            = a13;
        P[15]            = X[1];
        P[16]            = a6;
        P[17]            = a0 * X[1];
        P[18]            = a2 * X[1];
        P[19]            = a4 * X[1];
        P[20]            = a14;
        P[21]            = (1.0 / 2.0) * a8;
        P[22]            = a9;
        P[23]            = a14 * a2;
        P[24]            = a15;
        P[25]            = (1.0 / 3.0) * a11;
        P[26]            = a0 * a15;
        P[27]            = (1.0 / 4.0) * a12;
        P[28]            = (1.0 / 4.0) * a13;
        P[29]            = (1.0 / 5.0) * std::pow(X[1], 5);
        return Pm;
    }
};

/**
 * Monomial basis in 3D
 */

template <>
class MonomialBasis<3, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 4;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = X[1];
        P[3] = X[2];
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 0;
        G[2]      = 0;
        G[3]      = 1;
        G[4]      = 0;
        G[5]      = 0;
        G[6]      = 0;
        G[7]      = 1;
        G[8]      = 0;
        G[9]      = 0;
        G[10]     = 0;
        G[11]     = 1;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = X[0] * X[1];
        Scalar const a1 = X[0] * X[2];
        Scalar const a2 = X[1] * X[2];
        P[0]            = X[0];
        P[1]            = (1.0 / 2.0) * ((X[0]) * (X[0]));
        P[2]            = a0;
        P[3]            = a1;
        P[4]            = X[1];
        P[5]            = a0;
        P[6]            = (1.0 / 2.0) * ((X[1]) * (X[1]));
        P[7]            = a2;
        P[8]            = X[2];
        P[9]            = a1;
        P[10]           = a2;
        P[11]           = (1.0 / 2.0) * ((X[2]) * (X[2]));
        return Pm;
    }
};

template <>
class MonomialBasis<3, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 10;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = X[0];
        P[2] = ((X[0]) * (X[0]));
        P[3] = X[1];
        P[4] = X[0] * X[1];
        P[5] = ((X[1]) * (X[1]));
        P[6] = X[2];
        P[7] = X[0] * X[2];
        P[8] = X[1] * X[2];
        P[9] = ((X[2]) * (X[2]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 0;
        G[2]      = 0;
        G[3]      = 1;
        G[4]      = 0;
        G[5]      = 0;
        G[6]      = 2 * X[0];
        G[7]      = 0;
        G[8]      = 0;
        G[9]      = 0;
        G[10]     = 1;
        G[11]     = 0;
        G[12]     = X[1];
        G[13]     = X[0];
        G[14]     = 0;
        G[15]     = 0;
        G[16]     = 2 * X[1];
        G[17]     = 0;
        G[18]     = 0;
        G[19]     = 0;
        G[20]     = 1;
        G[21]     = X[2];
        G[22]     = 0;
        G[23]     = X[0];
        G[24]     = 0;
        G[25]     = X[2];
        G[26]     = X[1];
        G[27]     = 0;
        G[28]     = 0;
        G[29]     = 2 * X[2];
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = ((X[0]) * (X[0]));
        Scalar const a1  = (1.0 / 2.0) * a0;
        Scalar const a2  = X[0] * X[1];
        Scalar const a3  = ((X[1]) * (X[1]));
        Scalar const a4  = a3 * X[0];
        Scalar const a5  = X[0] * X[2];
        Scalar const a6  = a2 * X[2];
        Scalar const a7  = ((X[2]) * (X[2]));
        Scalar const a8  = a7 * X[0];
        Scalar const a9  = (1.0 / 2.0) * a3;
        Scalar const a10 = X[1] * X[2];
        Scalar const a11 = a7 * X[1];
        P[0]             = X[0];
        P[1]             = a1;
        P[2]             = (1.0 / 3.0) * ((X[0]) * (X[0]) * (X[0]));
        P[3]             = a2;
        P[4]             = a1 * X[1];
        P[5]             = a4;
        P[6]             = a5;
        P[7]             = a1 * X[2];
        P[8]             = a6;
        P[9]             = a8;
        P[10]            = X[1];
        P[11]            = a2;
        P[12]            = a0 * X[1];
        P[13]            = a9;
        P[14]            = (1.0 / 2.0) * a4;
        P[15]            = (1.0 / 3.0) * ((X[1]) * (X[1]) * (X[1]));
        P[16]            = a10;
        P[17]            = a6;
        P[18]            = a9 * X[2];
        P[19]            = a11;
        P[20]            = X[2];
        P[21]            = a5;
        P[22]            = a0 * X[2];
        P[23]            = a10;
        P[24]            = a6;
        P[25]            = a3 * X[2];
        P[26]            = (1.0 / 2.0) * a7;
        P[27]            = (1.0 / 2.0) * a8;
        P[28]            = (1.0 / 2.0) * a11;
        P[29]            = (1.0 / 3.0) * ((X[2]) * (X[2]) * (X[2]));
        return Pm;
    }
};

template <>
class MonomialBasis<3, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 20;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = X[0] * X[1];
        Scalar const a2 = ((X[1]) * (X[1]));
        Scalar const a3 = ((X[2]) * (X[2]));
        P[0]            = 1;
        P[1]            = X[0];
        P[2]            = a0;
        P[3]            = ((X[0]) * (X[0]) * (X[0]));
        P[4]            = X[1];
        P[5]            = a1;
        P[6]            = a0 * X[1];
        P[7]            = a2;
        P[8]            = a2 * X[0];
        P[9]            = ((X[1]) * (X[1]) * (X[1]));
        P[10]           = X[2];
        P[11]           = X[0] * X[2];
        P[12]           = a0 * X[2];
        P[13]           = X[1] * X[2];
        P[14]           = a1 * X[2];
        P[15]           = a2 * X[2];
        P[16]           = a3;
        P[17]           = a3 * X[0];
        P[18]           = a3 * X[1];
        P[19]           = ((X[2]) * (X[2]) * (X[2]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = 2 * X[0];
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = a0 * X[1];
        Scalar const a3 = 2 * X[1];
        Scalar const a4 = ((X[1]) * (X[1]));
        Scalar const a5 = a0 * X[2];
        Scalar const a6 = a3 * X[2];
        Scalar const a7 = ((X[2]) * (X[2]));
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 0;
        G[3]            = 1;
        G[4]            = 0;
        G[5]            = 0;
        G[6]            = a0;
        G[7]            = 0;
        G[8]            = 0;
        G[9]            = 3 * a1;
        G[10]           = 0;
        G[11]           = 0;
        G[12]           = 0;
        G[13]           = 1;
        G[14]           = 0;
        G[15]           = X[1];
        G[16]           = X[0];
        G[17]           = 0;
        G[18]           = a2;
        G[19]           = a1;
        G[20]           = 0;
        G[21]           = 0;
        G[22]           = a3;
        G[23]           = 0;
        G[24]           = a4;
        G[25]           = a2;
        G[26]           = 0;
        G[27]           = 0;
        G[28]           = 3 * a4;
        G[29]           = 0;
        G[30]           = 0;
        G[31]           = 0;
        G[32]           = 1;
        G[33]           = X[2];
        G[34]           = 0;
        G[35]           = X[0];
        G[36]           = a5;
        G[37]           = 0;
        G[38]           = a1;
        G[39]           = 0;
        G[40]           = X[2];
        G[41]           = X[1];
        G[42]           = X[1] * X[2];
        G[43]           = X[0] * X[2];
        G[44]           = X[0] * X[1];
        G[45]           = 0;
        G[46]           = a6;
        G[47]           = a4;
        G[48]           = 0;
        G[49]           = 0;
        G[50]           = 2 * X[2];
        G[51]           = a7;
        G[52]           = 0;
        G[53]           = a5;
        G[54]           = 0;
        G[55]           = a7;
        G[56]           = a6;
        G[57]           = 0;
        G[58]           = 0;
        G[59]           = 3 * a7;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = ((X[0]) * (X[0]));
        Scalar const a1  = (1.0 / 2.0) * a0;
        Scalar const a2  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a3  = (1.0 / 3.0) * a2;
        Scalar const a4  = X[0] * X[1];
        Scalar const a5  = ((X[1]) * (X[1]));
        Scalar const a6  = a5 * X[0];
        Scalar const a7  = a1 * a5;
        Scalar const a8  = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a9  = a8 * X[0];
        Scalar const a10 = X[0] * X[2];
        Scalar const a11 = a4 * X[2];
        Scalar const a12 = X[1] * X[2];
        Scalar const a13 = a6 * X[2];
        Scalar const a14 = ((X[2]) * (X[2]));
        Scalar const a15 = a14 * X[0];
        Scalar const a16 = a1 * a14;
        Scalar const a17 = a14 * a4;
        Scalar const a18 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a19 = a18 * X[0];
        Scalar const a20 = a0 * X[1];
        Scalar const a21 = (1.0 / 2.0) * a5;
        Scalar const a22 = (1.0 / 3.0) * a8;
        Scalar const a23 = a20 * X[2];
        Scalar const a24 = a14 * X[1];
        Scalar const a25 = a14 * a21;
        Scalar const a26 = a18 * X[1];
        Scalar const a27 = (1.0 / 2.0) * a14;
        P[0]             = X[0];
        P[1]             = a1;
        P[2]             = a3;
        P[3]             = (1.0 / 4.0) * ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[4]             = a4;
        P[5]             = a1 * X[1];
        P[6]             = a3 * X[1];
        P[7]             = a6;
        P[8]             = a7;
        P[9]             = a9;
        P[10]            = a10;
        P[11]            = a1 * X[2];
        P[12]            = a3 * X[2];
        P[13]            = a11;
        P[14]            = a1 * a12;
        P[15]            = a13;
        P[16]            = a15;
        P[17]            = a16;
        P[18]            = a17;
        P[19]            = a19;
        P[20]            = X[1];
        P[21]            = a4;
        P[22]            = a20;
        P[23]            = a2 * X[1];
        P[24]            = a21;
        P[25]            = (1.0 / 2.0) * a6;
        P[26]            = a7;
        P[27]            = a22;
        P[28]            = (1.0 / 3.0) * a9;
        P[29]            = (1.0 / 4.0) * ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        P[30]            = a12;
        P[31]            = a11;
        P[32]            = a23;
        P[33]            = a21 * X[2];
        P[34]            = (1.0 / 2.0) * a13;
        P[35]            = a22 * X[2];
        P[36]            = a24;
        P[37]            = a17;
        P[38]            = a25;
        P[39]            = a26;
        P[40]            = X[2];
        P[41]            = a10;
        P[42]            = a0 * X[2];
        P[43]            = a2 * X[2];
        P[44]            = a12;
        P[45]            = a11;
        P[46]            = a23;
        P[47]            = a5 * X[2];
        P[48]            = a13;
        P[49]            = a8 * X[2];
        P[50]            = a27;
        P[51]            = (1.0 / 2.0) * a15;
        P[52]            = a16;
        P[53]            = (1.0 / 2.0) * a24;
        P[54]            = a27 * a4;
        P[55]            = a25;
        P[56]            = (1.0 / 3.0) * a18;
        P[57]            = (1.0 / 3.0) * a19;
        P[58]            = (1.0 / 3.0) * a26;
        P[59]            = (1.0 / 4.0) * ((X[2]) * (X[2]) * (X[2]) * (X[2]));
        return Pm;
    }
};

template <>
class MonomialBasis<3, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 35;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = ((X[0]) * (X[0]));
        Scalar const a1 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a2 = X[0] * X[1];
        Scalar const a3 = a0 * X[1];
        Scalar const a4 = ((X[1]) * (X[1]));
        Scalar const a5 = a4 * X[0];
        Scalar const a6 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a7 = ((X[2]) * (X[2]));
        Scalar const a8 = ((X[2]) * (X[2]) * (X[2]));
        P[0]            = 1;
        P[1]            = X[0];
        P[2]            = a0;
        P[3]            = a1;
        P[4]            = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[5]            = X[1];
        P[6]            = a2;
        P[7]            = a3;
        P[8]            = a1 * X[1];
        P[9]            = a4;
        P[10]           = a5;
        P[11]           = a0 * a4;
        P[12]           = a6;
        P[13]           = a6 * X[0];
        P[14]           = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        P[15]           = X[2];
        P[16]           = X[0] * X[2];
        P[17]           = a0 * X[2];
        P[18]           = a1 * X[2];
        P[19]           = X[1] * X[2];
        P[20]           = a2 * X[2];
        P[21]           = a3 * X[2];
        P[22]           = a4 * X[2];
        P[23]           = a5 * X[2];
        P[24]           = a6 * X[2];
        P[25]           = a7;
        P[26]           = a7 * X[0];
        P[27]           = a0 * a7;
        P[28]           = a7 * X[1];
        P[29]           = a2 * a7;
        P[30]           = a4 * a7;
        P[31]           = a8;
        P[32]           = a8 * X[0];
        P[33]           = a8 * X[1];
        P[34]           = ((X[2]) * (X[2]) * (X[2]) * (X[2]));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G        = Gm.data();
        Scalar const a0  = 2 * X[0];
        Scalar const a1  = ((X[0]) * (X[0]));
        Scalar const a2  = 3 * a1;
        Scalar const a3  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a4  = a0 * X[1];
        Scalar const a5  = 2 * X[1];
        Scalar const a6  = ((X[1]) * (X[1]));
        Scalar const a7  = 3 * a6;
        Scalar const a8  = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a9  = a0 * X[2];
        Scalar const a10 = X[1] * X[2];
        Scalar const a11 = a0 * a10;
        Scalar const a12 = a1 * X[2];
        Scalar const a13 = a5 * X[2];
        Scalar const a14 = a6 * X[2];
        Scalar const a15 = ((X[2]) * (X[2]));
        Scalar const a16 = a15 * X[1];
        Scalar const a17 = a15 * X[0];
        Scalar const a18 = ((X[2]) * (X[2]) * (X[2]));
        G[0]             = 0;
        G[1]             = 0;
        G[2]             = 0;
        G[3]             = 1;
        G[4]             = 0;
        G[5]             = 0;
        G[6]             = a0;
        G[7]             = 0;
        G[8]             = 0;
        G[9]             = a2;
        G[10]            = 0;
        G[11]            = 0;
        G[12]            = 4 * a3;
        G[13]            = 0;
        G[14]            = 0;
        G[15]            = 0;
        G[16]            = 1;
        G[17]            = 0;
        G[18]            = X[1];
        G[19]            = X[0];
        G[20]            = 0;
        G[21]            = a4;
        G[22]            = a1;
        G[23]            = 0;
        G[24]            = a2 * X[1];
        G[25]            = a3;
        G[26]            = 0;
        G[27]            = 0;
        G[28]            = a5;
        G[29]            = 0;
        G[30]            = a6;
        G[31]            = a4;
        G[32]            = 0;
        G[33]            = a0 * a6;
        G[34]            = a1 * a5;
        G[35]            = 0;
        G[36]            = 0;
        G[37]            = a7;
        G[38]            = 0;
        G[39]            = a8;
        G[40]            = a7 * X[0];
        G[41]            = 0;
        G[42]            = 0;
        G[43]            = 4 * a8;
        G[44]            = 0;
        G[45]            = 0;
        G[46]            = 0;
        G[47]            = 1;
        G[48]            = X[2];
        G[49]            = 0;
        G[50]            = X[0];
        G[51]            = a9;
        G[52]            = 0;
        G[53]            = a1;
        G[54]            = a2 * X[2];
        G[55]            = 0;
        G[56]            = a3;
        G[57]            = 0;
        G[58]            = X[2];
        G[59]            = X[1];
        G[60]            = a10;
        G[61]            = X[0] * X[2];
        G[62]            = X[0] * X[1];
        G[63]            = a11;
        G[64]            = a12;
        G[65]            = a1 * X[1];
        G[66]            = 0;
        G[67]            = a13;
        G[68]            = a6;
        G[69]            = a14;
        G[70]            = a11;
        G[71]            = a6 * X[0];
        G[72]            = 0;
        G[73]            = a7 * X[2];
        G[74]            = a8;
        G[75]            = 0;
        G[76]            = 0;
        G[77]            = 2 * X[2];
        G[78]            = a15;
        G[79]            = 0;
        G[80]            = a9;
        G[81]            = a0 * a15;
        G[82]            = 0;
        G[83]            = 2 * a12;
        G[84]            = 0;
        G[85]            = a15;
        G[86]            = a13;
        G[87]            = a16;
        G[88]            = a17;
        G[89]            = a11;
        G[90]            = 0;
        G[91]            = a15 * a5;
        G[92]            = 2 * a14;
        G[93]            = 0;
        G[94]            = 0;
        G[95]            = 3 * a15;
        G[96]            = a18;
        G[97]            = 0;
        G[98]            = 3 * a17;
        G[99]            = 0;
        G[100]           = a18;
        G[101]           = 3 * a16;
        G[102]           = 0;
        G[103]           = 0;
        G[104]           = 4 * a18;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = ((X[0]) * (X[0]));
        Scalar const a1  = (1.0 / 2.0) * a0;
        Scalar const a2  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a3  = (1.0 / 3.0) * a2;
        Scalar const a4  = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a5  = (1.0 / 4.0) * a4;
        Scalar const a6  = X[0] * X[1];
        Scalar const a7  = ((X[1]) * (X[1]));
        Scalar const a8  = a7 * X[0];
        Scalar const a9  = a1 * a7;
        Scalar const a10 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a11 = a10 * X[0];
        Scalar const a12 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a13 = a12 * X[0];
        Scalar const a14 = X[0] * X[2];
        Scalar const a15 = a6 * X[2];
        Scalar const a16 = X[1] * X[2];
        Scalar const a17 = a8 * X[2];
        Scalar const a18 = a7 * X[2];
        Scalar const a19 = a1 * a18;
        Scalar const a20 = a11 * X[2];
        Scalar const a21 = ((X[2]) * (X[2]));
        Scalar const a22 = a21 * X[0];
        Scalar const a23 = a1 * a21;
        Scalar const a24 = a21 * a6;
        Scalar const a25 = a21 * X[1];
        Scalar const a26 = a1 * a25;
        Scalar const a27 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a28 = a27 * X[0];
        Scalar const a29 = a27 * a6;
        Scalar const a30 = ((X[2]) * (X[2]) * (X[2]) * (X[2]));
        Scalar const a31 = a30 * X[0];
        Scalar const a32 = a0 * X[1];
        Scalar const a33 = a2 * X[1];
        Scalar const a34 = (1.0 / 2.0) * a7;
        Scalar const a35 = (1.0 / 3.0) * a10;
        Scalar const a36 = (1.0 / 4.0) * a12;
        Scalar const a37 = a32 * X[2];
        Scalar const a38 = a33 * X[2];
        Scalar const a39 = a21 * a34;
        Scalar const a40 = (1.0 / 2.0) * a21;
        Scalar const a41 = a40 * a8;
        Scalar const a42 = a27 * X[1];
        Scalar const a43 = a30 * X[1];
        Scalar const a44 = a0 * X[2];
        Scalar const a45 = (1.0 / 3.0) * a27;
        P[0]             = X[0];
        P[1]             = a1;
        P[2]             = a3;
        P[3]             = a5;
        P[4]             = (1.0 / 5.0) * std::pow(X[0], 5);
        P[5]             = a6;
        P[6]             = a1 * X[1];
        P[7]             = a3 * X[1];
        P[8]             = a5 * X[1];
        P[9]             = a8;
        P[10]            = a9;
        P[11]            = a3 * a7;
        P[12]            = a11;
        P[13]            = a1 * a10;
        P[14]            = a13;
        P[15]            = a14;
        P[16]            = a1 * X[2];
        P[17]            = a3 * X[2];
        P[18]            = a5 * X[2];
        P[19]            = a15;
        P[20]            = a1 * a16;
        P[21]            = a16 * a3;
        P[22]            = a17;
        P[23]            = a19;
        P[24]            = a20;
        P[25]            = a22;
        P[26]            = a23;
        P[27]            = a21 * a3;
        P[28]            = a24;
        P[29]            = a26;
        P[30]            = a21 * a8;
        P[31]            = a28;
        P[32]            = a1 * a27;
        P[33]            = a29;
        P[34]            = a31;
        P[35]            = X[1];
        P[36]            = a6;
        P[37]            = a32;
        P[38]            = a33;
        P[39]            = a4 * X[1];
        P[40]            = a34;
        P[41]            = (1.0 / 2.0) * a8;
        P[42]            = a9;
        P[43]            = a2 * a34;
        P[44]            = a35;
        P[45]            = (1.0 / 3.0) * a11;
        P[46]            = a0 * a35;
        P[47]            = a36;
        P[48]            = (1.0 / 4.0) * a13;
        P[49]            = (1.0 / 5.0) * std::pow(X[1], 5);
        P[50]            = a16;
        P[51]            = a15;
        P[52]            = a37;
        P[53]            = a38;
        P[54]            = a34 * X[2];
        P[55]            = (1.0 / 2.0) * a17;
        P[56]            = a19;
        P[57]            = a35 * X[2];
        P[58]            = (1.0 / 3.0) * a20;
        P[59]            = a36 * X[2];
        P[60]            = a25;
        P[61]            = a24;
        P[62]            = a21 * a32;
        P[63]            = a39;
        P[64]            = a41;
        P[65]            = a21 * a35;
        P[66]            = a42;
        P[67]            = a29;
        P[68]            = a27 * a34;
        P[69]            = a43;
        P[70]            = X[2];
        P[71]            = a14;
        P[72]            = a44;
        P[73]            = a2 * X[2];
        P[74]            = a4 * X[2];
        P[75]            = a16;
        P[76]            = a15;
        P[77]            = a37;
        P[78]            = a38;
        P[79]            = a18;
        P[80]            = a17;
        P[81]            = a44 * a7;
        P[82]            = a10 * X[2];
        P[83]            = a20;
        P[84]            = a12 * X[2];
        P[85]            = a40;
        P[86]            = (1.0 / 2.0) * a22;
        P[87]            = a23;
        P[88]            = a2 * a40;
        P[89]            = (1.0 / 2.0) * a25;
        P[90]            = a40 * a6;
        P[91]            = a26;
        P[92]            = a39;
        P[93]            = a41;
        P[94]            = a10 * a40;
        P[95]            = a45;
        P[96]            = (1.0 / 3.0) * a28;
        P[97]            = a0 * a45;
        P[98]            = (1.0 / 3.0) * a42;
        P[99]            = a45 * a6;
        P[100]           = a45 * a7;
        P[101]           = (1.0 / 4.0) * a30;
        P[102]           = (1.0 / 4.0) * a31;
        P[103]           = (1.0 / 4.0) * a43;
        P[104]           = (1.0 / 5.0) * std::pow(X[2], 5);
        return Pm;
    }
};

/**
 * Orthonormalized polynomial basis on reference line
 */

template <>
class OrthonormalBasis<1, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 2;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = std::numbers::sqrt3_v<Scalar> * (2 * X[0] - 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 2 * std::numbers::sqrt3_v<Scalar>;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        P[0]            = X[0];
        P[1]            = a0 * ((X[0]) * (X[0])) - a0 * X[0];
        return Pm;
    }
};

template <>
class OrthonormalBasis<1, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 3;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = 1;
        P[1] = std::numbers::sqrt3_v<Scalar> * (2 * X[0] - 1);
        P[2] = std::sqrt(5) * (6 * ((X[0]) * (X[0])) - 6 * X[0] + 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G = Gm.data();
        G[0]      = 0;
        G[1]      = 2 * std::numbers::sqrt3_v<Scalar>;
        G[2]      = std::sqrt(5) * (12 * X[0] - 6);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = std::sqrt(5);
        P[0]            = X[0];
        P[1]            = a0 * a1 - a0 * X[0];
        P[2]            = -3 * a1 * a2 + 2 * a2 * ((X[0]) * (X[0]) * (X[0])) + a2 * X[0];
        return Pm;
    }
};

template <>
class OrthonormalBasis<1, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 4;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = 2 * X[0] - 1;
        Scalar const a1 = ((X[0]) * (X[0]));
        P[0]            = 1;
        P[1]            = std::numbers::sqrt3_v<Scalar> * a0;
        P[2]            = std::sqrt(5) * (6 * a1 - 6 * X[0] + 1);
        P[3]            = std::sqrt(7) * a0 * (10 * a1 - 10 * X[0] + 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = std::sqrt(7);
        G[0]            = 0;
        G[1]            = 2 * std::numbers::sqrt3_v<Scalar>;
        G[2]            = std::sqrt(5) * (12 * X[0] - 6);
        G[3]            = a0 * (2 * X[0] - 1) * (20 * X[0] - 10) +
               2 * a0 * (10 * ((X[0]) * (X[0])) - 10 * X[0] + 1);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = std::sqrt(5);
        Scalar const a3 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a4 = std::sqrt(7);
        P[0]            = X[0];
        P[1]            = a0 * a1 - a0 * X[0];
        P[2]            = -3 * a1 * a2 + 2 * a2 * a3 + a2 * X[0];
        P[3] =
            6 * a1 * a4 - 10 * a3 * a4 + 5 * a4 * ((X[0]) * (X[0]) * (X[0]) * (X[0])) - a4 * X[0];
        return Pm;
    }
};

template <>
class OrthonormalBasis<1, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 5;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = 2 * X[0] - 1;
        Scalar const a1 = ((X[0]) * (X[0]));
        P[0]            = 1;
        P[1]            = std::numbers::sqrt3_v<Scalar> * a0;
        P[2]            = std::sqrt(5) * (6 * a1 - 6 * X[0] + 1);
        P[3]            = std::sqrt(7) * a0 * (10 * a1 - 10 * X[0] + 1);
        P[4]            = 270 * a1 + 210 * ((X[0]) * (X[0]) * (X[0]) * (X[0])) -
               420 * ((X[0]) * (X[0]) * (X[0])) - 60 * X[0] + 3;
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = std::sqrt(7);
        Scalar const a1 = ((X[0]) * (X[0]));
        G[0]            = 0;
        G[1]            = 2 * std::numbers::sqrt3_v<Scalar>;
        G[2]            = std::sqrt(5) * (12 * X[0] - 6);
        G[3] = a0 * (2 * X[0] - 1) * (20 * X[0] - 10) + 2 * a0 * (10 * a1 - 10 * X[0] + 1);
        G[4] = -1260 * a1 + 840 * ((X[0]) * (X[0]) * (X[0])) + 540 * X[0] - 60;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = ((X[0]) * (X[0]));
        Scalar const a2 = std::sqrt(5);
        Scalar const a3 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a4 = std::sqrt(7);
        Scalar const a5 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        P[0]            = X[0];
        P[1]            = a0 * a1 - a0 * X[0];
        P[2]            = -3 * a1 * a2 + 2 * a2 * a3 + a2 * X[0];
        P[3]            = 6 * a1 * a4 - 10 * a3 * a4 + 5 * a4 * a5 - a4 * X[0];
        P[4]            = -30 * a1 + 90 * a3 - 105 * a5 + 42 * std::pow(X[0], 5) + 3 * X[0];
        return Pm;
    }
};

/**
 * Orthonormalized polynomial basis on reference triangle
 */

template <>
class OrthonormalBasis<2, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 3;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        P[0] = std::numbers::sqrt2_v<Scalar>;
        P[1] = 2 * (3 * X[0] - 1);
        P[2] = 2 * std::numbers::sqrt3_v<Scalar> * (X[0] + 2 * X[1] - 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 6;
        G[3]            = 0;
        G[4]            = 2 * a0;
        G[5]            = 4 * a0;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 2 * X[0];
        Scalar const a2 = ((X[0]) * (X[0]));
        Scalar const a3 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a4 = 2 * a3;
        Scalar const a5 = -a4;
        P[0]            = a0 * X[0];
        P[1]            = -a1 + 3 * a2;
        P[2]            = a2 * a3 + (4 * a3 * X[1] + a5) * X[0];
        P[3]            = a0 * X[1];
        P[4]            = 2 * (3 * X[0] - 1) * X[1];
        P[5]            = a4 * ((X[1]) * (X[1])) + (a1 * a3 + a5) * X[1];
        return Pm;
    }
};

template <>
class OrthonormalBasis<2, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 6;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = X[0] + 2 * X[1] - 1;
        Scalar const a2 = ((X[0]) * (X[0]));
        Scalar const a3 = 6 * X[1];
        P[0]            = a0;
        P[1]            = 2 * (3 * X[0] - 1);
        P[2]            = 2 * std::numbers::sqrt3_v<Scalar> * a1;
        P[3]            = std::sqrt(6) * (10 * a2 - 8 * X[0] + 1);
        P[4]            = 3 * a0 * a1 * (5 * X[0] - 1);
        P[5] = std::sqrt(30) * (a2 + a3 * X[0] - a3 - 2 * X[0] + 6 * ((X[1]) * (X[1])) + 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a2 = a1 * (5 * X[0] - 1);
        Scalar const a3 = std::sqrt(30);
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 6;
        G[3]            = 0;
        G[4]            = 2 * a0;
        G[5]            = 4 * a0;
        G[6]            = std::sqrt(6) * (20 * X[0] - 8);
        G[7]            = 0;
        G[8]            = 15 * a1 * (X[0] + 2 * X[1] - 1) + 3 * a2;
        G[9]            = 6 * a2;
        G[10]           = a3 * (2 * X[0] + 6 * X[1] - 2);
        G[11]           = a3 * (6 * X[0] + 12 * X[1] - 6);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = a0 * X[0];
        Scalar const a2  = 2 * X[0];
        Scalar const a3  = ((X[0]) * (X[0]));
        Scalar const a4  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5  = 2 * a4;
        Scalar const a6  = -a5;
        Scalar const a7  = std::sqrt(6);
        Scalar const a8  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a9  = 3 * a0;
        Scalar const a10 = -a9;
        Scalar const a11 = a0 * X[1];
        Scalar const a12 = std::sqrt(30);
        Scalar const a13 = 3 * a12;
        Scalar const a14 = 6 * a12;
        Scalar const a15 = ((X[1]) * (X[1]));
        Scalar const a16 = 3 * X[0];
        P[0]             = a1;
        P[1]             = -a2 + 3 * a3;
        P[2]             = a3 * a4 + (4 * a4 * X[1] + a6) * X[0];
        P[3]             = -4 * a3 * a7 + (10.0 / 3.0) * a7 * a8 + a7 * X[0];
        P[4]             = 5 * a0 * a8 + a3 * (-9 * a0 + 15 * a11) + (-a10 - 6 * a11) * X[0];
        P[5]             = (1.0 / 3.0) * a12 * a8 + a3 * (-a12 + a13 * X[1]) +
               (a12 + a14 * a15 - a14 * X[1]) * X[0];
        P[6]  = a11;
        P[7]  = 2 * (a16 - 1) * X[1];
        P[8]  = a15 * a5 + (a2 * a4 + a6) * X[1];
        P[9]  = a7 * (10 * a3 - 8 * X[0] + 1) * X[1];
        P[10] = a15 * (15 * a1 + a10) + (15 * a0 * a3 - 18 * a1 + a9) * X[1];
        P[11] = 2 * a12 * ((X[1]) * (X[1]) * (X[1])) + a15 * (a12 * a16 - a13) +
                (-a12 * a2 + a12 * a3 + a12) * X[1];
        return Pm;
    }
};

template <>
class OrthonormalBasis<2, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 10;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = X[0] + 2 * X[1] - 1;
        Scalar const a2 = 2 * a1;
        Scalar const a3 = std::sqrt(6);
        Scalar const a4 = ((X[0]) * (X[0]));
        Scalar const a5 = 6 * X[1];
        Scalar const a6 = ((X[1]) * (X[1]));
        Scalar const a7 = a4 - 2 * X[0] + 1;
        Scalar const a8 = a5 * X[0] - a5 + 6 * a6 + a7;
        Scalar const a9 = 10 * X[1];
        P[0]            = a0;
        P[1]            = 2 * (3 * X[0] - 1);
        P[2]            = std::numbers::sqrt3_v<Scalar> * a2;
        P[3]            = a3 * (10 * a4 - 8 * X[0] + 1);
        P[4]            = 3 * a0 * a1 * (5 * X[0] - 1);
        P[5]            = std::sqrt(30) * a8;
        P[6]            = 2 * a0 * (-45 * a4 + 35 * ((X[0]) * (X[0]) * (X[0])) + 15 * X[0] - 1);
        P[7]            = a2 * a3 * (21 * a4 - 12 * X[0] + 1);
        P[8]            = 2 * std::sqrt(10) * a8 * (7 * X[0] - 1);
        P[9]            = std::sqrt(14) * a2 * (10 * a6 + a7 + a9 * X[0] - a9);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G        = Gm.data();
        Scalar const a0  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1  = std::sqrt(6);
        Scalar const a2  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a3  = a2 * (5 * X[0] - 1);
        Scalar const a4  = X[0] + 2 * X[1] - 1;
        Scalar const a5  = std::sqrt(30);
        Scalar const a6  = 6 * X[1];
        Scalar const a7  = 2 * X[0];
        Scalar const a8  = a7 - 2;
        Scalar const a9  = a6 + a8;
        Scalar const a10 = 6 * X[0] + 12 * X[1] - 6;
        Scalar const a11 = ((X[0]) * (X[0]));
        Scalar const a12 = a1 * (21 * a11 - 12 * X[0] + 1);
        Scalar const a13 = 2 * a4;
        Scalar const a14 = std::sqrt(10);
        Scalar const a15 = 2 * a14 * (7 * X[0] - 1);
        Scalar const a16 = ((X[1]) * (X[1]));
        Scalar const a17 = a11 - a7 + 1;
        Scalar const a18 = 10 * X[1];
        Scalar const a19 = std::sqrt(14);
        Scalar const a20 = a13 * a19;
        Scalar const a21 = a19 * (10 * a16 + a17 + a18 * X[0] - a18);
        G[0]             = 0;
        G[1]             = 0;
        G[2]             = 6;
        G[3]             = 0;
        G[4]             = 2 * a0;
        G[5]             = 4 * a0;
        G[6]             = a1 * (20 * X[0] - 8);
        G[7]             = 0;
        G[8]             = 15 * a2 * a4 + 3 * a3;
        G[9]             = 6 * a3;
        G[10]            = a5 * a9;
        G[11]            = a10 * a5;
        G[12]            = 2 * a2 * (105 * a11 - 90 * X[0] + 15);
        G[13]            = 0;
        G[14]            = a1 * a13 * (42 * X[0] - 12) + 2 * a12;
        G[15]            = 4 * a12;
        G[16]            = 14 * a14 * (6 * a16 + a17 + a6 * X[0] - a6) + a15 * a9;
        G[17]            = a10 * a15;
        G[18]            = a20 * (a18 + a8) + 2 * a21;
        G[19]            = a20 * (10 * X[0] + 20 * X[1] - 10) + 4 * a21;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = a0 * X[0];
        Scalar const a2  = 2 * X[0];
        Scalar const a3  = ((X[0]) * (X[0]));
        Scalar const a4  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5  = 2 * a4;
        Scalar const a6  = -a5;
        Scalar const a7  = 4 * X[1];
        Scalar const a8  = std::sqrt(6);
        Scalar const a9  = a8 * X[0];
        Scalar const a10 = a3 * a8;
        Scalar const a11 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a12 = a11 * a8;
        Scalar const a13 = a0 * a11;
        Scalar const a14 = 3 * a0;
        Scalar const a15 = -a14;
        Scalar const a16 = a0 * X[1];
        Scalar const a17 = std::sqrt(30);
        Scalar const a18 = 3 * a17;
        Scalar const a19 = 6 * a17;
        Scalar const a20 = ((X[1]) * (X[1]));
        Scalar const a21 = 15 * a0 * a3;
        Scalar const a22 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a23 = 2 * a8;
        Scalar const a24 = -a23;
        Scalar const a25 = a8 * X[1];
        Scalar const a26 = std::sqrt(10);
        Scalar const a27 = a26 * X[1];
        Scalar const a28 = 2 * a26;
        Scalar const a29 = a20 * a26;
        Scalar const a30 = std::sqrt(14);
        Scalar const a31 = 2 * a30;
        Scalar const a32 = -a31;
        Scalar const a33 = a30 * X[1];
        Scalar const a34 = 24 * a33;
        Scalar const a35 = a20 * a30;
        Scalar const a36 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a37 = 3 * X[0];
        Scalar const a38 = a26 * X[0];
        Scalar const a39 = a26 * a3;
        Scalar const a40 = 20 * a30;
        Scalar const a41 = 12 * a30;
        Scalar const a42 = a30 * X[0];
        P[0]             = a1;
        P[1]             = -a2 + 3 * a3;
        P[2]             = a3 * a4 + (a4 * a7 + a6) * X[0];
        P[3]             = -4 * a10 + (10.0 / 3.0) * a12 + a9;
        P[4]             = 5 * a13 + a3 * (-9 * a0 + 15 * a16) + (-a15 - 6 * a16) * X[0];
        P[5]             = (1.0 / 3.0) * a11 * a17 + a3 * (-a17 + a18 * X[1]) +
               (a17 + a19 * a20 - a19 * X[1]) * X[0];
        P[6] = (35.0 / 2.0) * a0 * a22 - 2 * a1 - 30 * a13 + a21;
        P[7] = a11 * (28 * a25 - 22 * a8) + (21.0 / 2.0) * a22 * a8 + a3 * (-24 * a25 + 13 * a8) +
               (a24 + a7 * a8) * X[0];
        P[8] = a11 * (-10 * a26 + 28 * a27) + (7.0 / 2.0) * a22 * a26 +
               a3 * (9 * a26 - 48 * a27 + 42 * a29) + (12 * a26 * X[1] - a28 - 12 * a29) * X[0];
        P[9] = a11 * (a32 + 8 * a33) + (1.0 / 2.0) * a22 * a30 + a3 * (3 * a30 - a34 + 30 * a35) +
               (40 * a30 * a36 + a32 + a34 - 60 * a35) * X[0];
        P[10] = a16;
        P[11] = 2 * (a37 - 1) * X[1];
        P[12] = a20 * a5 + (a2 * a4 + a6) * X[1];
        P[13] = a25 * (10 * a3 - 8 * X[0] + 1);
        P[14] = a20 * (15 * a1 + a15) + (-18 * a1 + a14 + a21) * X[1];
        P[15] = 2 * a17 * a36 + a20 * (a17 * a37 - a18) + (-a17 * a2 + a17 * a3 + a17) * X[1];
        P[16] = 2 * a16 * (35 * a11 - 45 * a3 + 15 * X[0] - 1);
        P[17] = a20 * (42 * a10 + a23 - 24 * a9) + (-66 * a10 + 42 * a12 + a24 + 26 * a9) * X[1];
        P[18] = a20 * (6 * a26 - 48 * a38 + 42 * a39) + a36 * (-4 * a26 + 28 * a38) +
                (14 * a11 * a26 - a28 + 18 * a38 - 30 * a39) * X[1];
        P[19] = a20 * (a3 * a41 + a41 - 24 * a42) + 10 * a30 * ((X[1]) * (X[1]) * (X[1]) * (X[1])) +
                a36 * (a40 * X[0] - a40) + (a11 * a31 - 6 * a3 * a30 + a32 + 6 * a42) * X[1];
        return Pm;
    }
};

template <>
class OrthonormalBasis<2, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 15;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = X[0] + 2 * X[1] - 1;
        Scalar const a2  = 2 * a1;
        Scalar const a3  = std::sqrt(6);
        Scalar const a4  = ((X[0]) * (X[0]));
        Scalar const a5  = std::sqrt(30);
        Scalar const a6  = 6 * X[1];
        Scalar const a7  = ((X[1]) * (X[1]));
        Scalar const a8  = a4 - 2 * X[0] + 1;
        Scalar const a9  = a6 * X[0] - a6 + 6 * a7 + a8;
        Scalar const a10 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a11 = std::sqrt(10);
        Scalar const a12 = 10 * X[1];
        Scalar const a13 = a12 * X[0] - a12 + 10 * a7 + a8;
        Scalar const a14 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a15 = 20 * X[1];
        Scalar const a16 = 90 * a7;
        Scalar const a17 = 140 * ((X[1]) * (X[1]) * (X[1]));
        Scalar const a18 = 60 * X[1];
        P[0]             = a0;
        P[1]             = 2 * (3 * X[0] - 1);
        P[2]             = std::numbers::sqrt3_v<Scalar> * a2;
        P[3]             = a3 * (10 * a4 - 8 * X[0] + 1);
        P[4]             = 3 * a0 * a1 * (5 * X[0] - 1);
        P[5]             = a5 * a9;
        P[6]             = 2 * a0 * (35 * a10 - 45 * a4 + 15 * X[0] - 1);
        P[7]             = a2 * a3 * (21 * a4 - 12 * X[0] + 1);
        P[8]             = 2 * a11 * a9 * (7 * X[0] - 1);
        P[9]             = std::sqrt(14) * a13 * a2;
        P[10]            = a11 * (-224 * a10 + 126 * a14 + 126 * a4 - 24 * X[0] + 1);
        P[11]            = a1 * a5 * (84 * a10 - 84 * a4 + 21 * X[0] - 1);
        P[12]            = 5 * a0 * a9 * (36 * a4 - 16 * X[0] + 1);
        P[13]            = std::sqrt(70) * a1 * a13 * (9 * X[0] - 1);
        P[14]            = 3 * a11 *
                (a10 * a15 - 4 * a10 + a14 - a15 + a16 * a4 + a16 + a17 * X[0] - a17 - a18 * a4 +
                 a18 * X[0] + 6 * a4 - 180 * a7 * X[0] - 4 * X[0] +
                 70 * ((X[1]) * (X[1]) * (X[1]) * (X[1])) + 1);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G        = Gm.data();
        Scalar const a0  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a1  = std::sqrt(6);
        Scalar const a2  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a3  = a2 * (5 * X[0] - 1);
        Scalar const a4  = X[0] + 2 * X[1] - 1;
        Scalar const a5  = std::sqrt(30);
        Scalar const a6  = 6 * X[1];
        Scalar const a7  = 2 * X[0];
        Scalar const a8  = a7 - 2;
        Scalar const a9  = a6 + a8;
        Scalar const a10 = 6 * X[0] + 12 * X[1] - 6;
        Scalar const a11 = ((X[0]) * (X[0]));
        Scalar const a12 = 12 * X[0];
        Scalar const a13 = a1 * (21 * a11 - a12 + 1);
        Scalar const a14 = 2 * a4;
        Scalar const a15 = std::sqrt(10);
        Scalar const a16 = 2 * a15 * (7 * X[0] - 1);
        Scalar const a17 = ((X[1]) * (X[1]));
        Scalar const a18 = a11 - a7 + 1;
        Scalar const a19 = 6 * a17 + a18 + a6 * X[0] - a6;
        Scalar const a20 = 10 * X[1];
        Scalar const a21 = a20 + a8;
        Scalar const a22 = std::sqrt(14);
        Scalar const a23 = a14 * a22;
        Scalar const a24 = 10 * a17 + a18 + a20 * X[0] - a20;
        Scalar const a25 = a22 * a24;
        Scalar const a26 = 10 * X[0] + 20 * X[1] - 10;
        Scalar const a27 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a28 = a5 * (-84 * a11 + 84 * a27 + 21 * X[0] - 1);
        Scalar const a29 = 5 * a2;
        Scalar const a30 = a29 * (36 * a11 - 16 * X[0] + 1);
        Scalar const a31 = std::sqrt(70);
        Scalar const a32 = 9 * X[0] - 1;
        Scalar const a33 = a31 * a32 * a4;
        Scalar const a34 = a24 * a31;
        Scalar const a35 = a32 * a34;
        Scalar const a36 = 60 * X[1];
        Scalar const a37 = 180 * a17;
        Scalar const a38 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a39 = X[0] * X[1];
        Scalar const a40 = 3 * a15;
        Scalar const a41 = 180 * X[1];
        Scalar const a42 = 420 * a17;
        G[0]             = 0;
        G[1]             = 0;
        G[2]             = 6;
        G[3]             = 0;
        G[4]             = 2 * a0;
        G[5]             = 4 * a0;
        G[6]             = a1 * (20 * X[0] - 8);
        G[7]             = 0;
        G[8]             = 15 * a2 * a4 + 3 * a3;
        G[9]             = 6 * a3;
        G[10]            = a5 * a9;
        G[11]            = a10 * a5;
        G[12]            = 2 * a2 * (105 * a11 - 90 * X[0] + 15);
        G[13]            = 0;
        G[14]            = a1 * a14 * (42 * X[0] - 12) + 2 * a13;
        G[15]            = 4 * a13;
        G[16]            = 14 * a15 * a19 + a16 * a9;
        G[17]            = a10 * a16;
        G[18]            = a21 * a23 + 2 * a25;
        G[19]            = a23 * a26 + 4 * a25;
        G[20]            = a15 * (-672 * a11 + 504 * a27 + 252 * X[0] - 24);
        G[21]            = 0;
        G[22]            = a28 + a4 * a5 * (252 * a11 - 168 * X[0] + 21);
        G[23]            = 2 * a28;
        G[24]            = a19 * a29 * (72 * X[0] - 16) + a30 * a9;
        G[25]            = a10 * a30;
        G[26]            = a21 * a33 + 9 * a34 * a4 + a35;
        G[27]            = a26 * a33 + 2 * a35;
        G[28] = a40 * (a11 * a36 - 12 * a11 + a12 + 4 * a27 + a36 + a37 * X[0] - a37 + 140 * a38 -
                       120 * a39 - 4);
        G[29] = a40 * (a11 * a41 - 60 * a11 + 20 * a27 + 280 * a38 - 360 * a39 + a41 + a42 * X[0] -
                       a42 + 60 * X[0] - 20);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = a0 * X[0];
        Scalar const a2  = 2 * X[0];
        Scalar const a3  = ((X[0]) * (X[0]));
        Scalar const a4  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a5  = 2 * a4;
        Scalar const a6  = -a5;
        Scalar const a7  = 4 * X[1];
        Scalar const a8  = std::sqrt(6);
        Scalar const a9  = a8 * X[0];
        Scalar const a10 = a3 * a8;
        Scalar const a11 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a12 = a11 * a8;
        Scalar const a13 = 5 * a0;
        Scalar const a14 = 3 * a0;
        Scalar const a15 = -a14;
        Scalar const a16 = a0 * X[1];
        Scalar const a17 = std::sqrt(30);
        Scalar const a18 = a11 * a17;
        Scalar const a19 = -a17;
        Scalar const a20 = 3 * a17;
        Scalar const a21 = 6 * a17;
        Scalar const a22 = ((X[1]) * (X[1]));
        Scalar const a23 = 15 * a0;
        Scalar const a24 = a23 * a3;
        Scalar const a25 = 30 * a0;
        Scalar const a26 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a27 = a0 * a26;
        Scalar const a28 = 2 * a8;
        Scalar const a29 = -a28;
        Scalar const a30 = a8 * X[1];
        Scalar const a31 = std::sqrt(10);
        Scalar const a32 = a26 * a31;
        Scalar const a33 = a31 * X[1];
        Scalar const a34 = 2 * a31;
        Scalar const a35 = a22 * a31;
        Scalar const a36 = std::sqrt(14);
        Scalar const a37 = 2 * a36;
        Scalar const a38 = -a37;
        Scalar const a39 = a36 * X[1];
        Scalar const a40 = 24 * a39;
        Scalar const a41 = a22 * a36;
        Scalar const a42 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a43 = a31 * X[0];
        Scalar const a44 = a3 * a31;
        Scalar const a45 = a11 * a31;
        Scalar const a46 = std::pow(X[0], 5);
        Scalar const a47 = a31 * a46;
        Scalar const a48 = a17 * X[1];
        Scalar const a49 = 42 * a17;
        Scalar const a50 = a0 * a22;
        Scalar const a51 = std::sqrt(70);
        Scalar const a52 = a51 * X[1];
        Scalar const a53 = 10 * a51;
        Scalar const a54 = a22 * a51;
        Scalar const a55 = a42 * a51;
        Scalar const a56 = -6 * a51;
        Scalar const a57 = 3 * a31;
        Scalar const a58 = 6 * a31;
        Scalar const a59 = -60 * a33;
        Scalar const a60 = 90 * a31;
        Scalar const a61 = 270 * a35;
        Scalar const a62 = a31 * a42;
        Scalar const a63 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a64 = 3 * X[0];
        Scalar const a65 = a17 * a3;
        Scalar const a66 = 30 * a31;
        Scalar const a67 = 20 * a36;
        Scalar const a68 = 12 * a36;
        Scalar const a69 = 24 * X[0];
        Scalar const a70 = 6 * a36;
        Scalar const a71 = a17 * X[0];
        Scalar const a72 = a0 * a3;
        Scalar const a73 = a0 * a11;
        Scalar const a74 = a51 * X[0];
        Scalar const a75 = a3 * a51;
        Scalar const a76 = a11 * a51;
        Scalar const a77 = a3 * a60;
        P[0]             = a1;
        P[1]             = -a2 + 3 * a3;
        P[2]             = a3 * a4 + (a4 * a7 + a6) * X[0];
        P[3]             = -4 * a10 + (10.0 / 3.0) * a12 + a9;
        P[4]             = a11 * a13 + a3 * (-9 * a0 + 15 * a16) + (-a15 - 6 * a16) * X[0];
        P[5] = (1.0 / 3.0) * a18 + a3 * (a19 + a20 * X[1]) + (a17 + a21 * a22 - a21 * X[1]) * X[0];
        P[6] = -2 * a1 - a11 * a25 + a24 + (35.0 / 2.0) * a27;
        P[7] = a11 * (28 * a30 - 22 * a8) + (21.0 / 2.0) * a26 * a8 + a3 * (-24 * a30 + 13 * a8) +
               (a29 + a7 * a8) * X[0];
        P[8] = a11 * (-10 * a31 + 28 * a33) + a3 * (9 * a31 - 48 * a33 + 42 * a35) +
               (7.0 / 2.0) * a32 + (12 * a31 * X[1] - a34 - 12 * a35) * X[0];
        P[9] = a11 * (a38 + 8 * a39) + (1.0 / 2.0) * a26 * a36 + a3 * (3 * a36 - a40 + 30 * a41) +
               (40 * a36 * a42 + a38 + a40 - 60 * a41) * X[0];
        P[10] = -56 * a32 + a43 - 12 * a44 + 42 * a45 + (126.0 / 5.0) * a47;
        P[11] = a11 * (35 * a17 - 56 * a48) + (84.0 / 5.0) * a17 * a46 + a26 * (a49 * X[1] - a49) +
                a3 * (-11 * a17 + 21 * a48) + (-a19 - 2 * a48) * X[0];
        P[12] = 36 * a0 * a46 + a11 * (115 * a0 - 520 * a16 + 360 * a50) +
                a26 * (-110 * a0 + 270 * a16) + a3 * (255 * a0 * X[1] - 45 * a0 - 240 * a50) +
                (a13 - 30 * a16 + a22 * a25) * X[0];
        P[13] = a11 * (-76 * a52 + a53 + 90 * a54) + a26 * (-7 * a51 + 27 * a52) +
                a3 * (66 * a52 - 150 * a54 + 90 * a55 + a56) + (9.0 / 5.0) * a46 * a51 +
                (30 * a22 * a51 + a51 - 12 * a52 - 20 * a55) * X[0];
        P[14] = a11 * (a22 * a60 + a58 + a59) + a26 * (15 * a33 - a57) +
                a3 * (-a58 + a60 * X[1] - a61 + 210 * a62) + (3.0 / 5.0) * a47 +
                (210 * a31 * a63 + a57 + a59 + a61 - 420 * a62) * X[0];
        P[15] = a16;
        P[16] = 2 * (a64 - 1) * X[1];
        P[17] = a22 * a5 + (a2 * a4 + a6) * X[1];
        P[18] = a30 * (10 * a3 - 8 * X[0] + 1);
        P[19] = a22 * (15 * a1 + a15) + (-18 * a1 + a14 + a24) * X[1];
        P[20] = 2 * a17 * a42 + a22 * (a17 * a64 - a20) + (-a17 * a2 + a17 + a65) * X[1];
        P[21] = 2 * a16 * (35 * a11 - 45 * a3 + 15 * X[0] - 1);
        P[22] = a22 * (42 * a10 + a28 - 24 * a9) + (-66 * a10 + 42 * a12 + a29 + 26 * a9) * X[1];
        P[23] = a22 * (-48 * a43 + 42 * a44 + a58) + a42 * (-4 * a31 + 28 * a43) +
                (-a3 * a66 - a34 + 18 * a43 + 14 * a45) * X[1];
        P[24] = a22 * (a3 * a68 - a36 * a69 + a68) + 10 * a36 * a63 + a42 * (a67 * X[0] - a67) +
                (a11 * a37 - a3 * a70 + a38 + a70 * X[0]) * X[1];
        P[25] = a33 * (-224 * a11 + 126 * a26 + 126 * a3 - a69 + 1);
        P[26] = a22 * (84 * a18 + a19 - 84 * a65 + 21 * a71) +
                (84 * a17 * a26 + a17 - 168 * a18 + 105 * a65 - 22 * a71) * X[1];
        P[27] = a22 * (255 * a1 - a23 - 780 * a72 + 540 * a73) +
                a42 * (10 * a0 - 160 * a1 + 360 * a72) +
                (-90 * a1 + a13 + 180 * a27 + 345 * a72 - 440 * a73) * X[1];
        P[28] = a22 * (a56 + 66 * a74 - 114 * a75 + 54 * a76) + a42 * (a53 - 100 * a74 + 90 * a75) +
                a63 * (-5 * a51 + 45 * a74) +
                (9 * a26 * a51 + a51 - 12 * a74 + 30 * a75 - 28 * a76) * X[1];
        P[29] = a22 * (a11 * a66 + 90 * a43 - a66 - a77) + 42 * a31 * std::pow(X[1], 5) +
                a42 * (-180 * a43 + a60 + a77) + a63 * (-105 * a31 + 105 * a43) +
                (a26 * a57 - 12 * a43 + 18 * a44 - 12 * a45 + a57) * X[1];
        return Pm;
    }
};

/**
 * Orthonormalized polynomial basis on reference tetrahedron
 */

template <>
class OrthonormalBasis<3, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 4;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0 = X[0] - 1;
        P[0]            = std::sqrt(6);
        P[1]            = std::sqrt(10) * (4 * X[0] - 1);
        P[2]            = 2 * std::sqrt(5) * (a0 + 3 * X[1]);
        P[3]            = 2 * std::sqrt(15) * (a0 + X[1] + 2 * X[2]);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G       = Gm.data();
        Scalar const a0 = std::sqrt(5);
        Scalar const a1 = std::sqrt(15);
        Scalar const a2 = 2 * a1;
        G[0]            = 0;
        G[1]            = 0;
        G[2]            = 0;
        G[3]            = 4 * std::sqrt(10);
        G[4]            = 0;
        G[5]            = 0;
        G[6]            = 2 * a0;
        G[7]            = 6 * a0;
        G[8]            = 0;
        G[9]            = a2;
        G[10]           = a2;
        G[11]           = 4 * a1;
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = std::sqrt(10);
        Scalar const a2  = ((X[0]) * (X[0]));
        Scalar const a3  = std::sqrt(5);
        Scalar const a4  = 2 * a3;
        Scalar const a5  = -a4;
        Scalar const a6  = std::sqrt(15);
        Scalar const a7  = 2 * a6;
        Scalar const a8  = a7 * X[1];
        Scalar const a9  = -a7;
        Scalar const a10 = 4 * a6 * X[2] + a9;
        Scalar const a11 = a1 * (4 * X[0] - 1);
        Scalar const a12 = ((X[1]) * (X[1]));
        Scalar const a13 = a7 * X[0];
        P[0]             = a0 * X[0];
        P[1]             = 2 * a1 * a2 - a1 * X[0];
        P[2]             = a2 * a3 + (6 * a3 * X[1] + a5) * X[0];
        P[3]             = a2 * a6 + (a10 + a8) * X[0];
        P[4]             = a0 * X[1];
        P[5]             = a11 * X[1];
        P[6]             = 3 * a12 * a3 + (a4 * X[0] + a5) * X[1];
        P[7]             = a12 * a6 + (a10 + a13) * X[1];
        P[8]             = a0 * X[2];
        P[9]             = a11 * X[2];
        P[10]            = a4 * (X[0] + 3 * X[1] - 1) * X[2];
        P[11]            = a7 * ((X[2]) * (X[2])) + (a13 + a8 + a9) * X[2];
        return Pm;
    }
};

template <>
class OrthonormalBasis<3, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 10;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0  = X[0] - 1;
        Scalar const a1  = 2 * a0 + 6 * X[1];
        Scalar const a2  = a0 + X[1] + 2 * X[2];
        Scalar const a3  = 2 * a2;
        Scalar const a4  = std::sqrt(14);
        Scalar const a5  = ((X[0]) * (X[0]));
        Scalar const a6  = 6 * X[0];
        Scalar const a7  = a6 - 1;
        Scalar const a8  = 8 * X[1];
        Scalar const a9  = ((X[1]) * (X[1]));
        Scalar const a10 = 2 * X[0];
        Scalar const a11 = -a10 + a5 + 1;
        Scalar const a12 = 6 * X[2];
        P[0]             = std::sqrt(6);
        P[1]             = std::sqrt(10) * (4 * X[0] - 1);
        P[2]             = std::sqrt(5) * a1;
        P[3]             = std::sqrt(15) * a3;
        P[4]             = a4 * (15 * a5 - 10 * X[0] + 1);
        P[5]             = std::sqrt(7) * a1 * a7;
        P[6]             = std::sqrt(42) * (a11 + a8 * X[0] - a8 + 10 * a9);
        P[7]             = std::sqrt(21) * a3 * a7;
        P[8]             = 3 * a2 * a4 * (a0 + 5 * X[1]);
        P[9] = std::sqrt(210) * (a10 * X[1] + a11 + a12 * X[1] - a12 + a6 * X[2] + a9 - 2 * X[1] +
                                 6 * ((X[2]) * (X[2])));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G        = Gm.data();
        Scalar const a0  = std::sqrt(5);
        Scalar const a1  = std::sqrt(15);
        Scalar const a2  = 2 * a1;
        Scalar const a3  = std::sqrt(14);
        Scalar const a4  = std::sqrt(7);
        Scalar const a5  = 6 * X[0];
        Scalar const a6  = a5 - 1;
        Scalar const a7  = a4 * a6;
        Scalar const a8  = X[0] - 1;
        Scalar const a9  = std::sqrt(42);
        Scalar const a10 = 2 * X[0] - 2;
        Scalar const a11 = std::sqrt(21);
        Scalar const a12 = a11 * a6;
        Scalar const a13 = 2 * a12;
        Scalar const a14 = a8 + X[1] + 2 * X[2];
        Scalar const a15 = a8 + 5 * X[1];
        Scalar const a16 = 3 * a3;
        Scalar const a17 = a15 * a16;
        Scalar const a18 = std::sqrt(210);
        Scalar const a19 = a18 * (a10 + 2 * X[1] + 6 * X[2]);
        G[0]             = 0;
        G[1]             = 0;
        G[2]             = 0;
        G[3]             = 4 * std::sqrt(10);
        G[4]             = 0;
        G[5]             = 0;
        G[6]             = 2 * a0;
        G[7]             = 6 * a0;
        G[8]             = 0;
        G[9]             = a2;
        G[10]            = a2;
        G[11]            = 4 * a1;
        G[12]            = a3 * (30 * X[0] - 10);
        G[13]            = 0;
        G[14]            = 0;
        G[15]            = 12 * a4 * (a8 + 3 * X[1]) + 2 * a7;
        G[16]            = 6 * a7;
        G[17]            = 0;
        G[18]            = a9 * (a10 + 8 * X[1]);
        G[19]            = a9 * (8 * X[0] + 20 * X[1] - 8);
        G[20]            = 0;
        G[21]            = 12 * a11 * a14 + a13;
        G[22]            = a13;
        G[23]            = 4 * a12;
        G[24]            = a14 * a16 + a17;
        G[25]            = 15 * a14 * a3 + a17;
        G[26]            = 6 * a15 * a3;
        G[27]            = a19;
        G[28]            = a19;
        G[29]            = a18 * (a5 + 6 * X[1] + 12 * X[2] - 6);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = std::sqrt(10);
        Scalar const a2  = ((X[0]) * (X[0]));
        Scalar const a3  = std::sqrt(5);
        Scalar const a4  = 2 * a3;
        Scalar const a5  = -a4;
        Scalar const a6  = 6 * X[1];
        Scalar const a7  = std::sqrt(15);
        Scalar const a8  = 2 * a7;
        Scalar const a9  = a8 * X[1];
        Scalar const a10 = -a8;
        Scalar const a11 = 4 * X[2];
        Scalar const a12 = a10 + a11 * a7;
        Scalar const a13 = std::sqrt(14);
        Scalar const a14 = a13 * X[0];
        Scalar const a15 = 5 * a13;
        Scalar const a16 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a17 = a13 * a16;
        Scalar const a18 = std::sqrt(7);
        Scalar const a19 = 4 * a16;
        Scalar const a20 = 18 * X[1];
        Scalar const a21 = std::sqrt(42);
        Scalar const a22 = (1.0 / 3.0) * a16;
        Scalar const a23 = 4 * a21;
        Scalar const a24 = 8 * X[1];
        Scalar const a25 = ((X[1]) * (X[1]));
        Scalar const a26 = 10 * a25;
        Scalar const a27 = std::sqrt(21);
        Scalar const a28 = 2 * a27;
        Scalar const a29 = -a28;
        Scalar const a30 = a28 * X[1];
        Scalar const a31 = a11 * a27;
        Scalar const a32 = 12 * a27;
        Scalar const a33 = 3 * a13;
        Scalar const a34 = -a33;
        Scalar const a35 = 9 * a13;
        Scalar const a36 = 6 * X[2];
        Scalar const a37 = -a13 * a36 + a33;
        Scalar const a38 = 15 * a13;
        Scalar const a39 = -a13 * a20 + a25 * a38;
        Scalar const a40 = std::sqrt(210);
        Scalar const a41 = a40 * X[1];
        Scalar const a42 = 3 * a40;
        Scalar const a43 = -a40 + a42 * X[2];
        Scalar const a44 = ((X[2]) * (X[2]));
        Scalar const a45 = -a36 * a40 + 6 * a40 * a44 + a40;
        Scalar const a46 = a25 * a40 - 2 * a41;
        Scalar const a47 = 4 * X[0];
        Scalar const a48 = a1 * (a47 - 1);
        Scalar const a49 = a8 * X[0];
        Scalar const a50 = a13 * (15 * a2 - 10 * X[0] + 1);
        Scalar const a51 = a18 * X[0];
        Scalar const a52 = 2 * a18;
        Scalar const a53 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a54 = 2 * X[0];
        Scalar const a55 = 6 * X[0];
        Scalar const a56 = a27 * X[0];
        Scalar const a57 = a2 * a32 + a28 - 14 * a56;
        Scalar const a58 = -6 * a14 + a2 * a33;
        Scalar const a59 = a40 * X[0];
        Scalar const a60 = a2 * a40 - 2 * a59;
        Scalar const a61 = (X[0] + 3 * X[1] - 1) * X[2];
        Scalar const a62 = a32 * X[0];
        P[0]             = a0 * X[0];
        P[1]             = 2 * a1 * a2 - a1 * X[0];
        P[2]             = a2 * a3 + (a3 * a6 + a5) * X[0];
        P[3]             = a2 * a7 + (a12 + a9) * X[0];
        P[4]             = a14 - a15 * a2 + 5 * a17;
        P[5]             = a18 * a19 + a2 * (a18 * a20 - 7 * a18) + (-a18 * a6 + 2 * a18) * X[0];
        P[6] = a2 * (-a21 + a23 * X[1]) + a21 * a22 + (-a21 * a24 + a21 * a26 + a21) * X[0];
        P[7] = a19 * a27 + a2 * (a27 * a6 - 7 * a27 + a32 * X[2]) + (-a29 - a30 - a31) * X[0];
        P[8] = a17 + a2 * (a33 * X[2] + a34 + a35 * X[1]) +
               (30 * a13 * X[1] * X[2] + a37 + a39) * X[0];
        P[9]  = a2 * (a41 + a43) + a22 * a40 + (a36 * a41 + a45 + a46) * X[0];
        P[10] = a0 * X[1];
        P[11] = a48 * X[1];
        P[12] = 3 * a25 * a3 + (a4 * X[0] + a5) * X[1];
        P[13] = a25 * a7 + (a12 + a49) * X[1];
        P[14] = a50 * X[1];
        P[15] = a25 * (-3 * a18 + 18 * a51) + (12 * a18 * a2 - 14 * a51 + a52) * X[1];
        P[16] = (10.0 / 3.0) * a21 * a53 + a25 * (a21 * a47 - a23) +
                (a2 * a21 - a21 * a54 + a21) * X[1];
        P[17] = a25 * (a27 * a55 - a27) + (-a31 + 24 * a56 * X[2] + a57) * X[1];
        P[18] = a15 * a53 + a25 * (9 * a14 - a35 + a38 * X[2]) + (a14 * a36 + a37 + a58) * X[1];
        P[19] = a25 * (a43 + a59) + (1.0 / 3.0) * a40 * a53 + (a36 * a59 + a45 + a60) * X[1];
        P[20] = a0 * X[2];
        P[21] = a48 * X[2];
        P[22] = a4 * a61;
        P[23] = a44 * a8 + (a10 + a49 + a9) * X[2];
        P[24] = a50 * X[2];
        P[25] = a52 * a61 * (a55 - 1);
        P[26] = a21 * (a2 + a24 * X[0] - a24 + a26 - a54 + 1) * X[2];
        P[27] = a44 * (a29 + a62) + (-a30 + a57 + a62 * X[1]) * X[2];
        P[28] = a44 * (3 * a14 + a34 + a38 * X[1]) + (a14 * a20 + a33 + a39 + a58) * X[2];
        P[29] = 2 * a40 * ((X[2]) * (X[2]) * (X[2])) + a44 * (3 * a41 - a42 + 3 * a59) +
                (a40 + a41 * a54 + a46 + a60) * X[2];
        return Pm;
    }
};

template <>
class OrthonormalBasis<3, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 20;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = std::sqrt(10);
        Scalar const a2  = X[0] - 1;
        Scalar const a3  = a2 + 3 * X[1];
        Scalar const a4  = 2 * a3;
        Scalar const a5  = a2 + X[1] + 2 * X[2];
        Scalar const a6  = 2 * a5;
        Scalar const a7  = std::sqrt(14);
        Scalar const a8  = 10 * X[0];
        Scalar const a9  = ((X[0]) * (X[0]));
        Scalar const a10 = 15 * a9;
        Scalar const a11 = 6 * X[0];
        Scalar const a12 = a11 - 1;
        Scalar const a13 = 8 * X[1];
        Scalar const a14 = ((X[1]) * (X[1]));
        Scalar const a15 = 2 * X[0];
        Scalar const a16 = -a15 + a9 + 1;
        Scalar const a17 = a13 * X[0] - a13 + 10 * a14 + a16;
        Scalar const a18 = a2 + 5 * X[1];
        Scalar const a19 = a5 * a7;
        Scalar const a20 = 6 * X[2];
        Scalar const a21 = ((X[2]) * (X[2]));
        Scalar const a22 = a14 + a15 * X[1] + a16 - 2 * X[1];
        Scalar const a23 = a11 * X[2] + a20 * X[1] - a20 + 6 * a21 + a22;
        Scalar const a24 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a25 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a26 = 168 * a9 - 84 * X[0] + 6;
        Scalar const a27 = 8 * X[0] - 1;
        Scalar const a28 = 3 * a27;
        Scalar const a29 = 45 * a14;
        Scalar const a30 = 12 * X[1];
        Scalar const a31 = 10 * X[2];
        P[0]             = a0;
        P[1]             = a1 * (4 * X[0] - 1);
        P[2]             = std::sqrt(5) * a4;
        P[3]             = std::sqrt(15) * a6;
        P[4]             = a7 * (a10 - a8 + 1);
        P[5]             = std::sqrt(7) * a12 * a4;
        P[6]             = std::sqrt(42) * a17;
        P[7]             = std::sqrt(21) * a12 * a6;
        P[8]             = 3 * a18 * a19;
        P[9]             = std::sqrt(210) * a23;
        P[10]            = 3 * a24 * (56 * a25 - 63 * a9 + 18 * X[0] - 1);
        P[11]            = a26 * a3;
        P[12]            = a0 * a17 * a28;
        P[13]            = 6 * a24 *
                (a10 * X[1] + a25 + a29 * X[0] - a29 - 3 * a9 - 30 * X[0] * X[1] + 3 * X[0] +
                 35 * ((X[1]) * (X[1]) * (X[1])) + 15 * X[1] - 1);
        P[14] = std::numbers::sqrt3_v<Scalar> * a26 * a5;
        P[15] = 9 * a18 * a24 * a27 * a5;
        P[16] = 6 * a0 * a5 * (21 * a14 + a16 + a30 * X[0] - a30);
        P[17] = std::sqrt(30) * a23 * a28;
        P[18] = 6 * a1 * a23 * (a2 + 7 * X[1]);
        P[19] = 6 * a19 * (10 * a21 + a22 + a31 * X[1] - a31 + a8 * X[2]);
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G        = Gm.data();
        Scalar const a0  = std::sqrt(10);
        Scalar const a1  = std::sqrt(5);
        Scalar const a2  = std::sqrt(15);
        Scalar const a3  = 2 * a2;
        Scalar const a4  = std::sqrt(14);
        Scalar const a5  = 30 * X[0];
        Scalar const a6  = std::sqrt(7);
        Scalar const a7  = 6 * X[0];
        Scalar const a8  = a7 - 1;
        Scalar const a9  = a6 * a8;
        Scalar const a10 = X[0] - 1;
        Scalar const a11 = a10 + 3 * X[1];
        Scalar const a12 = std::sqrt(42);
        Scalar const a13 = 8 * X[1];
        Scalar const a14 = 2 * X[0];
        Scalar const a15 = a14 - 2;
        Scalar const a16 = a13 + a15;
        Scalar const a17 = 8 * X[0];
        Scalar const a18 = a17 + 20 * X[1] - 8;
        Scalar const a19 = std::sqrt(21);
        Scalar const a20 = a19 * a8;
        Scalar const a21 = 2 * a20;
        Scalar const a22 = a10 + X[1] + 2 * X[2];
        Scalar const a23 = a10 + 5 * X[1];
        Scalar const a24 = 3 * a4;
        Scalar const a25 = a23 * a24;
        Scalar const a26 = a22 * a4;
        Scalar const a27 = 6 * a4;
        Scalar const a28 = std::sqrt(210);
        Scalar const a29 = 6 * X[2];
        Scalar const a30 = 2 * X[1];
        Scalar const a31 = a15 + a30;
        Scalar const a32 = a29 + a31;
        Scalar const a33 = a28 * a32;
        Scalar const a34 = a7 + 6 * X[1] + 12 * X[2] - 6;
        Scalar const a35 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a36 = ((X[0]) * (X[0]));
        Scalar const a37 = 168 * a36;
        Scalar const a38 = std::sqrt(6);
        Scalar const a39 = a17 - 1;
        Scalar const a40 = 3 * a39;
        Scalar const a41 = a38 * a40;
        Scalar const a42 = ((X[1]) * (X[1]));
        Scalar const a43 = -a14 + a36 + 1;
        Scalar const a44 = 6 * a35;
        Scalar const a45 = 90 * X[1];
        Scalar const a46 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a47 = a46 * (28 * a36 - 14 * X[0] + 1);
        Scalar const a48 = 6 * a47;
        Scalar const a49 = 6 * a22;
        Scalar const a50 = a23 * a35;
        Scalar const a51 = 9 * a39;
        Scalar const a52 = a50 * a51;
        Scalar const a53 = a22 * a35;
        Scalar const a54 = 12 * X[1];
        Scalar const a55 = a38 * a49;
        Scalar const a56 = a38 * (21 * a42 + a43 + a54 * X[0] - a54);
        Scalar const a57 = 6 * a56;
        Scalar const a58 = std::sqrt(30);
        Scalar const a59 = a40 * a58;
        Scalar const a60 = a32 * a59;
        Scalar const a61 = ((X[2]) * (X[2]));
        Scalar const a62 = a14 * X[1] - a30 + a42 + a43;
        Scalar const a63 = a29 * X[1] - a29 + 6 * a61 + a62 + a7 * X[2];
        Scalar const a64 = 6 * a0 * (a10 + 7 * X[1]);
        Scalar const a65 = a32 * a64;
        Scalar const a66 = a0 * a63;
        Scalar const a67 = 10 * X[2];
        Scalar const a68 = 6 * a26;
        Scalar const a69 = 10 * a61 + a62 + a67 * X[0] + a67 * X[1] - a67;
        Scalar const a70 = a27 * a69 + a68 * (a31 + a67);
        G[0]             = 0;
        G[1]             = 0;
        G[2]             = 0;
        G[3]             = 4 * a0;
        G[4]             = 0;
        G[5]             = 0;
        G[6]             = 2 * a1;
        G[7]             = 6 * a1;
        G[8]             = 0;
        G[9]             = a3;
        G[10]            = a3;
        G[11]            = 4 * a2;
        G[12]            = a4 * (a5 - 10);
        G[13]            = 0;
        G[14]            = 0;
        G[15]            = 12 * a11 * a6 + 2 * a9;
        G[16]            = 6 * a9;
        G[17]            = 0;
        G[18]            = a12 * a16;
        G[19]            = a12 * a18;
        G[20]            = 0;
        G[21]            = 12 * a19 * a22 + a21;
        G[22]            = a21;
        G[23]            = 4 * a20;
        G[24]            = a22 * a24 + a25;
        G[25]            = a25 + 15 * a26;
        G[26]            = a23 * a27;
        G[27]            = a33;
        G[28]            = a33;
        G[29]            = a28 * a34;
        G[30]            = 3 * a35 * (a37 - 126 * X[0] + 18);
        G[31]            = 0;
        G[32]            = 0;
        G[33]            = a11 * (336 * X[0] - 84) + a37 - 84 * X[0] + 6;
        G[34]            = 504 * a36 - 252 * X[0] + 18;
        G[35]            = 0;
        G[36]            = a16 * a41 + 24 * a38 * (a13 * X[0] - a13 + 10 * a42 + a43);
        G[37]            = a18 * a41;
        G[38]            = 0;
        G[39]            = a44 * (3 * a36 + 45 * a42 + a5 * X[1] - a7 - 30 * X[1] + 3);
        G[40]            = a44 * (15 * a36 + 105 * a42 + a45 * X[0] - a45 - a5 + 15);
        G[41]            = 0;
        G[42]            = a46 * a49 * (56 * X[0] - 14) + a48;
        G[43]            = a48;
        G[44]            = 12 * a47;
        G[45]            = 72 * a22 * a50 + a51 * a53 + a52;
        G[46]            = 45 * a39 * a53 + a52;
        G[47]            = 18 * a39 * a50;
        G[48]            = a55 * (a15 + a54) + a57;
        G[49]            = a55 * (12 * X[0] + 42 * X[1] - 12) + a57;
        G[50]            = 12 * a56;
        G[51]            = 24 * a58 * a63 + a60;
        G[52]            = a60;
        G[53]            = a34 * a59;
        G[54]            = a65 + 6 * a66;
        G[55]            = a65 + 42 * a66;
        G[56]            = a34 * a64;
        G[57]            = a70;
        G[58]            = a70;
        G[59]            = 12 * a4 * a69 + a68 * (10 * X[0] + 10 * X[1] + 20 * X[2] - 10);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P         = Pm.data();
        Scalar const a0   = std::sqrt(6);
        Scalar const a1   = a0 * X[0];
        Scalar const a2   = std::sqrt(10);
        Scalar const a3   = a2 * X[0];
        Scalar const a4   = ((X[0]) * (X[0]));
        Scalar const a5   = a2 * a4;
        Scalar const a6   = std::sqrt(5);
        Scalar const a7   = 2 * a6;
        Scalar const a8   = -a7;
        Scalar const a9   = 6 * X[1];
        Scalar const a10  = std::sqrt(15);
        Scalar const a11  = 2 * a10;
        Scalar const a12  = a11 * X[1];
        Scalar const a13  = -a11;
        Scalar const a14  = 4 * X[2];
        Scalar const a15  = a10 * a14 + a13;
        Scalar const a16  = std::sqrt(14);
        Scalar const a17  = a16 * X[0];
        Scalar const a18  = 5 * a16;
        Scalar const a19  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a20  = a16 * a19;
        Scalar const a21  = std::sqrt(7);
        Scalar const a22  = 4 * a19;
        Scalar const a23  = 18 * X[1];
        Scalar const a24  = std::sqrt(42);
        Scalar const a25  = (1.0 / 3.0) * a19;
        Scalar const a26  = 4 * a24;
        Scalar const a27  = 8 * X[1];
        Scalar const a28  = ((X[1]) * (X[1]));
        Scalar const a29  = 10 * a28;
        Scalar const a30  = std::sqrt(21);
        Scalar const a31  = 2 * a30;
        Scalar const a32  = -a31;
        Scalar const a33  = a31 * X[1];
        Scalar const a34  = a14 * a30;
        Scalar const a35  = 12 * X[2];
        Scalar const a36  = 3 * a16;
        Scalar const a37  = -a36;
        Scalar const a38  = 9 * a16;
        Scalar const a39  = a16 * X[2];
        Scalar const a40  = 30 * X[1];
        Scalar const a41  = 6 * a16;
        Scalar const a42  = a36 - a41 * X[2];
        Scalar const a43  = a16 * a23;
        Scalar const a44  = -a43;
        Scalar const a45  = a16 * a28;
        Scalar const a46  = a44 + 15 * a45;
        Scalar const a47  = std::sqrt(210);
        Scalar const a48  = a47 * X[1];
        Scalar const a49  = 3 * a47;
        Scalar const a50  = -a47 + a49 * X[2];
        Scalar const a51  = 6 * X[2];
        Scalar const a52  = 6 * a47;
        Scalar const a53  = ((X[2]) * (X[2]));
        Scalar const a54  = a47 + a52 * a53 - a52 * X[2];
        Scalar const a55  = a28 * a47 - 2 * a48;
        Scalar const a56  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a57  = 3 * X[0];
        Scalar const a58  = 27 * a56;
        Scalar const a59  = a19 * a56;
        Scalar const a60  = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a61  = 42 * a60;
        Scalar const a62  = 168 * X[1];
        Scalar const a63  = 6 * a0;
        Scalar const a64  = a0 * X[1];
        Scalar const a65  = 3 * a0;
        Scalar const a66  = a0 * a28;
        Scalar const a67  = a56 * a60;
        Scalar const a68  = 6 * a56;
        Scalar const a69  = -a68;
        Scalar const a70  = a56 * X[1];
        Scalar const a71  = 9 * a56;
        Scalar const a72  = 90 * a56;
        Scalar const a73  = a72 * X[1];
        Scalar const a74  = a28 * a56;
        Scalar const a75  = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a76  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a77  = 6 * a76;
        Scalar const a78  = a77 * X[1];
        Scalar const a79  = -a77;
        Scalar const a80  = a35 * a76 + a79;
        Scalar const a81  = a76 * X[1];
        Scalar const a82  = 84 * a76;
        Scalar const a83  = a76 * X[2];
        Scalar const a84  = a56 * X[2];
        Scalar const a85  = 45 * a56;
        Scalar const a86  = a28 * a85;
        Scalar const a87  = 360 * a70;
        Scalar const a88  = (3.0 / 2.0) * a60;
        Scalar const a89  = -a63;
        Scalar const a90  = a0 * X[2];
        Scalar const a91  = 78 * a64;
        Scalar const a92  = 12 * a0;
        Scalar const a93  = a92 * X[2];
        Scalar const a94  = 72 * a64;
        Scalar const a95  = 144 * X[2];
        Scalar const a96  = a89 + a93;
        Scalar const a97  = 126 * a0 * a75 - 198 * a66 + a91;
        Scalar const a98  = std::sqrt(30);
        Scalar const a99  = 6 * a98;
        Scalar const a100 = a98 * X[1];
        Scalar const a101 = a98 * X[2];
        Scalar const a102 = 3 * a98;
        Scalar const a103 = a102 * a28;
        Scalar const a104 = a53 * a98;
        Scalar const a105 = 18 * a104;
        Scalar const a106 = a28 * a98;
        Scalar const a107 = 72 * a101;
        Scalar const a108 = 6 * a2;
        Scalar const a109 = -a108;
        Scalar const a110 = 12 * a2;
        Scalar const a111 = a2 * X[1];
        Scalar const a112 = 54 * a111;
        Scalar const a113 = 36 * a2;
        Scalar const a114 = a113 * X[2];
        Scalar const a115 = 45 * a28;
        Scalar const a116 = 18 * a2;
        Scalar const a117 = 144 * a111;
        Scalar const a118 = a2 * a28;
        Scalar const a119 = a109 - a113 * a53 + a114;
        Scalar const a120 = a112 - 90 * a118 + 42 * a2 * a75;
        Scalar const a121 = -a41;
        Scalar const a122 = a121 + 24 * a39;
        Scalar const a123 = 72 * a39;
        Scalar const a124 = a16 * a53;
        Scalar const a125 = -a123 + 90 * a124 + a38;
        Scalar const a126 = 144 * X[1];
        Scalar const a127 = 72 * X[2];
        Scalar const a128 = 180 * a124;
        Scalar const a129 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a130 = a121 + a123 - a128 + 120 * a129 * a16;
        Scalar const a131 = a41 * a75 + a43 - 18 * a45;
        Scalar const a132 = 4 * X[0];
        Scalar const a133 = a132 - 1;
        Scalar const a134 = a11 * X[0];
        Scalar const a135 = 15 * a4;
        Scalar const a136 = a135 - 10 * X[0] + 1;
        Scalar const a137 = a16 * X[1];
        Scalar const a138 = 18 * X[0];
        Scalar const a139 = 2 * a21;
        Scalar const a140 = 14 * X[0];
        Scalar const a141 = 12 * a4;
        Scalar const a142 = 2 * X[0];
        Scalar const a143 = 6 * X[0];
        Scalar const a144 = a30 * X[0];
        Scalar const a145 = 24 * X[2];
        Scalar const a146 = -a140 * a30 + a141 * a30 + a31;
        Scalar const a147 = 6 * a17;
        Scalar const a148 = -a147 + a36 * a4;
        Scalar const a149 = a47 * X[0];
        Scalar const a150 = -2 * a149 + a4 * a47;
        Scalar const a151 = a138 + 56 * a19 - 63 * a4 - 1;
        Scalar const a152 = 3 * X[1];
        Scalar const a153 = 252 * a4;
        Scalar const a154 = 90 * X[0];
        Scalar const a155 = 168 * a19;
        Scalar const a156 = a0 * a4;
        Scalar const a157 = 24 * a19;
        Scalar const a158 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a159 = a154 * a56;
        Scalar const a160 = a4 * a56;
        Scalar const a161 = -a153 * a76 + a154 * a76 + a155 * a76;
        Scalar const a162 = a56 * X[0];
        Scalar const a163 = a84 * X[0];
        Scalar const a164 = 144 * a4;
        Scalar const a165 = a159 - 153 * a160 + 72 * a59 - a71;
        Scalar const a166 = 39 * a0;
        Scalar const a167 = 18 * a1 - 18 * a156 + a19 * a63;
        Scalar const a168 = 8 * X[0];
        Scalar const a169 = a98 * X[0];
        Scalar const a170 = 9 * a98;
        Scalar const a171 = a4 * a98;
        Scalar const a172 = -a102 + a157 * a98 + 30 * a169 - 51 * a171;
        Scalar const a173 = a2 * X[2];
        Scalar const a174 = 27 * a2;
        Scalar const a175 = 36 * a3;
        Scalar const a176 = a116 * a4;
        Scalar const a177 = a108 * a19 - a176 + 18 * a3;
        Scalar const a178 = 18 * a17;
        Scalar const a179 = 72 * a17;
        Scalar const a180 = -18 * a16 * a4 + a178 + 6 * a20;
        Scalar const a181 = a152 + X[0] - 1;
        Scalar const a182 = a181 * X[2];
        Scalar const a183 = (-a142 + a27 * X[0] - a27 + a29 + a4 + 1) * X[2];
        Scalar const a184 = 12 * a144;
        Scalar const a185 = 15 * X[1];
        Scalar const a186 = a82 * X[0];
        Scalar const a187 = a4 * a76;
        Scalar const a188 = a1 * X[1];
        Scalar const a189 = a100 * X[0];
        Scalar const a190 = 60 * a16;
        Scalar const a191 = 36 * a16;
        P[0]              = a1;
        P[1]              = -a3 + 2 * a5;
        P[2]              = a4 * a6 + (a6 * a9 + a8) * X[0];
        P[3]              = a10 * a4 + (a12 + a15) * X[0];
        P[4]              = a17 - a18 * a4 + 5 * a20;
        P[5]              = a21 * a22 + a4 * (a21 * a23 - 7 * a21) + (-a21 * a9 + 2 * a21) * X[0];
        P[6]  = a24 * a25 + a4 * (-a24 + a26 * X[1]) + (-a24 * a27 + a24 * a29 + a24) * X[0];
        P[7]  = a22 * a30 + a4 * (a30 * a35 + a30 * a9 - 7 * a30) + (-a32 - a33 - a34) * X[0];
        P[8]  = a20 + a4 * (a36 * X[2] + a37 + a38 * X[1]) + (a39 * a40 + a42 + a46) * X[0];
        P[9]  = a25 * a47 + a4 * (a48 + a50) + (a48 * a51 + a54 + a55) * X[0];
        P[10] = a4 * a58 - a56 * a57 + a56 * a61 - 63 * a59;
        P[11] = a19 * (a62 - 84) + a4 * (45 - 126 * X[1]) + a61 + (a23 - 6) * X[0];
        P[12] = a19 * (-17 * a0 + 64 * a64) + a4 * (15 * a0 - 108 * a64 + 120 * a66) + a60 * a63 +
                (24 * a0 * X[1] - a65 - 30 * a66) * X[0];
        P[13] = a19 * (a69 + 30 * a70) + a4 * (a71 - a73 + 135 * a74) + (3.0 / 2.0) * a67 +
                (210 * a56 * a75 + a69 + a73 - 270 * a74) * X[0];
        P[14] = a19 * (56 * a81 - a82 + 112 * a83) + a4 * (45 * a76 - 42 * a81 - a82 * X[2]) +
                a61 * a76 + (a78 + a80) * X[0];
        P[15] = a19 * (-51 * a56 + 144 * a70 + 48 * a84) +
                a4 * (-243 * a70 + 180 * a74 - 81 * a84 + a85 + a87 * X[2]) + 18 * a67 +
                (54 * a56 * X[1] + 18 * a56 * X[2] - a71 - a73 * X[2] - a86) * X[0];
        P[16] = a0 * a88 + a19 * (26 * a64 + a89 + 4 * a90) +
                a4 * (9 * a0 + 99 * a66 - a91 - a93 + a94 * X[2]) +
                (252 * a28 * a90 - a64 * a95 + a96 + a97) * X[0];
        P[17] = a19 * (16 * a100 + 48 * a101 - 17 * a98) +
                a4 * (-27 * a100 - 81 * a101 + 72 * a104 + 12 * a106 + a107 * X[1] + 15 * a98) +
                a60 * a99 +
                (-a101 * a23 - a102 - a103 - a105 + 6 * a98 * X[1] + 18 * a98 * X[2]) * X[0];
        P[18] = a19 * (a109 + a110 * X[2] + a2 * a23) + a2 * a88 +
                a4 * (-a112 - a114 + a115 * a2 + a116 * a53 + a117 * X[2] + 9 * a2) +
                (252 * a111 * a53 - 288 * a111 * X[2] + 252 * a118 * X[2] + a119 + a120) * X[0];
        P[19] = a16 * a88 + a19 * (a122 + a41 * X[1]) +
                a4 * (a123 * X[1] + a125 + a28 * a38 + a44) +
                (-a126 * a39 + a127 * a45 + a128 * X[1] + a130 + a131) * X[0];
        P[20] = a64;
        P[21] = a111 * a133;
        P[22] = 3 * a28 * a6 + (a7 * X[0] + a8) * X[1];
        P[23] = a10 * a28 + (a134 + a15) * X[1];
        P[24] = a136 * a137;
        P[25] = a28 * (a138 * a21 - 3 * a21) + (a139 - a140 * a21 + a141 * a21) * X[1];
        P[26] = (10.0 / 3.0) * a24 * a75 + a28 * (a132 * a24 - a26) +
                (-a142 * a24 + a24 * a4 + a24) * X[1];
        P[27] = a28 * (a143 * a30 - a30) + (a144 * a145 + a146 - a34) * X[1];
        P[28] = a18 * a75 + a28 * (9 * a17 - a38 + 15 * a39) + (a147 * X[2] + a148 + a42) * X[1];
        P[29] = a28 * (a149 + a50) + (1.0 / 3.0) * a47 * a75 + (a149 * a51 + a150 + a54) * X[1];
        P[30] = a151 * a152 * a56;
        P[31] = a28 * (a153 - 126 * X[0] + 9) + (-a153 + a154 + a155 - 6) * X[1];
        P[32] = a28 * (-108 * a1 + 96 * a156 + a92) + a75 * (-10 * a0 + 80 * a1) +
                (a0 * a157 + 30 * a1 - 51 * a156 - a65) * X[1];
        P[33] = (105.0 / 2.0) * a158 * a56 + a28 * (-a159 + a4 * a85 + a85) + a75 * (a159 - a72) +
                (a138 * a56 - 18 * a160 + a19 * a68 + a69) * X[1];
        P[34] = a28 * (a4 * a82 - 42 * a76 * X[0] + 3 * a76) +
                (a161 + 336 * a4 * a83 + a80 - 168 * a83 * X[0]) * X[1];
        P[35] = a28 * (216 * a160 - 243 * a162 + 360 * a163 + a58 - a85 * X[2]) +
                a75 * (120 * a162 - 15 * a56) + (-162 * a163 + a164 * a84 + a165 + 18 * a84) * X[1];
        P[36] = (63.0 / 2.0) * a0 * a158 +
                a28 * (a1 * a127 - 78 * a1 + a166 * a4 + a166 - 72 * a90) +
                a75 * (-66 * a0 + 66 * a1 + 84 * a90) + (-a1 * a145 + a167 + a4 * a93 + a96) * X[1];
        P[37] =
            a28 * (a102 + a107 * X[0] - 27 * a169 - a170 * X[2] + 24 * a171) +
            a75 * (a168 * a98 - a98) +
            (a101 * a164 - 162 * a101 * X[0] + 18 * a101 + 144 * a104 * X[0] - a105 + a172) * X[1];
        P[38] = (21.0 / 2.0) * a158 * a2 +
                a28 * (-144 * a173 + a174 * a4 + a174 + 126 * a2 * a53 + a3 * a95 - 54 * a3) +
                a75 * (84 * a173 - 30 * a2 + 30 * a3) +
                (a119 - a127 * a3 + a175 * a53 + a177 + 36 * a5 * X[2]) * X[1];
        P[39] = (3.0 / 2.0) * a158 * a16 + a28 * (a125 - a178 + a179 * X[2] + a38 * a4) +
                a75 * (a122 + a147) +
                (a123 * a4 + a130 + 180 * a17 * a53 - a17 * a95 + a180) * X[1];
        P[40] = a90;
        P[41] = a133 * a173;
        P[42] = a182 * a7;
        P[43] = a11 * a53 + (a12 + a13 + a134) * X[2];
        P[44] = a136 * a39;
        P[45] = a139 * a182 * (a143 - 1);
        P[46] = a183 * a24;
        P[47] = a53 * (a184 + a32) + (a146 + a184 * X[1] - a33) * X[2];
        P[48] = a53 * (a16 * a185 + 3 * a17 + a37) + (a148 + a17 * a23 + a36 + a46) * X[2];
        P[49] = 2 * a129 * a47 + a53 * (3 * a149 + 3 * a48 - a49) +
                (a142 * a48 + a150 + a47 + a55) * X[2];
        P[50] = 3 * a151 * a84;
        P[51] = a181 * a51 * (-a140 + 28 * a4 + 1);
        P[52] = a183 * a65 * (a168 - 1);
        P[53] = a68 *
                (a115 * X[0] - a115 + a135 * X[1] + a185 + a19 - 3 * a4 - a40 * X[0] + a57 +
                 35 * a75 - 1) *
                X[2];
        P[54] =
            a53 * (-a186 + 168 * a187 + a77) + (a161 - a186 * X[1] + a187 * a62 + a78 + a79) * X[2];
        P[55] =
            a53 * (72 * a160 - 81 * a162 + a71 - a85 * X[1] + a87 * X[0]) +
            (a165 + 432 * a4 * a70 - 486 * a70 * X[0] + 54 * a70 + 360 * a74 * X[0] - a86) * X[2];
        P[56] = a53 * (-12 * a1 + 72 * a188 + a4 * a63 + a63 + 126 * a66 - a94) +
                (198 * a1 * a28 + a167 - 156 * a188 + a4 * a91 + a89 + a97) * X[2];
        P[57] = a129 * (48 * a169 - a99) +
                a53 * (-81 * a169 - a170 * X[1] + a170 + 72 * a171 + 72 * a189) +
                (48 * a100 * a4 - a103 + 24 * a106 * X[0] + a172 - 54 * a189 + a99 * X[1]) * X[2];
        P[58] = a129 * (-a110 + 84 * a111 + 12 * a3) +
                a53 * (a116 - a117 + 126 * a118 + a126 * a3 - a175 + a176) +
                (a109 + a120 + a177 + 90 * a28 * a3 - 108 * a3 * X[1] + 54 * a5 * X[1]) * X[2];
        P[59] = a129 * (60 * a17 + a190 * X[1] - a190) +
                30 * a16 * ((X[2]) * (X[2]) * (X[2]) * (X[2])) +
                a53 * (-72 * a137 + a179 * X[1] - a179 + a191 * a28 + a191 * a4 + a191) +
                (a121 + a131 - 36 * a17 * X[1] + a178 * a28 + a180 + a4 * a43) * X[2];
        return Pm;
    }
};

template <>
class OrthonormalBasis<3, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 35;

    [[maybe_unused]] Vector<kSize> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Vector<kSize> P;
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = std::sqrt(10);
        Scalar const a2  = 4 * X[0];
        Scalar const a3  = X[0] - 1;
        Scalar const a4  = a3 + 3 * X[1];
        Scalar const a5  = 2 * a4;
        Scalar const a6  = a3 + X[1] + 2 * X[2];
        Scalar const a7  = 2 * a6;
        Scalar const a8  = std::sqrt(14);
        Scalar const a9  = 10 * X[0];
        Scalar const a10 = ((X[0]) * (X[0]));
        Scalar const a11 = 15 * a10;
        Scalar const a12 = 6 * X[0];
        Scalar const a13 = a12 - 1;
        Scalar const a14 = 8 * X[1];
        Scalar const a15 = ((X[1]) * (X[1]));
        Scalar const a16 = 2 * X[0];
        Scalar const a17 = a10 - a16 + 1;
        Scalar const a18 = a14 * X[0] - a14 + 10 * a15 + a17;
        Scalar const a19 = a6 * a8;
        Scalar const a20 = a3 + 5 * X[1];
        Scalar const a21 = 3 * a20;
        Scalar const a22 = 6 * X[2];
        Scalar const a23 = ((X[2]) * (X[2]));
        Scalar const a24 = a15 + a16 * X[1] + a17 - 2 * X[1];
        Scalar const a25 = a12 * X[2] + a22 * X[1] - a22 + 6 * a23 + a24;
        Scalar const a26 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a27 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a28 = 168 * a10 - 84 * X[0] + 6;
        Scalar const a29 = 8 * X[0] - 1;
        Scalar const a30 = 3 * a29;
        Scalar const a31 = 45 * a15;
        Scalar const a32 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a33 = X[0] * X[1];
        Scalar const a34 = 3 * X[0] - 1;
        Scalar const a35 = -3 * a10 + a27 + a34;
        Scalar const a36 = a11 * X[1] + a31 * X[0] - a31 + 35 * a32 - 30 * a33 + a35 + 15 * X[1];
        Scalar const a37 = 12 * X[1];
        Scalar const a38 = a37 * X[0];
        Scalar const a39 = 21 * a15 + a17 - a37 + a38;
        Scalar const a40 = a25 * (a3 + 7 * X[1]);
        Scalar const a41 = 10 * X[2];
        Scalar const a42 = 10 * a23 + a24 + a41 * X[1] - a41 + a9 * X[2];
        Scalar const a43 = std::sqrt(22);
        Scalar const a44 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a45 = -108 * a10 + 120 * a27 + 24 * X[0] - 1;
        Scalar const a46 = std::sqrt(66);
        Scalar const a47 = a34 * (15 * X[0] - 1);
        Scalar const a48 = a9 - 1;
        Scalar const a49 = 2 * a48;
        Scalar const a50 = std::sqrt(110);
        Scalar const a51 = 224 * a32;
        Scalar const a52 = 24 * X[1];
        Scalar const a53 = 126 * a15;
        Scalar const a54 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a55 = a15 * X[0];
        Scalar const a56 = a10 * X[1];
        Scalar const a57 = 4 * a27;
        Scalar const a58 = 6 * a10;
        Scalar const a59 = -a2 + a44 - a57 + a58 + 1;
        Scalar const a60 = a48 * a7;
        Scalar const a61 = std::sqrt(330);
        Scalar const a62 = 84 * a15;
        Scalar const a63 = 21 * X[1];
        Scalar const a64 = 16 * X[1];
        Scalar const a65 = 140 * ((X[2]) * (X[2]) * (X[2]));
        Scalar const a66 = 20 * X[2];
        Scalar const a67 = 90 * a23;
        Scalar const a68 = 180 * a23;
        Scalar const a69 = 60 * X[2];
        P[0]             = a0;
        P[1]             = a1 * (a2 - 1);
        P[2]             = std::sqrt(5) * a5;
        P[3]             = std::sqrt(15) * a7;
        P[4]             = a8 * (a11 - a9 + 1);
        P[5]             = std::sqrt(7) * a13 * a5;
        P[6]             = std::sqrt(42) * a18;
        P[7]             = std::sqrt(21) * a13 * a7;
        P[8]             = a19 * a21;
        P[9]             = std::sqrt(210) * a25;
        P[10]            = 3 * a26 * (-63 * a10 + 56 * a27 + 18 * X[0] - 1);
        P[11]            = a28 * a4;
        P[12]            = a0 * a18 * a30;
        P[13]            = 6 * a26 * a36;
        P[14]            = std::numbers::sqrt3_v<Scalar> * a28 * a6;
        P[15]            = 9 * a20 * a26 * a29 * a6;
        P[16]            = 6 * a0 * a39 * a6;
        P[17]            = std::sqrt(30) * a25 * a30;
        P[18]            = 6 * a1 * a40;
        P[19]            = 6 * a19 * a42;
        P[20]            = a43 * (168 * a10 - 336 * a27 + 210 * a44 - 28 * X[0] + 1);
        P[21]            = std::sqrt(11) * a45 * a5;
        P[22]            = a18 * a46 * a47;
        P[23]            = a36 * a43 * a49;
        P[24] = a50 * (a10 * a53 + a27 * a52 + 72 * a33 + a51 * X[0] - a51 - a52 + a53 + 126 * a54 -
                       252 * a55 - 72 * a56 + a59);
        P[25] = std::sqrt(33) * a45 * a7;
        P[26] = a21 * a43 * a47 * a6;
        P[27] = a39 * a46 * a60;
        P[28] = a6 * a61 * (a10 * a63 + 84 * a32 - 42 * a33 + a35 + a62 * X[0] - a62 + a63);
        P[29] = a25 * a47 * a61;
        P[30] = a40 * a49 * a50;
        P[31] = 5 * a25 * a43 * (36 * a15 + a17 + a64 * X[0] - a64);
        P[32] = std::sqrt(154) * a42 * a60;
        P[33] = std::sqrt(770) * a42 * a6 * (a3 + 9 * X[1]);
        P[34] = 3 * a50 *
                (-a10 * a37 + a10 * a67 - a10 * a69 + a15 * a58 + a15 * a67 - a15 * a69 + 6 * a15 +
                 a2 * a32 + a27 * a66 + a32 * a66 - 4 * a32 + a33 * a68 - 120 * a33 * X[2] + a38 +
                 a54 + a55 * a69 - 12 * a55 + a56 * a69 + a57 * X[1] + a59 + a65 * X[0] +
                 a65 * X[1] - a65 - a66 + a67 - a68 * X[0] - a68 * X[1] + a69 * X[0] + a69 * X[1] -
                 4 * X[1] + 70 * ((X[2]) * (X[2]) * (X[2]) * (X[2])));
        return P;
    }

    [[maybe_unused]] Matrix<kDims, kSize> derivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kDims, kSize> Gm;
        Scalar* G         = Gm.data();
        Scalar const a0   = std::sqrt(10);
        Scalar const a1   = std::sqrt(5);
        Scalar const a2   = std::sqrt(15);
        Scalar const a3   = 2 * a2;
        Scalar const a4   = std::sqrt(14);
        Scalar const a5   = 30 * X[0];
        Scalar const a6   = std::sqrt(7);
        Scalar const a7   = 6 * X[0];
        Scalar const a8   = a7 - 1;
        Scalar const a9   = a6 * a8;
        Scalar const a10  = X[0] - 1;
        Scalar const a11  = a10 + 3 * X[1];
        Scalar const a12  = std::sqrt(42);
        Scalar const a13  = 8 * X[1];
        Scalar const a14  = 2 * X[0];
        Scalar const a15  = a14 - 2;
        Scalar const a16  = a13 + a15;
        Scalar const a17  = 8 * X[0];
        Scalar const a18  = a17 + 20 * X[1] - 8;
        Scalar const a19  = std::sqrt(21);
        Scalar const a20  = a19 * a8;
        Scalar const a21  = 2 * a20;
        Scalar const a22  = a10 + X[1] + 2 * X[2];
        Scalar const a23  = a10 + 5 * X[1];
        Scalar const a24  = 3 * a4;
        Scalar const a25  = a23 * a24;
        Scalar const a26  = a22 * a4;
        Scalar const a27  = 6 * a4;
        Scalar const a28  = std::sqrt(210);
        Scalar const a29  = 6 * X[2];
        Scalar const a30  = 2 * X[1];
        Scalar const a31  = a15 + a30;
        Scalar const a32  = a29 + a31;
        Scalar const a33  = a28 * a32;
        Scalar const a34  = a7 + 6 * X[1] + 12 * X[2] - 6;
        Scalar const a35  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a36  = ((X[0]) * (X[0]));
        Scalar const a37  = 168 * a36;
        Scalar const a38  = 84 * X[0];
        Scalar const a39  = 336 * X[0];
        Scalar const a40  = 252 * X[0];
        Scalar const a41  = std::sqrt(6);
        Scalar const a42  = a17 - 1;
        Scalar const a43  = 3 * a42;
        Scalar const a44  = a41 * a43;
        Scalar const a45  = ((X[1]) * (X[1]));
        Scalar const a46  = -a14 + a36 + 1;
        Scalar const a47  = a13 * X[0] - a13 + 10 * a45 + a46;
        Scalar const a48  = 45 * a45;
        Scalar const a49  = a5 * X[1];
        Scalar const a50  = 3 * a36;
        Scalar const a51  = a50 - a7 + 3;
        Scalar const a52  = a48 + a49 + a51 - 30 * X[1];
        Scalar const a53  = 6 * a35;
        Scalar const a54  = 90 * X[1];
        Scalar const a55  = 15 * a36;
        Scalar const a56  = 105 * a45 - a5 + a54 * X[0] - a54 + a55 + 15;
        Scalar const a57  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a58  = a57 * (28 * a36 - 14 * X[0] + 1);
        Scalar const a59  = 6 * a58;
        Scalar const a60  = 6 * a22;
        Scalar const a61  = a23 * a35;
        Scalar const a62  = 9 * a42;
        Scalar const a63  = a61 * a62;
        Scalar const a64  = a22 * a35;
        Scalar const a65  = 12 * X[1];
        Scalar const a66  = a15 + a65;
        Scalar const a67  = a41 * a60;
        Scalar const a68  = 21 * a45 + a46 + a65 * X[0] - a65;
        Scalar const a69  = a41 * a68;
        Scalar const a70  = 6 * a69;
        Scalar const a71  = 12 * X[0];
        Scalar const a72  = 42 * X[1];
        Scalar const a73  = a71 + a72 - 12;
        Scalar const a74  = std::sqrt(30);
        Scalar const a75  = a43 * a74;
        Scalar const a76  = a32 * a75;
        Scalar const a77  = ((X[2]) * (X[2]));
        Scalar const a78  = a14 * X[1] - a30 + a45 + a46;
        Scalar const a79  = a29 * X[1] - a29 + a7 * X[2] + 6 * a77 + a78;
        Scalar const a80  = a10 + 7 * X[1];
        Scalar const a81  = a32 * a80;
        Scalar const a82  = 6 * a0;
        Scalar const a83  = a81 * a82;
        Scalar const a84  = a0 * a79;
        Scalar const a85  = a34 * a80;
        Scalar const a86  = 10 * X[2];
        Scalar const a87  = a31 + a86;
        Scalar const a88  = 6 * a26;
        Scalar const a89  = 10 * a77 + a78 + a86 * X[0] + a86 * X[1] - a86;
        Scalar const a90  = a27 * a89 + a87 * a88;
        Scalar const a91  = 10 * X[0];
        Scalar const a92  = a91 + 10 * X[1] + 20 * X[2] - 10;
        Scalar const a93  = std::sqrt(22);
        Scalar const a94  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a95  = std::sqrt(11);
        Scalar const a96  = 24 * X[0];
        Scalar const a97  = -108 * a36 + 120 * a94 + a96 - 1;
        Scalar const a98  = a95 * a97;
        Scalar const a99  = 720 * a36 - 432 * X[0] + 48;
        Scalar const a100 = std::sqrt(66);
        Scalar const a101 = 15 * X[0] - 1;
        Scalar const a102 = 3 * X[0] - 1;
        Scalar const a103 = a101 * a102;
        Scalar const a104 = a100 * a103;
        Scalar const a105 = a100 * a47;
        Scalar const a106 = 15 * a102;
        Scalar const a107 = 3 * a101;
        Scalar const a108 = a91 - 1;
        Scalar const a109 = 2 * a108;
        Scalar const a110 = a109 * a93;
        Scalar const a111 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a112 = a102 - a50 + a94;
        Scalar const a113 = std::sqrt(110);
        Scalar const a114 = 72 * X[1];
        Scalar const a115 = 252 * a45;
        Scalar const a116 = X[0] * X[1];
        Scalar const a117 = -12 * a36 + a71 + 4 * a94 - 4;
        Scalar const a118 = 252 * X[1];
        Scalar const a119 = 672 * a45;
        Scalar const a120 = std::sqrt(33);
        Scalar const a121 = a120 * a97;
        Scalar const a122 = 2 * a121;
        Scalar const a123 = a23 * a93;
        Scalar const a124 = a102 * a107;
        Scalar const a125 = a123 * a124;
        Scalar const a126 = a22 * a93;
        Scalar const a127 = a123 * a22;
        Scalar const a128 = a100 * a109 * a22;
        Scalar const a129 = a100 * a68;
        Scalar const a130 = a109 * a129;
        Scalar const a131 = 20 * a22;
        Scalar const a132 = 4 * a108;
        Scalar const a133 = 84 * a45;
        Scalar const a134 = a72 * X[0];
        Scalar const a135 = std::sqrt(330);
        Scalar const a136 = a135 * a22;
        Scalar const a137 = 21 * X[1];
        Scalar const a138 = a135 * (84 * a111 + a112 - a133 - a134 + a137 * a36 + a137 + a38 * a45);
        Scalar const a139 = 168 * X[1];
        Scalar const a140 = a103 * a135;
        Scalar const a141 = a140 * a32;
        Scalar const a142 = a135 * a79;
        Scalar const a143 = a109 * a113;
        Scalar const a144 = a143 * a81;
        Scalar const a145 = a113 * a79;
        Scalar const a146 = 16 * X[1];
        Scalar const a147 = 5 * a93;
        Scalar const a148 = a147 * (a146 * X[0] - a146 + 36 * a45 + a46);
        Scalar const a149 = a148 * a32;
        Scalar const a150 = a147 * a79;
        Scalar const a151 = std::sqrt(154);
        Scalar const a152 = a151 * a89;
        Scalar const a153 = a22 * a87;
        Scalar const a154 = a109 * a151;
        Scalar const a155 = a109 * a152 + a153 * a154;
        Scalar const a156 = a22 * a92;
        Scalar const a157 = std::sqrt(770);
        Scalar const a158 = a157 * a89;
        Scalar const a159 = a158 * a22;
        Scalar const a160 = a10 + 9 * X[1];
        Scalar const a161 = a158 * a160;
        Scalar const a162 = a157 * a160;
        Scalar const a163 = a153 * a162 + a161;
        Scalar const a164 = 180 * a77;
        Scalar const a165 = 60 * X[2];
        Scalar const a166 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a167 = 120 * X[2];
        Scalar const a168 = 3 * a113;
        Scalar const a169 =
            a168 * (4 * a111 + a116 * a167 + a117 + a164 * X[0] + a164 * X[1] - a164 + a165 * a36 +
                    a165 * a45 + a165 + 140 * a166 - a167 * X[0] - a167 * X[1] + a36 * a65 +
                    a45 * a71 - 12 * a45 + a65 - a96 * X[1]);
        Scalar const a170 = 180 * X[2];
        Scalar const a171 = 60 * a36;
        Scalar const a172 = 60 * a45;
        Scalar const a173 = 420 * a77;
        Scalar const a174 = 360 * X[2];
        G[0]              = 0;
        G[1]              = 0;
        G[2]              = 0;
        G[3]              = 4 * a0;
        G[4]              = 0;
        G[5]              = 0;
        G[6]              = 2 * a1;
        G[7]              = 6 * a1;
        G[8]              = 0;
        G[9]              = a3;
        G[10]             = a3;
        G[11]             = 4 * a2;
        G[12]             = a4 * (a5 - 10);
        G[13]             = 0;
        G[14]             = 0;
        G[15]             = 12 * a11 * a6 + 2 * a9;
        G[16]             = 6 * a9;
        G[17]             = 0;
        G[18]             = a12 * a16;
        G[19]             = a12 * a18;
        G[20]             = 0;
        G[21]             = 12 * a19 * a22 + a21;
        G[22]             = a21;
        G[23]             = 4 * a20;
        G[24]             = a22 * a24 + a25;
        G[25]             = a25 + 15 * a26;
        G[26]             = a23 * a27;
        G[27]             = a33;
        G[28]             = a33;
        G[29]             = a28 * a34;
        G[30]             = 3 * a35 * (a37 - 126 * X[0] + 18);
        G[31]             = 0;
        G[32]             = 0;
        G[33]             = a11 * (a39 - 84) + a37 - a38 + 6;
        G[34]             = 504 * a36 - a40 + 18;
        G[35]             = 0;
        G[36]             = a16 * a44 + 24 * a41 * a47;
        G[37]             = a18 * a44;
        G[38]             = 0;
        G[39]             = a52 * a53;
        G[40]             = a53 * a56;
        G[41]             = 0;
        G[42]             = a57 * a60 * (56 * X[0] - 14) + a59;
        G[43]             = a59;
        G[44]             = 12 * a58;
        G[45]             = 72 * a22 * a61 + a62 * a64 + a63;
        G[46]             = 45 * a42 * a64 + a63;
        G[47]             = 18 * a42 * a61;
        G[48]             = a66 * a67 + a70;
        G[49]             = a67 * a73 + a70;
        G[50]             = 12 * a69;
        G[51]             = 24 * a74 * a79 + a76;
        G[52]             = a76;
        G[53]             = a34 * a75;
        G[54]             = a83 + 6 * a84;
        G[55]             = a83 + 42 * a84;
        G[56]             = a82 * a85;
        G[57]             = a90;
        G[58]             = a90;
        G[59]             = 12 * a4 * a89 + a88 * a92;
        G[60]             = a93 * (-1008 * a36 + a39 + 840 * a94 - 28);
        G[61]             = 0;
        G[62]             = 0;
        G[63]             = a11 * a95 * a99 + 2 * a98;
        G[64]             = 6 * a98;
        G[65]             = 0;
        G[66]             = a104 * a16 + a105 * a106 + a105 * a107;
        G[67]             = a104 * a18;
        G[68]             = 0;
        G[69]             = a110 * a52 +
                20 * a93 * (35 * a111 + a112 + a48 * X[0] - a48 - a49 + a55 * X[1] + 15 * X[1]);
        G[70]  = a110 * a56;
        G[71]  = 0;
        G[72]  = a113 * (224 * a111 + a114 * a36 + a114 - a115 - 144 * a116 + a117 + a40 * a45);
        G[73]  = a113 * (504 * a111 - 504 * a116 + a118 * a36 + a118 + a119 * X[0] - a119 -
                        72 * a36 + 24 * a94 + 72 * X[0] - 24);
        G[74]  = 0;
        G[75]  = a120 * a22 * a99 + a122;
        G[76]  = a122;
        G[77]  = 4 * a121;
        G[78]  = 9 * a101 * a127 + 45 * a102 * a127 + a124 * a126 + a125;
        G[79]  = a101 * a106 * a126 + a125;
        G[80]  = 6 * a103 * a123;
        G[81]  = a128 * a66 + a129 * a131 + a130;
        G[82]  = a128 * a73 + a130;
        G[83]  = a129 * a132;
        G[84]  = a136 * (a133 + a134 + a51 - a72) + a138;
        G[85]  = a136 * (a115 + a139 * X[0] - a139 + 21 * a36 - 42 * X[0] + 21) + a138;
        G[86]  = 2 * a138;
        G[87]  = a106 * a142 + a107 * a142 + a141;
        G[88]  = a141;
        G[89]  = a140 * a34;
        G[90]  = a109 * a145 + a144 + 20 * a145 * a80;
        G[91]  = 14 * a108 * a145 + a144;
        G[92]  = a143 * a85;
        G[93]  = a149 + a150 * (a146 + a15);
        G[94]  = a149 + a150 * (a114 + 16 * X[0] - 16);
        G[95]  = a148 * a34;
        G[96]  = a131 * a152 + a155;
        G[97]  = a155;
        G[98]  = a132 * a152 + a154 * a156;
        G[99]  = a159 + a163;
        G[100] = 9 * a159 + a163;
        G[101] = a156 * a162 + 2 * a161;
        G[102] = a169;
        G[103] = a169;
        G[104] =
            a168 * (20 * a111 + a116 * a174 - 120 * a116 + 280 * a166 + a170 * a36 + a170 * a45 +
                    a170 + a171 * X[1] - a171 + a172 * X[0] - a172 + a173 * X[0] + a173 * X[1] -
                    a173 - a174 * X[0] - a174 * X[1] + 20 * a94 + 60 * X[0] + 60 * X[1] - 20);
        return Gm;
    }

    [[maybe_unused]] Matrix<kSize, kDims>
    antiderivatives([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P         = Pm.data();
        Scalar const a0   = std::sqrt(6);
        Scalar const a1   = a0 * X[0];
        Scalar const a2   = std::sqrt(10);
        Scalar const a3   = a2 * X[0];
        Scalar const a4   = ((X[0]) * (X[0]));
        Scalar const a5   = a2 * a4;
        Scalar const a6   = std::sqrt(5);
        Scalar const a7   = 2 * a6;
        Scalar const a8   = -a7;
        Scalar const a9   = 6 * X[1];
        Scalar const a10  = std::sqrt(15);
        Scalar const a11  = 2 * a10;
        Scalar const a12  = a11 * X[1];
        Scalar const a13  = -a11;
        Scalar const a14  = 4 * X[2];
        Scalar const a15  = a10 * a14 + a13;
        Scalar const a16  = std::sqrt(14);
        Scalar const a17  = a16 * X[0];
        Scalar const a18  = 5 * a16;
        Scalar const a19  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a20  = a16 * a19;
        Scalar const a21  = std::sqrt(7);
        Scalar const a22  = 4 * a19;
        Scalar const a23  = 18 * X[1];
        Scalar const a24  = std::sqrt(42);
        Scalar const a25  = (1.0 / 3.0) * a19;
        Scalar const a26  = 4 * a24;
        Scalar const a27  = 8 * X[1];
        Scalar const a28  = ((X[1]) * (X[1]));
        Scalar const a29  = 10 * a28;
        Scalar const a30  = std::sqrt(21);
        Scalar const a31  = 2 * a30;
        Scalar const a32  = -a31;
        Scalar const a33  = a31 * X[1];
        Scalar const a34  = a14 * a30;
        Scalar const a35  = 12 * X[2];
        Scalar const a36  = 3 * a16;
        Scalar const a37  = -a36;
        Scalar const a38  = 9 * a16;
        Scalar const a39  = a16 * X[2];
        Scalar const a40  = a39 * X[1];
        Scalar const a41  = 6 * a16;
        Scalar const a42  = a36 - a41 * X[2];
        Scalar const a43  = a16 * a23;
        Scalar const a44  = -a43;
        Scalar const a45  = a16 * a28;
        Scalar const a46  = a44 + 15 * a45;
        Scalar const a47  = std::sqrt(210);
        Scalar const a48  = a47 * X[1];
        Scalar const a49  = 3 * a47;
        Scalar const a50  = -a47 + a49 * X[2];
        Scalar const a51  = 6 * X[2];
        Scalar const a52  = 6 * a47;
        Scalar const a53  = ((X[2]) * (X[2]));
        Scalar const a54  = a47 + a52 * a53 - a52 * X[2];
        Scalar const a55  = a28 * a47 - 2 * a48;
        Scalar const a56  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a57  = 3 * X[0];
        Scalar const a58  = 27 * a56;
        Scalar const a59  = a19 * a56;
        Scalar const a60  = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a61  = 42 * a60;
        Scalar const a62  = 126 * X[1];
        Scalar const a63  = 168 * X[1];
        Scalar const a64  = 6 * a0;
        Scalar const a65  = a0 * X[1];
        Scalar const a66  = 3 * a0;
        Scalar const a67  = a0 * a28;
        Scalar const a68  = a56 * a60;
        Scalar const a69  = 6 * a56;
        Scalar const a70  = -a69;
        Scalar const a71  = a56 * X[1];
        Scalar const a72  = 9 * a56;
        Scalar const a73  = 90 * a56;
        Scalar const a74  = a73 * X[1];
        Scalar const a75  = a28 * a56;
        Scalar const a76  = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a77  = 210 * a76;
        Scalar const a78  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a79  = 6 * a78;
        Scalar const a80  = a79 * X[1];
        Scalar const a81  = -a79;
        Scalar const a82  = a35 * a78 + a81;
        Scalar const a83  = a78 * X[1];
        Scalar const a84  = 84 * a78;
        Scalar const a85  = a78 * X[2];
        Scalar const a86  = a56 * X[2];
        Scalar const a87  = 45 * a56;
        Scalar const a88  = a28 * a87;
        Scalar const a89  = 360 * a71;
        Scalar const a90  = (3.0 / 2.0) * a60;
        Scalar const a91  = -a64;
        Scalar const a92  = a0 * X[2];
        Scalar const a93  = 78 * a65;
        Scalar const a94  = 12 * a0;
        Scalar const a95  = a94 * X[2];
        Scalar const a96  = 72 * a65;
        Scalar const a97  = 144 * X[2];
        Scalar const a98  = 252 * a28;
        Scalar const a99  = a91 + a95;
        Scalar const a100 = 126 * a0 * a76 - 198 * a67 + a93;
        Scalar const a101 = std::sqrt(30);
        Scalar const a102 = 6 * a101;
        Scalar const a103 = a101 * X[1];
        Scalar const a104 = a101 * X[2];
        Scalar const a105 = 3 * a101;
        Scalar const a106 = a105 * a28;
        Scalar const a107 = a101 * a53;
        Scalar const a108 = 18 * a107;
        Scalar const a109 = a101 * a28;
        Scalar const a110 = 72 * X[1];
        Scalar const a111 = 6 * a2;
        Scalar const a112 = -a111;
        Scalar const a113 = 12 * a2;
        Scalar const a114 = a2 * X[1];
        Scalar const a115 = 54 * a114;
        Scalar const a116 = 36 * a2;
        Scalar const a117 = a116 * X[2];
        Scalar const a118 = 45 * a28;
        Scalar const a119 = 18 * a2;
        Scalar const a120 = 144 * a114;
        Scalar const a121 = a2 * X[2];
        Scalar const a122 = a112 - a116 * a53 + a117;
        Scalar const a123 = 90 * a28;
        Scalar const a124 = 42 * a76;
        Scalar const a125 = a115 - a123 * a2 + a124 * a2;
        Scalar const a126 = -a41;
        Scalar const a127 = a126 + 24 * a39;
        Scalar const a128 = 72 * a39;
        Scalar const a129 = a16 * a53;
        Scalar const a130 = -a128 + 90 * a129 + a38;
        Scalar const a131 = 72 * X[2];
        Scalar const a132 = 180 * a129;
        Scalar const a133 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a134 = a126 + a128 - a132 + 120 * a133 * a16;
        Scalar const a135 = a41 * a76 + a43 - 18 * a45;
        Scalar const a136 = std::sqrt(22);
        Scalar const a137 = a136 * X[0];
        Scalar const a138 = a136 * a4;
        Scalar const a139 = 56 * a19;
        Scalar const a140 = a136 * a60;
        Scalar const a141 = std::pow(X[0], 5);
        Scalar const a142 = a136 * a141;
        Scalar const a143 = std::sqrt(11);
        Scalar const a144 = 48 * a141;
        Scalar const a145 = a143 * X[1];
        Scalar const a146 = std::sqrt(66);
        Scalar const a147 = 9 * a141;
        Scalar const a148 = a146 * X[1];
        Scalar const a149 = 10 * a146;
        Scalar const a150 = a146 * a28;
        Scalar const a151 = a136 * X[1];
        Scalar const a152 = a136 * a28;
        Scalar const a153 = 30 * a136;
        Scalar const a154 = a153 * X[1];
        Scalar const a155 = a136 * a76;
        Scalar const a156 = std::sqrt(110);
        Scalar const a157 = (1.0 / 5.0) * a141;
        Scalar const a158 = 6 * a156;
        Scalar const a159 = 2 * a156;
        Scalar const a160 = 24 * X[1];
        Scalar const a161 = -a156 * a160;
        Scalar const a162 = 42 * a156;
        Scalar const a163 = a162 * a28;
        Scalar const a164 = -a159;
        Scalar const a165 = a156 * X[1];
        Scalar const a166 = 36 * a165;
        Scalar const a167 = 126 * a28;
        Scalar const a168 = a156 * a167;
        Scalar const a169 = a156 * a76;
        Scalar const a170 = 224 * a76;
        Scalar const a171 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a172 = 126 * a171;
        Scalar const a173 = std::sqrt(33);
        Scalar const a174 = 2 * a173;
        Scalar const a175 = -a174;
        Scalar const a176 = a174 * X[1];
        Scalar const a177 = a14 * a173;
        Scalar const a178 = a173 * X[1];
        Scalar const a179 = a173 * X[2];
        Scalar const a180 = a136 * X[2];
        Scalar const a181 = 3 * a136;
        Scalar const a182 = -6 * a180 + a181;
        Scalar const a183 = 15 * a136;
        Scalar const a184 = -a136 * a23 + a183 * a28;
        Scalar const a185 = a151 * X[2];
        Scalar const a186 = 4 * a146;
        Scalar const a187 = a148 * X[2];
        Scalar const a188 = a146 * X[2];
        Scalar const a189 = 22 * a146 - 28 * a188;
        Scalar const a190 = 2 * a146;
        Scalar const a191 = -a190;
        Scalar const a192 = 26 * a148;
        Scalar const a193 = a186 * X[2];
        Scalar const a194 = a124 * a146;
        Scalar const a195 = a150 * X[2];
        Scalar const a196 = -13 * a146 + 24 * a188;
        Scalar const a197 = std::sqrt(330);
        Scalar const a198 = -a197;
        Scalar const a199 = a197 * X[1];
        Scalar const a200 = a197 * X[2];
        Scalar const a201 = 2 * a197;
        Scalar const a202 = a197 * a28;
        Scalar const a203 = a199 * X[2];
        Scalar const a204 = -22 * a199;
        Scalar const a205 = a201 * X[2];
        Scalar const a206 = -a205;
        Scalar const a207 = a204 + a206;
        Scalar const a208 = 3 * a197;
        Scalar const a209 = a208 * X[2];
        Scalar const a210 = 105 * a202;
        Scalar const a211 = a197 * a76;
        Scalar const a212 = 84 * a211;
        Scalar const a213 = 42 * a197;
        Scalar const a214 = a213 * X[1];
        Scalar const a215 = a214 * X[2];
        Scalar const a216 = 84 * a202;
        Scalar const a217 = 168 * a211;
        Scalar const a218 = 84 * a171 * a197 + a197 + a210 - a217;
        Scalar const a219 = a197 * a53;
        Scalar const a220 = a197 - 6 * a200 + 6 * a219;
        Scalar const a221 = -a201 * X[1] + a202;
        Scalar const a222 = 4 * a156;
        Scalar const a223 = 30 * a156;
        Scalar const a224 = a156 * X[2];
        Scalar const a225 = a156 * a28;
        Scalar const a226 = a156 * a53;
        Scalar const a227 = a165 * X[2];
        Scalar const a228 = a156 * a23;
        Scalar const a229 = 12 * a156;
        Scalar const a230 = a229 * X[2];
        Scalar const a231 = 14 * a169;
        Scalar const a232 = a165 * a53;
        Scalar const a233 = a225 * X[2];
        Scalar const a234 = 5 * a136;
        Scalar const a235 = 10 * a136;
        Scalar const a236 = 115 * a136;
        Scalar const a237 = 90 * a151;
        Scalar const a238 = -a237;
        Scalar const a239 = a153 * X[2];
        Scalar const a240 = -a239;
        Scalar const a241 = a238 + a240;
        Scalar const a242 = 45 * a136;
        Scalar const a243 = 345 * a152;
        Scalar const a244 = a153 * a53;
        Scalar const a245 = 510 * a185;
        Scalar const a246 = a151 * a53;
        Scalar const a247 = 780 * a152;
        Scalar const a248 = a234 + a244;
        Scalar const a249 = 180 * a136 * a171 - 440 * a155 + a243;
        Scalar const a250 = std::sqrt(154);
        Scalar const a251 = 15 * X[1];
        Scalar const a252 = a250 * X[2];
        Scalar const a253 = 22 * a250;
        Scalar const a254 = a250 * X[1];
        Scalar const a255 = 20 * a250;
        Scalar const a256 = a250 * a53;
        Scalar const a257 = a254 * X[2];
        Scalar const a258 = a250 * a9;
        Scalar const a259 = 24 * a252;
        Scalar const a260 = 2 * a250;
        Scalar const a261 = a260 * a76;
        Scalar const a262 = a133 * a250;
        Scalar const a263 = 40 * a262;
        Scalar const a264 = 60 * a256;
        Scalar const a265 = a250 * a28;
        Scalar const a266 = a265 * X[2];
        Scalar const a267 = 10 * a250;
        Scalar const a268 = 300 * a256;
        Scalar const a269 = std::sqrt(770);
        Scalar const a270 = 3 * X[1];
        Scalar const a271 = a269 * X[2];
        Scalar const a272 = 2 * a269;
        Scalar const a273 = 10 * a269;
        Scalar const a274 = a271 * X[1];
        Scalar const a275 = a269 * X[1];
        Scalar const a276 = 12 * a275;
        Scalar const a277 = -a276;
        Scalar const a278 = a269 * a35;
        Scalar const a279 = -a278;
        Scalar const a280 = a277 + a279;
        Scalar const a281 = 30 * a269;
        Scalar const a282 = a28 * a281;
        Scalar const a283 = a269 * a76;
        Scalar const a284 = a281 * a53;
        Scalar const a285 = 132 * a274;
        Scalar const a286 = a275 * a53;
        Scalar const a287 = 114 * a28;
        Scalar const a288 = a269 * a53;
        Scalar const a289 = 270 * a28;
        Scalar const a290 = a133 * a269;
        Scalar const a291 = 20 * a290;
        Scalar const a292 = a269 + a284 - a291;
        Scalar const a293 = 9 * a171 * a269 + a282 - 28 * a283;
        Scalar const a294 = (3.0 / 5.0) * a156;
        Scalar const a295 = 3 * a156;
        Scalar const a296 = 15 * a224 - a295;
        Scalar const a297 = 60 * a224;
        Scalar const a298 = -a297;
        Scalar const a299 = a229 * X[1];
        Scalar const a300 = -a299;
        Scalar const a301 = a298 + a300;
        Scalar const a302 = 90 * a156;
        Scalar const a303 = a158 + a302 * a53;
        Scalar const a304 = 18 * a225;
        Scalar const a305 = 180 * a165;
        Scalar const a306 = a305 * X[2];
        Scalar const a307 = a302 * X[2];
        Scalar const a308 = -a158;
        Scalar const a309 = 270 * a226;
        Scalar const a310 = a133 * a156;
        Scalar const a311 = a307 + a308 - a309 + 210 * a310;
        Scalar const a312 = 420 * a310;
        Scalar const a313 = ((X[2]) * (X[2]) * (X[2]) * (X[2]));
        Scalar const a314 = 210 * a156 * a313 + a295 + a309 - a312;
        Scalar const a315 = a229 * a76;
        Scalar const a316 = a171 * a295 + a304 - a315;
        Scalar const a317 = 4 * X[0];
        Scalar const a318 = a317 - 1;
        Scalar const a319 = a11 * X[0];
        Scalar const a320 = 10 * X[0];
        Scalar const a321 = 15 * a4;
        Scalar const a322 = -a320 + a321 + 1;
        Scalar const a323 = a16 * X[1];
        Scalar const a324 = 18 * X[0];
        Scalar const a325 = 2 * a21;
        Scalar const a326 = 14 * X[0];
        Scalar const a327 = 12 * a4;
        Scalar const a328 = 2 * X[0];
        Scalar const a329 = 6 * X[0];
        Scalar const a330 = 24 * X[0];
        Scalar const a331 = -a30 * a326 + a30 * a327 + a31;
        Scalar const a332 = 6 * a17;
        Scalar const a333 = -a332 + a36 * a4;
        Scalar const a334 = a47 * X[0];
        Scalar const a335 = -2 * a334 + a4 * a47;
        Scalar const a336 = 63 * a4;
        Scalar const a337 = a139 + a324 - a336 - 1;
        Scalar const a338 = 252 * a4;
        Scalar const a339 = 90 * X[0];
        Scalar const a340 = 168 * a19;
        Scalar const a341 = a0 * a4;
        Scalar const a342 = 24 * a19;
        Scalar const a343 = a339 * a56;
        Scalar const a344 = a4 * a56;
        Scalar const a345 = -a338 * a78 + a339 * a78 + a340 * a78;
        Scalar const a346 = a56 * X[0];
        Scalar const a347 = a86 * X[0];
        Scalar const a348 = 144 * a4;
        Scalar const a349 = a343 - 153 * a344 + 72 * a59 - a72;
        Scalar const a350 = 39 * a0;
        Scalar const a351 = 18 * a1 + a19 * a64 - 18 * a341;
        Scalar const a352 = 8 * X[0];
        Scalar const a353 = a101 * X[0];
        Scalar const a354 = 9 * a101;
        Scalar const a355 = a101 * a4;
        Scalar const a356 = a104 * X[0];
        Scalar const a357 = 144 * X[0];
        Scalar const a358 = a101 * a342 - a105 + 30 * a353 - 51 * a355;
        Scalar const a359 = 27 * a2;
        Scalar const a360 = 36 * a3;
        Scalar const a361 = a119 * a4;
        Scalar const a362 = a111 * a19 + 18 * a3 - a361;
        Scalar const a363 = 18 * a17;
        Scalar const a364 = 72 * a17;
        Scalar const a365 = 18 * a4;
        Scalar const a366 = -a16 * a365 + 6 * a20 + a363;
        Scalar const a367 = 28 * X[0];
        Scalar const a368 = 168 * a4;
        Scalar const a369 = -336 * a19 - a367 + a368 + 210 * a60 + 1;
        Scalar const a370 = a143 * X[0];
        Scalar const a371 = a143 * a4;
        Scalar const a372 = a143 * a19;
        Scalar const a373 = 2 * a143;
        Scalar const a374 = 240 * a60;
        Scalar const a375 = a146 * X[0];
        Scalar const a376 = a146 * a4;
        Scalar const a377 = a146 * a19;
        Scalar const a378 = a146 * a60;
        Scalar const a379 = -a183;
        Scalar const a380 = a136 * a19;
        Scalar const a381 = 2 * a136;
        Scalar const a382 = std::pow(X[1], 5);
        Scalar const a383 = 56 * a156;
        Scalar const a384 = a156 * X[0];
        Scalar const a385 = 36 * a156;
        Scalar const a386 = a19 * a229;
        Scalar const a387 = a156 * a60;
        Scalar const a388 = a158 * a4;
        Scalar const a389 = 108 * a4;
        Scalar const a390 = 120 * a19;
        Scalar const a391 = 432 * a4;
        Scalar const a392 = a173 * a19;
        Scalar const a393 = a173 * X[0];
        Scalar const a394 = a173 * a4;
        Scalar const a395 = a173 * a374 + a174 - 456 * a392 - 50 * a393 + 264 * a394;
        Scalar const a396 = 90 * a137;
        Scalar const a397 = a137 * X[2];
        Scalar const a398 = a138 * X[2];
        Scalar const a399 = 270 * a180;
        Scalar const a400 = 60 * a137;
        Scalar const a401 = 246 * a138 + 135 * a140 - 324 * a380 - a400;
        Scalar const a402 = a188 * X[0];
        Scalar const a403 = a188 * a4;
        Scalar const a404 = a190 - 26 * a375 + 66 * a376 - 62 * a377 + 20 * a378;
        Scalar const a405 = a213 * X[0];
        Scalar const a406 = 35 * a197;
        Scalar const a407 = a197 * X[0];
        Scalar const a408 = 56 * a200;
        Scalar const a409 = 11 * a197;
        Scalar const a410 = 33 * a197;
        Scalar const a411 = 21 * a200;
        Scalar const a412 = 6 * a4;
        Scalar const a413 = a197 * a60;
        Scalar const a414 = -a197 * a22 - a197 * a317 + a197 * a412 + a413;
        Scalar const a415 = a19 * a197;
        Scalar const a416 = a200 * X[0];
        Scalar const a417 = a200 * a4;
        Scalar const a418 = a197 * a4;
        Scalar const a419 = -20 * a407 + 45 * a413 - 108 * a415 + 82 * a418;
        Scalar const a420 = a156 * a4;
        Scalar const a421 = a224 * X[0];
        Scalar const a422 = a226 * X[0];
        Scalar const a423 = a224 * a4;
        Scalar const a424 = 62 * a19;
        Scalar const a425 = -a156 * a424 + a159 - 26 * a384 + 20 * a387 + 66 * a420;
        Scalar const a426 = a136 * a53;
        Scalar const a427 = -20 * a137;
        Scalar const a428 = a153 * a4 + a234 * a60 - 20 * a380 + a427;
        Scalar const a429 = a250 * X[0];
        Scalar const a430 = a252 * X[0];
        Scalar const a431 = a252 * a4;
        Scalar const a432 = 240 * a19;
        Scalar const a433 = a250 * a4;
        Scalar const a434 = -a250 * a424 + a255 * a60 + a260 - 26 * a429 + 66 * a433;
        Scalar const a435 = 7 * a269;
        Scalar const a436 = 76 * a271;
        Scalar const a437 = a269 * X[0];
        Scalar const a438 = a273 * a4 + a273 - 20 * a437;
        Scalar const a439 = 150 * a288;
        Scalar const a440 = 66 * a271;
        Scalar const a441 = a271 * X[0];
        Scalar const a442 = 6 * a269;
        Scalar const a443 = a19 * a442 + a269 * a324 - a269 * a365 - a442;
        Scalar const a444 = 36 * a4;
        Scalar const a445 = -a22 * a269 - a269 * a317 + a269 * a60 + a4 * a442;
        Scalar const a446 = -a229 * X[0];
        Scalar const a447 = a298 + a446;
        Scalar const a448 = a156 * a365;
        Scalar const a449 = 180 * a421;
        Scalar const a450 = a295 * a60 - a386 + a448;
        Scalar const a451 = a270 + X[0] - 1;
        Scalar const a452 = a451 * X[2];
        Scalar const a453 = a27 * X[0] - a27 + a29 - a328 + a4 + 1;
        Scalar const a454 = a453 * X[2];
        Scalar const a455 = 12 * a30 * X[0];
        Scalar const a456 = 3 * a4;
        Scalar const a457 = X[0] * X[1];
        Scalar const a458 = a57 - 1;
        Scalar const a459 =
            (a118 * X[0] - a118 + a19 + a251 + a321 * X[1] - a456 - 30 * a457 + a458 + 35 * a76) *
            X[2];
        Scalar const a460 = a103 * X[0];
        Scalar const a461 = a3 * X[1];
        Scalar const a462 = 60 * a16;
        Scalar const a463 = 36 * a16;
        Scalar const a464 = a137 * X[1];
        Scalar const a465 = 270 * a464;
        Scalar const a466 = a138 * X[1];
        Scalar const a467 = a138 * a28;
        Scalar const a468 = a148 * X[0];
        Scalar const a469 = a150 * X[0];
        Scalar const a470 = a148 * a4;
        Scalar const a471 = 21 * a199;
        Scalar const a472 = 66 * a199;
        Scalar const a473 = a415 * X[1];
        Scalar const a474 = a199 * X[0];
        Scalar const a475 = a165 * X[0];
        Scalar const a476 = 60 * a19;
        Scalar const a477 = a225 * X[0];
        Scalar const a478 = a165 * a4;
        Scalar const a479 = a137 * a28;
        Scalar const a480 = a254 * X[0];
        Scalar const a481 = 12 * a250;
        Scalar const a482 = a265 * X[0];
        Scalar const a483 = 5 * a269;
        Scalar const a484 = 100 * a275;
        Scalar const a485 = 66 * a275;
        Scalar const a486 = a275 * X[0];
        Scalar const a487 = 105 * a156;
        Scalar const a488 = a302 * a4;
        Scalar const a489 = a28 * a302;
        Scalar const a490 = a305 * X[0];
        P[0]              = a1;
        P[1]              = -a3 + 2 * a5;
        P[2]              = a4 * a6 + (a6 * a9 + a8) * X[0];
        P[3]              = a10 * a4 + (a12 + a15) * X[0];
        P[4]              = a17 - a18 * a4 + 5 * a20;
        P[5]              = a21 * a22 + a4 * (a21 * a23 - 7 * a21) + (-a21 * a9 + 2 * a21) * X[0];
        P[6]  = a24 * a25 + a4 * (-a24 + a26 * X[1]) + (-a24 * a27 + a24 * a29 + a24) * X[0];
        P[7]  = a22 * a30 + a4 * (a30 * a35 + a30 * a9 - 7 * a30) + (-a32 - a33 - a34) * X[0];
        P[8]  = a20 + a4 * (a36 * X[2] + a37 + a38 * X[1]) + (30 * a40 + a42 + a46) * X[0];
        P[9]  = a25 * a47 + a4 * (a48 + a50) + (a48 * a51 + a54 + a55) * X[0];
        P[10] = a4 * a58 - a56 * a57 + a56 * a61 - 63 * a59;
        P[11] = a19 * (a63 - 84) + a4 * (45 - a62) + a61 + (a23 - 6) * X[0];
        P[12] = a19 * (-17 * a0 + 64 * a65) + a4 * (15 * a0 - 108 * a65 + 120 * a67) + a60 * a64 +
                (24 * a0 * X[1] - a66 - 30 * a67) * X[0];
        P[13] = a19 * (a70 + 30 * a71) + a4 * (a72 - a74 + 135 * a75) + (3.0 / 2.0) * a68 +
                (a56 * a77 + a70 + a74 - 270 * a75) * X[0];
        P[14] = a19 * (56 * a83 - a84 + 112 * a85) + a4 * (45 * a78 - 42 * a83 - a84 * X[2]) +
                a61 * a78 + (a80 + a82) * X[0];
        P[15] = a19 * (-51 * a56 + 144 * a71 + 48 * a86) +
                a4 * (-243 * a71 + 180 * a75 - 81 * a86 + a87 + a89 * X[2]) + 18 * a68 +
                (54 * a56 * X[1] + 18 * a56 * X[2] - a72 - a74 * X[2] - a88) * X[0];
        P[16] = a0 * a90 + a19 * (26 * a65 + a91 + 4 * a92) +
                a4 * (9 * a0 + 99 * a67 - a93 - a95 + a96 * X[2]) +
                (a100 - a65 * a97 + a92 * a98 + a99) * X[0];
        P[17] = a102 * a60 + a19 * (-17 * a101 + 16 * a103 + 48 * a104) +
                a4 * (15 * a101 - 27 * a103 + a104 * a110 - 81 * a104 + 72 * a107 + 12 * a109) +
                (6 * a101 * X[1] + 18 * a101 * X[2] - a104 * a23 - a105 - a106 - a108) * X[0];
        P[18] = a19 * (a112 + a113 * X[2] + a2 * a23) + a2 * a90 +
                a4 * (-a115 - a117 + a118 * a2 + a119 * a53 + a120 * X[2] + 9 * a2) +
                (252 * a114 * a53 - 288 * a114 * X[2] + a121 * a98 + a122 + a125) * X[0];
        P[19] = a16 * a90 + a19 * (a127 + a41 * X[1]) +
                a4 * (a128 * X[1] + a130 + a28 * a38 + a44) +
                (a131 * a45 + a132 * X[1] + a134 + a135 - 144 * a40) * X[0];
        P[20] = a136 * a139 + a137 - 14 * a138 - 84 * a140 + 42 * a142;
        P[21] = a143 * a144 + a19 * (88 * a143 - 216 * a145) + a4 * (-25 * a143 + 72 * a145) +
                a60 * (-114 * a143 + 180 * a145) + (-a143 * a9 + 2 * a143) * X[0];
        P[22] = a146 * a147 + a19 * (-a146 * a63 + (82.0 / 3.0) * a146 + 150 * a150) +
                a4 * (-a123 * a146 + 76 * a146 * X[1] - a149) + a60 * (-27 * a146 + 90 * a148) +
                (-a146 * a27 + a146 + a149 * a28) * X[0];
        P[23] = 4 * a142 + a19 * (22 * a136 - 210 * a151 + 300 * a152) +
                a4 * (-13 * a136 + 180 * a151 - 495 * a152 + 350 * a155) +
                a60 * (-31.0 / 2.0 * a136 + 75 * a151) +
                (90 * a136 * a28 + 2 * a136 - a154 - 70 * a155) * X[0];
        P[24] = a156 * a157 + a19 * (a159 + a161 + a163) + a4 * (a164 + a166 - a168 + 112 * a169) +
                a60 * (-a156 + a158 * X[1]) +
                (-a156 * a170 + a156 * a172 + a156 + a161 + a168) * X[0];
        P[25] = a144 * a173 + a19 * (88 * a173 - 72 * a178 - 144 * a179) +
                a4 * (a160 * a173 - 25 * a173 + 48 * a179) +
                a60 * (-114 * a173 + 60 * a178 + 120 * a179) + (-a175 - a176 - a177) * X[0];
        P[26] = 27 * a142 + a19 * (82 * a136 - 378 * a151 + 225 * a152 - 126 * a180 + 450 * a185) +
                a4 * (171 * a136 * X[1] + 57 * a136 * X[2] - 135 * a152 - a153 - 270 * a185) +
                a60 * (-81 * a136 + (405.0 / 2.0) * a151 + (135.0 / 2.0) * a180) +
                (a154 * X[2] + a182 + a184) * X[0];
        P[27] =
            a141 * a186 + a19 * (-182 * a148 + 220 * a150 + 160 * a187 + a189) +
            a4 * (a146 * a77 + 156 * a148 - 363 * a150 - 264 * a187 + 420 * a195 + a196) +
            a60 * (-31.0 / 2.0 * a146 + 65 * a148 + a149 * X[2]) +
            (66 * a146 * a28 + 48 * a146 * X[1] * X[2] - a191 - a192 - a193 - a194 - 84 * a195) *
                X[0];
        P[28] = a157 * a197 + a19 * (a201 + 35 * a202 + 14 * a203 + a207) +
                a4 * (33 * a199 - a201 + a209 - a210 + a212 - a215 + a216 * X[2]) +
                a60 * (a198 + (11.0 / 2.0) * a199 + (1.0 / 2.0) * a200) +
                (-168 * a202 * X[2] + a207 + a215 + a217 * X[2] + a218) * X[0];
        P[29] =
            a147 * a197 +
            a19 * ((82.0 / 3.0) * a197 - 126 * a200 + 15 * a202 + 90 * a203 - a214 + 90 * a219) +
            a4 * (19 * a197 * X[1] + 57 * a197 * X[2] - 10 * a197 - 9 * a202 - 54 * a203 -
                  54 * a219) +
            a60 * (-27 * a197 + (45.0 / 2.0) * a199 + (135.0 / 2.0) * a200) +
            (a200 * a9 + a220 + a221) * X[0];
        P[30] = a141 * a222 +
                a19 * (-a156 * a62 + 22 * a156 - 84 * a224 + 100 * a225 + 40 * a226 + 320 * a227) +
                a4 * (-13 * a156 + 108 * a165 + 70 * a169 + 72 * a224 - 165 * a225 - 66 * a226 -
                      528 * a227 + 420 * a232 + 420 * a233) +
                a60 * (-31.0 / 2.0 * a156 + 45 * a165 + a223 * X[2]) +
                (30 * a156 * a28 + 12 * a156 * a53 + 96 * a156 * X[1] * X[2] - a164 - a228 - a230 -
                 a231 - 84 * a232 - 84 * a233) *
                    X[0];
        P[31] = a142 + a19 * (170 * a185 + a235 * a53 + a235 + a236 * a28 + a241) +
                a4 * (135 * a151 + 220 * a155 - a235 + a242 * X[2] - a243 - a244 - a245 +
                      240 * a246 + a247 * X[2]) +
                a60 * ((45.0 / 2.0) * a151 + (15.0 / 2.0) * a180 - a234) +
                (1080 * a152 * a53 - 1560 * a152 * X[2] + 1080 * a155 * X[2] + a241 + a245 -
                 480 * a246 + a248 + a249) *
                    X[0];
        P[32] = 4 * a141 * a250 +
                a19 * (-168 * a252 + a253 - 42 * a254 + a255 * a28 + 200 * a256 + 160 * a257) +
                a4 * (-13 * a250 + 144 * a252 + 36 * a254 - 330 * a256 - 264 * a257 + 200 * a262 -
                      33 * a265 + 120 * a266 + a267 * a76 + a268 * X[1]) +
                a60 * (a250 * a251 - 31.0 / 2.0 * a250 + 60 * a252) +
                (6 * a250 * a28 + 60 * a250 * a53 + 48 * a250 * X[1] * X[2] + 2 * a250 - a258 -
                 a259 - a261 - a263 - a264 * X[1] - 24 * a266) *
                    X[0];
        P[33] = a157 * a269 + a19 * (a272 + a273 * a28 + a273 * a53 + 44 * a274 + a280) +
                a4 * (a133 * a273 + a23 * a269 + a271 * a287 + 18 * a271 - a272 - a282 + 14 * a283 -
                      a284 - a285 + 150 * a286) +
                a60 * (a269 * a270 - a269 + 3 * a271) +
                (180 * a133 * a275 - 228 * a271 * a28 + a280 + 108 * a283 * X[2] + a285 -
                 300 * a286 + a288 * a289 + a292 + a293) *
                    X[0];
        P[34] = a141 * a294 + a19 * (a158 * a28 + 60 * a227 + a301 + a303) +
                a4 * (a158 * a76 + a228 + 270 * a232 + a28 * a307 - a304 - a306 + a311) +
                a60 * (a295 * X[1] + a296) +
                (420 * a133 * a165 + 60 * a169 * X[2] + 270 * a225 * a53 - 540 * a232 - 180 * a233 +
                 a301 + a306 + a314 + a316) *
                    X[0];
        P[35] = a65;
        P[36] = a114 * a318;
        P[37] = 3 * a28 * a6 + (a7 * X[0] + a8) * X[1];
        P[38] = a10 * a28 + (a15 + a319) * X[1];
        P[39] = a322 * a323;
        P[40] = a28 * (a21 * a324 - 3 * a21) + (-a21 * a326 + a21 * a327 + a325) * X[1];
        P[41] = (10.0 / 3.0) * a24 * a76 + a28 * (a24 * a317 - a26) +
                (-a24 * a328 + a24 * a4 + a24) * X[1];
        P[42] = a28 * (a30 * a329 - a30) + (a30 * a330 * X[2] + a331 - a34) * X[1];
        P[43] = a18 * a76 + a28 * (9 * a17 - a38 + 15 * a39) + (a332 * X[2] + a333 + a42) * X[1];
        P[44] = a28 * (a334 + a50) + (1.0 / 3.0) * a47 * a76 + (a334 * a51 + a335 + a54) * X[1];
        P[45] = a270 * a337 * a56;
        P[46] = a28 * (a338 - 126 * X[0] + 9) + (-a338 + a339 + a340 - 6) * X[1];
        P[47] = a28 * (-108 * a1 + 96 * a341 + a94) + a76 * (-10 * a0 + 80 * a1) +
                (a0 * a342 + 30 * a1 - 51 * a341 - a66) * X[1];
        P[48] = (105.0 / 2.0) * a171 * a56 + a28 * (-a343 + a4 * a87 + a87) + a76 * (a343 - a73) +
                (a19 * a69 + a324 * a56 - 18 * a344 + a70) * X[1];
        P[49] = a28 * (a4 * a84 - 42 * a78 * X[0] + 3 * a78) +
                (a345 + 336 * a4 * a85 + a82 - 168 * a85 * X[0]) * X[1];
        P[50] = a28 * (216 * a344 - 243 * a346 + 360 * a347 + a58 - a87 * X[2]) +
                a76 * (120 * a346 - 15 * a56) + (-162 * a347 + a348 * a86 + a349 + 18 * a86) * X[1];
        P[51] = (63.0 / 2.0) * a0 * a171 +
                a28 * (a1 * a131 - 78 * a1 + a350 * a4 + a350 - 72 * a92) +
                a76 * (-66 * a0 + 66 * a1 + 84 * a92) +
                (-24 * a1 * X[2] + a351 + a4 * a95 + a99) * X[1];
        P[52] = a28 * (a105 - 27 * a353 - a354 * X[2] + 24 * a355 + 72 * a356) +
                a76 * (a101 * a352 - a101) +
                (a104 * a348 + 18 * a104 + a107 * a357 - a108 - 162 * a356 + a358) * X[1];
        P[53] = (21.0 / 2.0) * a171 * a2 +
                a28 * (-144 * a121 + 126 * a2 * a53 + a3 * a97 - 54 * a3 + a359 * a4 + a359) +
                a76 * (84 * a121 - 30 * a2 + 30 * a3) +
                (a122 - a131 * a3 + a360 * a53 + a362 + 36 * a5 * X[2]) * X[1];
        P[54] = (3.0 / 2.0) * a16 * a171 + a28 * (a130 - a363 + a364 * X[2] + a38 * a4) +
                a76 * (a127 + a332) +
                (a128 * a4 + a134 + 180 * a17 * a53 - a17 * a97 + a366) * X[1];
        P[55] = a151 * a369;
        P[56] = a28 * (-3 * a143 + 72 * a370 - 324 * a371 + 360 * a372) +
                (a143 * a374 - 50 * a370 + 264 * a371 - 456 * a372 + a373) * X[1];
        P[57] = a28 * (-a146 * a338 - a186 + 76 * a375 + 180 * a377) +
                a76 * ((10.0 / 3.0) * a146 - 60 * a375 + 150 * a376) +
                (a146 - 20 * a375 + 82 * a376 - 108 * a377 + 45 * a378) * X[1];
        P[58] = a171 * (-35.0 / 2.0 * a136 + 175 * a137) +
                a28 * (180 * a137 - 315 * a138 + a379 + 150 * a380) +
                a76 * (-330 * a137 + 300 * a138 + a153) +
                (-26 * a137 + 66 * a138 + 20 * a140 - 62 * a380 + a381) * X[1];
        P[59] = (126.0 / 5.0) * a156 * a382 + a171 * (a383 * X[0] - a383) +
                a28 * (-a229 - a385 * a4 + a385 * X[0] + a386) +
                a76 * (a162 * a4 + a162 - 84 * a384) +
                (-a156 * a22 - a156 * a317 + a156 + a387 + a388) * X[1];
        P[60] = a28 * (a173 * a330 - a173 * a389 + a173 * a390 - a173) +
                (-a177 + 480 * a179 * a19 - a179 * a391 + 96 * a179 * X[0] + a395) * X[1];
        P[61] = a28 * (-9 * a136 + 171 * a137 - 567 * a138 + a183 * X[2] + 405 * a380 - 270 * a397 +
                       675 * a398) +
                a76 * (225 * a138 + a234 - a396) +
                (a182 + a19 * a399 + 114 * a397 - 378 * a398 + a401) * X[1];
        P[62] = a171 * (-21.0 / 2.0 * a146 + 105 * a375) +
                a28 * (a196 + 156 * a375 - 273 * a376 + 130 * a377 - 264 * a402 + 240 * a403) +
                a76 * (a189 - 242 * a375 + 220 * a376 + 280 * a402) +
                (40 * a188 * a19 - a193 + 48 * a402 - 84 * a403 + a404) * X[1];
        P[63] =
            a171 * (a213 * X[2] - a213 + a405) + (84.0 / 5.0) * a197 * a382 +
            a28 * (a19 * a409 - a4 * a410 + a4 * a411 - a405 * X[2] - a409 + a410 * X[0] + a411) +
            a76 * (a4 * a406 + a406 - 70 * a407 + a408 * X[0] - a408) +
            (a19 * a205 + a197 + a200 * a329 - a200 * a412 + a206 + a414) * X[1];
        P[64] =
            a28 * (-a197 * a336 + a198 + a209 + 19 * a407 + 45 * a415 - 54 * a416 + 135 * a417) +
            a76 * (a197 * a321 - a197 * a329 + (1.0 / 3.0) * a197) +
            (270 * a219 * a4 - 108 * a219 * X[0] + a220 + 270 * a415 * X[2] + 114 * a416 -
             378 * a417 + a419) *
                X[1];
        P[65] = a171 * (-7.0 / 2.0 * a156 + 35 * a384) +
                a28 * (-9 * a156 - a162 * a53 + a19 * a302 + 48 * a224 + 108 * a384 - 189 * a420 -
                       528 * a421 + 420 * a422 + 480 * a423) +
                a76 * (10 * a156 - 28 * a224 - 110 * a384 + 100 * a420 + 280 * a421) +
                (-a224 * a338 + a224 * a357 + a224 * a390 + 120 * a226 * a4 + a229 * a53 - a230 -
                 132 * a422 + a425) *
                    X[1];
        P[66] =
            36 * a136 * a382 + a171 * (-110 * a136 + 110 * a137 + a399) +
            a28 * (240 * a137 * a53 + 135 * a137 - 135 * a138 + 255 * a180 + a19 * a242 - a242 -
                   510 * a397 + 255 * a398 - 240 * a426) +
            a76 * (-230 * a137 - 520 * a180 + a236 * a4 + a236 + 520 * a397 + 360 * a426) +
            (a19 * a239 + a240 + a244 * a4 + a248 + a396 * X[2] - 90 * a398 - a400 * a53 + a428) *
                X[1];
        P[67] = a171 * (-1.0 / 2.0 * a250 + 5 * a429) +
                a28 * (30 * a19 * a250 - a250 * a336 - 3 * a250 - 30 * a256 + a259 + a268 * X[0] +
                       36 * a429 - 264 * a430 + 240 * a431) +
                a76 * (-8 * a252 - a253 * X[0] + a255 * a4 + a260 + 80 * a430) +
                (a252 * a432 + 600 * a256 * a4 - 660 * a256 * X[0] - a259 + 400 * a262 * X[0] -
                 a263 + a264 + 288 * a430 - 504 * a431 + a434) *
                    X[1];
        P[68] = a171 * (27 * a271 + a435 * X[0] - a435) + (9.0 / 5.0) * a269 * a382 +
                a28 * (90 * a290 + a4 * a440 + a439 * X[0] - a439 + a440 - 132 * a441 + a443) +
                a76 * (90 * a288 + a436 * X[0] - a436 + a438) +
                (a19 * a278 - a271 * a444 + a279 + a284 * a4 - 60 * a288 * X[0] + a291 * X[0] +
                 a292 + 36 * a441 + a445) *
                    X[1];
        P[69] = a171 * (a295 * X[0] + a296) +
                a28 * (a156 * a324 + a158 * a19 + a307 * a4 + a309 * X[0] + a311 - a448 - a449) +
                a294 * a382 + a76 * (a297 * X[0] + a303 + a388 + a447) +
                (a19 * a297 + a309 * a4 + a312 * X[0] + a314 - 540 * a422 - 180 * a423 + a447 +
                 a449 + a450) *
                    X[1];
        P[70] = a92;
        P[71] = a121 * a318;
        P[72] = a452 * a7;
        P[73] = a11 * a53 + (a12 + a13 + a319) * X[2];
        P[74] = a322 * a39;
        P[75] = a325 * a452 * (a329 - 1);
        P[76] = a24 * a454;
        P[77] = a53 * (a32 + a455) + (-a33 + a331 + a455 * X[1]) * X[2];
        P[78] = a53 * (a16 * a251 + 3 * a17 + a37) + (a17 * a23 + a333 + a36 + a46) * X[2];
        P[79] = 2 * a133 * a47 + a53 * (3 * a334 + 3 * a48 - a49) +
                (a328 * a48 + a335 + a47 + a55) * X[2];
        P[80] = 3 * a337 * a86;
        P[81] = a451 * a51 * (-a326 + 28 * a4 + 1);
        P[82] = a454 * a66 * (a352 - 1);
        P[83] = a459 * a69;
        P[84] = a53 * (a368 * a78 + a79 - a84 * X[0]) +
                (a345 + a4 * a63 * a78 - a457 * a84 + a80 + a81) * X[2];
        P[85] = a53 * (72 * a344 - 81 * a346 + a72 - a87 * X[1] + a89 * X[0]) +
                (a349 + a391 * a71 - 486 * a71 * X[0] + 54 * a71 + 360 * a75 * X[0] - a88) * X[2];
        P[86] = a53 * (a0 * a167 + a1 * a110 - 12 * a1 + a4 * a64 + a64 - a96) +
                (198 * a1 * a28 - 156 * a1 * X[1] + a100 + a351 + a4 * a93 + a91) * X[2];
        P[87] = a133 * (-a102 + 48 * a353) +
                a53 * (-81 * a353 - a354 * X[1] + a354 + 72 * a355 + 72 * a460) +
                (a102 * X[1] + 48 * a103 * a4 - a106 + a109 * a330 + a358 - 54 * a460) * X[2];
        P[88] = a133 * (-a113 + 84 * a114 + 12 * a3) +
                a53 * (a119 - a120 + a167 * a2 - a360 + a361 + 144 * a461) +
                (a112 + a123 * a3 + a125 + a362 - 108 * a461 + 54 * a5 * X[1]) * X[2];
        P[89] = a133 * (60 * a17 + a462 * X[1] - a462) + 30 * a16 * a313 +
                a53 * (a28 * a463 - 72 * a323 + a364 * X[1] - a364 + a4 * a463 + a463) +
                (a126 + a135 - 36 * a17 * X[1] + a28 * a363 + a366 + a4 * a43) * X[2];
        P[90] = a180 * a369;
        P[91] = a373 * a452 * (a330 - a389 + a390 - 1);
        P[92] = a188 * a453 * a458 * (15 * X[0] - 1);
        P[93] = a381 * a459 * (a320 - 1);
        P[94] = a224 * (-a110 * a4 + a160 * a19 - a160 + a167 * a4 + a167 + a170 * X[0] - a170 +
                        a172 - a22 - a317 + a412 + 72 * a457 + a60 - a98 * X[0] + 1);
        P[95] = a53 * (a175 + 240 * a392 + 48 * a393 - 216 * a394) +
                (-a176 - 216 * a178 * a4 + a178 * a432 + 48 * a178 * X[0] + a395) * X[2];
        P[96] =
            a53 * (57 * a137 - 189 * a138 - a181 + a183 * X[1] + 135 * a380 - a465 + 675 * a466) +
            (-a137 * a289 + 810 * a151 * a19 + a181 + a184 + a401 + 342 * a464 - 1134 * a466 +
             675 * a467) *
                X[2];
        P[97] = a53 * (a146 * a160 + a146 * a330 - 42 * a150 + a191 - 42 * a376 + 20 * a377 -
                       264 * a468 + 420 * a469 + 240 * a470) +
                (260 * a148 * a19 + 660 * a150 * a4 + 66 * a150 - a192 - a194 + 420 * a375 * a76 +
                 a404 + 312 * a468 - 726 * a469 - 546 * a470) *
                    X[2];
        P[98] = a53 * (-a197 * a456 + a197 * a57 + a198 + a212 - a214 * X[0] + a216 * X[0] - a216 +
                       a4 * a471 + a415 + a471) +
                (-210 * a202 * X[0] + a204 + a210 * a4 + a217 * X[0] + a218 - a4 * a472 + a414 +
                 a472 * X[0] + 22 * a473) *
                    X[2];
        P[99] = a133 * (a201 - 36 * a407 + 90 * a418) +
                a53 * (a197 * a270 + 135 * a199 * a4 - a208 + 57 * a407 + 135 * a415 - 189 * a418 -
                       54 * a474) +
                (a197 - a202 * a324 + 45 * a202 * a4 + a221 - a418 * a62 + a419 + 90 * a473 +
                 38 * a474) *
                    X[2];
        P[100] = a133 * (-28 * a165 + a222 - 44 * a384 + 40 * a420 + 280 * a475) +
                 a53 * (a156 * a476 - a163 + 48 * a165 + a308 + 72 * a384 - 126 * a420 -
                        528 * a475 + 420 * a477 + 480 * a478) +
                 (140 * a169 * X[0] + a19 * a305 + a223 * a28 + 300 * a225 * a4 - a228 - a231 +
                  a425 + 216 * a475 - 330 * a477 - 378 * a478) *
                     X[2];
        P[101] = a133 * (-160 * a151 + 360 * a152 + a235 * a4 + a235 + a427 + 160 * a464) +
                 a53 * (45 * a137 + 255 * a151 + 540 * a155 + a183 * a19 - a242 * a4 - a247 + a379 -
                        510 * a464 + 255 * a466 + 780 * a479) +
                 (440 * a137 * a76 + a19 * a237 + a234 + a238 + a249 + a428 + a465 - 270 * a466 +
                  345 * a467 - 690 * a479) *
                     X[2];
        P[102] = a133 * (-a255 * X[1] + a255 - 220 * a429 + 200 * a433 + 200 * a480) +
                 a313 * (-a267 + 100 * a429) +
                 a53 * (a160 * a250 - a250 * a338 + a250 * a390 + 240 * a254 * a4 - a28 * a481 +
                        144 * a429 - 264 * a480 - a481 + 120 * a482) +
                 (a254 * a476 + a255 * a76 * X[0] - a258 - a261 + 60 * a265 * a4 + 6 * a265 -
                  a433 * a62 + a434 + 72 * a480 - 66 * a482) *
                     X[2];
        P[103] =
            a133 * (a123 * a269 + a438 + a484 * X[0] - a484) +
            a313 * (45 * a275 + a483 * X[0] - a483) +
            a53 * (-a269 * a287 + 54 * a283 + a287 * a437 + a4 * a485 + a443 + a485 - 132 * a486) +
            (a19 * a276 + a269 - a275 * a444 + a277 - 60 * a28 * a437 + a282 * a4 + a283 * a367 +
             a293 + a445 + 36 * a486) *
                X[2];
        P[104] = a133 * (a302 - a305 - 180 * a384 + a488 + a489 + a490) + a162 * std::pow(X[2], 5) +
                 a313 * (a487 * X[0] + a487 * X[1] - a487) +
                 a53 * (a156 * a339 + a19 * a223 + a223 * a76 - a223 + a225 * a339 + a302 * X[1] +
                        a488 * X[1] - a488 - a489 - a490) +
                 (-a166 * a4 + a166 * X[0] + a19 * a299 + a295 + a300 + a304 * a4 + a315 * X[0] +
                  a316 + a446 + a450 - 36 * a477) *
                     X[2];
        return Pm;
    }
};

/**
 * Divergence free polynomial basis on reference line
 */

template <>
class DivergenceFreeBasis<1, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 1;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = 1;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<1, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 1;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = 1;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<1, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 1;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = 1;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<1, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 1;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 1;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P = Pm.data();
        P[0]      = 1;
        return Pm;
    }
};

/**
 * Divergence free polynomial basis on reference triangle
 */

template <>
class DivergenceFreeBasis<2, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 5;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = std::numbers::sqrt3_v<Scalar>;
        P[0]            = a0;
        P[1]            = 0;
        P[2]            = 0;
        P[3]            = a0;
        P[4]            = 0;
        P[5]            = 6 * X[0] - 2;
        P[6]            = a1 * (4 * X[1] - 4.0 / 3.0);
        P[7]            = 0;
        P[8]            = a1 * (4.0 / 3.0 - 4 * X[0]);
        P[9]            = 2 * a1 * (X[0] + 2 * X[1] - 1);
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<2, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 9;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1 = 6 * X[0];
        Scalar const a2 = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3 = X[0] + 2 * X[1] - 1;
        Scalar const a4 = ((X[0]) * (X[0]));
        Scalar const a5 = 30 * a4;
        Scalar const a6 = std::sqrt(30);
        Scalar const a7 = ((X[1]) * (X[1]));
        P[0]            = a0;
        P[1]            = 0;
        P[2]            = 0;
        P[3]            = a0;
        P[4]            = 0;
        P[5]            = a1 - 2;
        P[6]            = a2 * (4 * X[1] - 4.0 / 3.0);
        P[7]            = 0;
        P[8]            = a2 * (4.0 / 3.0 - 4 * X[0]);
        P[9]            = 2 * a2 * a3;
        P[10]           = 0;
        P[11]           = std::sqrt(6) * (10 * a4 - 8 * X[0] + 1);
        P[12]           = (1.0 / 2.0) * a0 * (-a5 + 12 * X[0] + 1);
        P[13]           = 3 * a0 * a3 * (5 * X[0] - 1);
        P[14]           = (3.0 / 5.0) * a6 * (10 * a7 - 8 * X[1] + 1);
        P[15]           = 0;
        P[16] = (1.0 / 10.0) * a6 * (-a5 - 120 * X[0] * X[1] + 60 * X[0] + 24 * X[1] - 13);
        P[17] = a6 * (a1 * X[1] + a4 + 6 * a7 - 2 * X[0] - 6 * X[1] + 1);
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<2, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 14;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = 6 * X[0];
        Scalar const a2  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3  = 4 * X[0];
        Scalar const a4  = X[0] + 2 * X[1] - 1;
        Scalar const a5  = 2 * a4;
        Scalar const a6  = std::sqrt(6);
        Scalar const a7  = ((X[0]) * (X[0]));
        Scalar const a8  = 12 * X[0];
        Scalar const a9  = 30 * a7;
        Scalar const a10 = std::sqrt(30);
        Scalar const a11 = ((X[1]) * (X[1]));
        Scalar const a12 = 10 * a11 + 1;
        Scalar const a13 = -6 * X[1];
        Scalar const a14 = a7 - 2 * X[0];
        Scalar const a15 = a1 * X[1] + 6 * a11 + a13 + a14 + 1;
        Scalar const a16 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a17 = std::sqrt(10);
        Scalar const a18 = 210 * a16;
        Scalar const a19 = std::sqrt(14);
        Scalar const a20 = 10 * X[1];
        P[0]             = a0;
        P[1]             = 0;
        P[2]             = 0;
        P[3]             = a0;
        P[4]             = 0;
        P[5]             = a1 - 2;
        P[6]             = a2 * (4 * X[1] - 4.0 / 3.0);
        P[7]             = 0;
        P[8]             = a2 * (4.0 / 3.0 - a3);
        P[9]             = a2 * a5;
        P[10]            = 0;
        P[11]            = a6 * (10 * a7 - 8 * X[0] + 1);
        P[12]            = (1.0 / 2.0) * a0 * (a8 - a9 + 1);
        P[13]            = 3 * a0 * a4 * (5 * X[0] - 1);
        P[14]            = (3.0 / 5.0) * a10 * (a12 - 8 * X[1]);
        P[15]            = 0;
        P[16] = (1.0 / 10.0) * a10 * (-a9 - 120 * X[0] * X[1] + 60 * X[0] + 24 * X[1] - 13);
        P[17] = a10 * a15;
        P[18] = 0;
        P[19] = a0 * (70 * a16 - 90 * a7 + 30 * X[0] - 2);
        P[20] = a6 * (-28 * a16 - a3 + 24 * a7 + 2.0 / 15.0);
        P[21] = a5 * a6 * (21 * a7 - a8 + 1);
        P[22] = (2.0 / 15.0) * a17 *
                (-a13 - a18 - 630 * a7 * X[1] + 360 * a7 + 180 * X[0] * X[1] - 90 * X[0] - 5);
        P[23] = 2 * a15 * a17 * (7 * X[0] - 1);
        P[24] = (8.0 / 7.0) * a19 * (-45 * a11 + 35 * ((X[1]) * (X[1]) * (X[1])) + 15 * X[1] - 1);
        P[25] = 0;
        P[26] = (4.0 / 105.0) * a19 *
                (-3150 * a11 * X[0] + 450 * a11 - a18 - 1575 * a7 * X[1] + 630 * a7 +
                 3150 * X[0] * X[1] - 630 * X[0] - 465 * X[1] + 101);
        P[27] = a19 * a5 * (a12 + a14 + a20 * X[0] - a20);
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<2, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 2;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 20;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a1  = 6 * X[0];
        Scalar const a2  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a3  = 4 * X[0];
        Scalar const a4  = X[0] + 2 * X[1] - 1;
        Scalar const a5  = 2 * a4;
        Scalar const a6  = std::sqrt(6);
        Scalar const a7  = ((X[0]) * (X[0]));
        Scalar const a8  = 12 * X[0];
        Scalar const a9  = 30 * a7;
        Scalar const a10 = std::sqrt(30);
        Scalar const a11 = ((X[1]) * (X[1]));
        Scalar const a12 = 10 * a11 + 1;
        Scalar const a13 = -24 * X[1];
        Scalar const a14 = X[0] * X[1];
        Scalar const a15 = -6 * X[1];
        Scalar const a16 = a7 - 2 * X[0];
        Scalar const a17 = a1 * X[1] + 6 * a11 + a15 + a16 + 1;
        Scalar const a18 = 90 * a7;
        Scalar const a19 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a20 = std::sqrt(10);
        Scalar const a21 = 210 * a19;
        Scalar const a22 = 630 * a7;
        Scalar const a23 = std::sqrt(14);
        Scalar const a24 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a25 = -a22;
        Scalar const a26 = -450 * a11;
        Scalar const a27 = a11 * X[0];
        Scalar const a28 = a7 * X[1];
        Scalar const a29 = 10 * X[1];
        Scalar const a30 = a12 + a16 + a29 * X[0] - a29;
        Scalar const a31 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a32 = 60 * X[0];
        Scalar const a33 = 1260 * a31;
        Scalar const a34 = 11340 * a31;
        Scalar const a35 = a19 * X[1];
        Scalar const a36 = std::sqrt(70);
        Scalar const a37 = -5040 * X[0];
        Scalar const a38 = a11 * a7;
        Scalar const a39 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a40 = 20 * X[1];
        Scalar const a41 = 140 * a24;
        P[0]             = a0;
        P[1]             = 0;
        P[2]             = 0;
        P[3]             = a0;
        P[4]             = 0;
        P[5]             = a1 - 2;
        P[6]             = a2 * (4 * X[1] - 4.0 / 3.0);
        P[7]             = 0;
        P[8]             = a2 * (4.0 / 3.0 - a3);
        P[9]             = a2 * a5;
        P[10]            = 0;
        P[11]            = a6 * (10 * a7 - 8 * X[0] + 1);
        P[12]            = (1.0 / 2.0) * a0 * (a8 - a9 + 1);
        P[13]            = 3 * a0 * a4 * (5 * X[0] - 1);
        P[14]            = (3.0 / 5.0) * a10 * (a12 - 8 * X[1]);
        P[15]            = 0;
        P[16]            = (1.0 / 10.0) * a10 * (-a13 - 120 * a14 - a9 + 60 * X[0] - 13);
        P[17]            = a10 * a17;
        P[18]            = 0;
        P[19]            = a0 * (-a18 + 70 * a19 + 30 * X[0] - 2);
        P[20]            = a6 * (-28 * a19 - a3 + 24 * a7 + 2.0 / 15.0);
        P[21]            = a5 * a6 * (21 * a7 - a8 + 1);
        P[22]            = (2.0 / 15.0) * a20 *
                (-a15 - a21 - a22 * X[1] + 360 * a7 + 180 * X[0] * X[1] - 90 * X[0] - 5);
        P[23] = 2 * a17 * a20 * (7 * X[0] - 1);
        P[24] = (8.0 / 7.0) * a23 * (-45 * a11 + 35 * a24 + 15 * X[1] - 1);
        P[25] = 0;
        P[26] = (4.0 / 105.0) * a23 *
                (-a21 - a25 - a26 - 3150 * a27 - 1575 * a28 + 3150 * X[0] * X[1] - 630 * X[0] -
                 465 * X[1] + 101);
        P[27] = a23 * a30 * a5;
        P[28] = 0;
        P[29] = a20 * (-224 * a19 + 126 * a31 + 126 * a7 - 24 * X[0] + 1);
        P[30] = (1.0 / 30.0) * a10 * (1680 * a19 + a25 + a32 - a33 + 1);
        P[31] = a10 * a4 * (84 * a19 - 84 * a7 + 21 * X[0] - 1);
        P[32] = (1.0 / 42.0) * a0 *
                (-a13 - 2520 * a14 + 21840 * a19 - a34 - 30240 * a35 + 20160 * a7 * X[1] -
                 10710 * a7 + 1260 * X[0] - 29);
        P[33] = 5 * a0 * a17 * (36 * a7 - 16 * X[0] + 1);
        P[34] = (1.0 / 420.0) * a36 *
                (25200 * a11 * X[0] - 25200 * a14 + 31920 * a19 - a26 - a34 - 75600 * a35 - a37 -
                 113400 * a38 + 126000 * a7 * X[1] - 27720 * a7 - 600 * X[1] + 209);
        P[35] = a30 * a36 * a4 * (9 * X[0] - 1);
        P[36] = (5.0 / 3.0) * a20 * (126 * a11 + a13 - 224 * a24 + 126 * a39 + 1);
        P[37] = 0;
        P[38] = (1.0 / 84.0) * a20 *
                (105840 * a11 * X[0] - 11970 * a11 - 45360 * a14 + 5040 * a19 - 70560 * a24 * X[0] +
                 7840 * a24 - a33 - 15120 * a35 - a37 - 52920 * a38 + 45360 * a7 * X[1] -
                 7560 * a7 + 5304 * X[1] - 641);
        P[39] = 3 * a20 *
                (a11 * a18 + 90 * a11 + a19 * a40 - 4 * a19 - 180 * a27 - 60 * a28 - a3 + a31 +
                 a32 * X[1] + 70 * a39 - a40 + a41 * X[0] - a41 + 6 * a7 + 1);
        return Pm;
    }
};

/**
 * Divergence free polynomial basis on reference tetrahedron
 */

template <>
class DivergenceFreeBasis<3, 1>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 1;
    inline static constexpr std::size_t kSize  = 11;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P       = Pm.data();
        Scalar const a0 = std::sqrt(6);
        Scalar const a1 = 4 * X[0] - 1;
        Scalar const a2 = std::sqrt(10) * a1;
        Scalar const a3 = std::sqrt(5);
        Scalar const a4 = X[0] - 1;
        Scalar const a5 = 2 * a3 * (a4 + 3 * X[1]);
        Scalar const a6 = std::sqrt(15);
        Scalar const a7 = -a1 * a6;
        Scalar const a8 = 2 * a6 * (a4 + X[1] + 2 * X[2]);
        P[0]            = a0;
        P[1]            = 0;
        P[2]            = 0;
        P[3]            = 0;
        P[4]            = a0;
        P[5]            = 0;
        P[6]            = 0;
        P[7]            = 0;
        P[8]            = a0;
        P[9]            = 0;
        P[10]           = a2;
        P[11]           = 0;
        P[12]           = 0;
        P[13]           = 0;
        P[14]           = a2;
        P[15]           = a3 * (6 * X[1] - 3.0 / 2.0);
        P[16]           = 0;
        P[17]           = 0;
        P[18]           = a3 * (3.0 / 2.0 - 6 * X[0]);
        P[19]           = a5;
        P[20]           = 0;
        P[21]           = 0;
        P[22]           = 0;
        P[23]           = a5;
        P[24]           = (1.0 / 2.0) * a6 * (4 * X[1] + 8 * X[2] - 3);
        P[25]           = 0;
        P[26]           = 0;
        P[27]           = (1.0 / 2.0) * a7;
        P[28]           = a8;
        P[29]           = 0;
        P[30]           = a7;
        P[31]           = 0;
        P[32]           = a8;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<3, 2>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 2;
    inline static constexpr std::size_t kSize  = 26;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = 4 * X[0] - 1;
        Scalar const a2  = std::sqrt(10) * a1;
        Scalar const a3  = std::sqrt(5);
        Scalar const a4  = 6 * X[1];
        Scalar const a5  = 6 * X[0];
        Scalar const a6  = X[0] - 1;
        Scalar const a7  = 2 * a6 + 6 * X[1];
        Scalar const a8  = a3 * a7;
        Scalar const a9  = std::sqrt(15);
        Scalar const a10 = -a1 * a9;
        Scalar const a11 = a6 + X[1] + 2 * X[2];
        Scalar const a12 = 2 * a11 * a9;
        Scalar const a13 = std::sqrt(14);
        Scalar const a14 = ((X[0]) * (X[0]));
        Scalar const a15 = a13 * (15 * a14 - 10 * X[0] + 1);
        Scalar const a16 = std::sqrt(7);
        Scalar const a17 = a5 - 1;
        Scalar const a18 = a16 * a17 * a7;
        Scalar const a19 = std::sqrt(42);
        Scalar const a20 = ((X[1]) * (X[1]));
        Scalar const a21 = 15 * a20;
        Scalar const a22 = X[0] * X[1];
        Scalar const a23 = 600 * a22 - 100 * X[1];
        Scalar const a24 = 8 * X[1];
        Scalar const a25 = 2 * X[0];
        Scalar const a26 = a14 - a25;
        Scalar const a27 = a26 + 1;
        Scalar const a28 = a19 * (10 * a20 + a24 * X[0] - a24 + a27);
        Scalar const a29 = std::sqrt(21);
        Scalar const a30 = 60 * a14;
        Scalar const a31 = a29 * (-a30 + 20 * X[0] + 1);
        Scalar const a32 = a11 * a17;
        Scalar const a33 = 2 * a29 * a32;
        Scalar const a34 = X[1] * X[2];
        Scalar const a35 = 5 * X[1];
        Scalar const a36 = 3 * a11 * a13 * (a35 + a6);
        Scalar const a37 = std::sqrt(210);
        Scalar const a38 = ((X[2]) * (X[2]));
        Scalar const a39 = a25 * X[1] + a5 * X[2];
        Scalar const a40 = a37 * (a20 + a27 + 6 * a38 + a39 + a4 * X[2] - 2 * X[1] - 6 * X[2]);
        P[0]             = a0;
        P[1]             = 0;
        P[2]             = 0;
        P[3]             = 0;
        P[4]             = a0;
        P[5]             = 0;
        P[6]             = 0;
        P[7]             = 0;
        P[8]             = a0;
        P[9]             = 0;
        P[10]            = a2;
        P[11]            = 0;
        P[12]            = 0;
        P[13]            = 0;
        P[14]            = a2;
        P[15]            = a3 * (a4 - 3.0 / 2.0);
        P[16]            = 0;
        P[17]            = 0;
        P[18]            = a3 * (3.0 / 2.0 - a5);
        P[19]            = a8;
        P[20]            = 0;
        P[21]            = 0;
        P[22]            = 0;
        P[23]            = a8;
        P[24]            = (1.0 / 2.0) * a9 * (4 * X[1] + 8 * X[2] - 3);
        P[25]            = 0;
        P[26]            = 0;
        P[27]            = (1.0 / 2.0) * a10;
        P[28]            = a12;
        P[29]            = 0;
        P[30]            = a10;
        P[31]            = 0;
        P[32]            = a12;
        P[33]            = 0;
        P[34]            = a15;
        P[35]            = 0;
        P[36]            = 0;
        P[37]            = 0;
        P[38]            = a15;
        P[39]            = a16 * (-18 * a14 + a5 + 3.0 / 10.0);
        P[40]            = a18;
        P[41]            = 0;
        P[42]            = 0;
        P[43]            = 0;
        P[44]            = a18;
        P[45]            = (2.0 / 3.0) * a19 * (a21 - 10 * X[1] + 1);
        P[46]            = 0;
        P[47]            = 0;
        P[48]            = (1.0 / 30.0) * a19 * (-120 * a14 - a23 + 240 * X[0] - 43);
        P[49]            = a28;
        P[50]            = 0;
        P[51]            = 0;
        P[52]            = 0;
        P[53]            = a28;
        P[54]            = (1.0 / 10.0) * a31;
        P[55]            = a33;
        P[56]            = 0;
        P[57]            = (1.0 / 5.0) * a31;
        P[58]            = 0;
        P[59]            = a33;
        P[60]            = a13 * (a21 + 30 * a34 - 15 * X[1] - 5 * X[2] + 2);
        P[61]            = 0;
        P[62]            = 0;
        P[63] =
            (1.0 / 10.0) * a13 *
            (-90 * a14 - 300 * a22 - 300 * X[0] * X[2] + 180 * X[0] + 50 * X[1] + 50 * X[2] - 31);
        P[64] = a36;
        P[65] = 0;
        P[66] = (1.0 / 20.0) * a13 * (-a23 - a30 + 120 * X[0] - 19);
        P[67] = 0;
        P[68] = a36;
        P[69] = (1.0 / 3.0) * a37 * (3 * a20 + 18 * a34 - a35 + 18 * a38 - 15 * X[2] + 2);
        P[70] = 0;
        P[71] = 0;
        P[72] = a37 * (-a26 - a39 + (1.0 / 3.0) * X[1] + X[2] - 1.0 / 3.0);
        P[73] = a40;
        P[74] = 0;
        P[75] = a37 * (3 * a14 - a32 - X[0] - 1.0 / 20.0);
        P[76] = 0;
        P[77] = a40;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<3, 3>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 3;
    inline static constexpr std::size_t kSize  = 50;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P        = Pm.data();
        Scalar const a0  = std::sqrt(6);
        Scalar const a1  = std::sqrt(10);
        Scalar const a2  = 4 * X[0] - 1;
        Scalar const a3  = a1 * a2;
        Scalar const a4  = std::sqrt(5);
        Scalar const a5  = 6 * X[1];
        Scalar const a6  = 6 * X[0];
        Scalar const a7  = X[0] - 1;
        Scalar const a8  = a7 + 3 * X[1];
        Scalar const a9  = 2 * a8;
        Scalar const a10 = a4 * a9;
        Scalar const a11 = std::sqrt(15);
        Scalar const a12 = -a11 * a2;
        Scalar const a13 = a7 + X[1] + 2 * X[2];
        Scalar const a14 = 2 * a11 * a13;
        Scalar const a15 = std::sqrt(14);
        Scalar const a16 = 10 * X[0];
        Scalar const a17 = ((X[0]) * (X[0]));
        Scalar const a18 = 15 * a17;
        Scalar const a19 = a15 * (-a16 + a18 + 1);
        Scalar const a20 = std::sqrt(7);
        Scalar const a21 = a6 - 1;
        Scalar const a22 = a20 * a21 * a9;
        Scalar const a23 = std::sqrt(42);
        Scalar const a24 = 10 * X[1];
        Scalar const a25 = ((X[1]) * (X[1]));
        Scalar const a26 = 15 * a25;
        Scalar const a27 = -100 * X[1];
        Scalar const a28 = X[0] * X[1];
        Scalar const a29 = a27 + 600 * a28;
        Scalar const a30 = 8 * X[1];
        Scalar const a31 = 2 * X[0];
        Scalar const a32 = a17 - a31;
        Scalar const a33 = a32 + 1;
        Scalar const a34 = 10 * a25 + a30 * X[0] - a30 + a33;
        Scalar const a35 = a23 * a34;
        Scalar const a36 = std::sqrt(21);
        Scalar const a37 = 60 * a17;
        Scalar const a38 = a36 * (-a37 + 20 * X[0] + 1);
        Scalar const a39 = a13 * a21;
        Scalar const a40 = 2 * a36 * a39;
        Scalar const a41 = 15 * X[1];
        Scalar const a42 = X[1] * X[2];
        Scalar const a43 = -50 * X[1] - 50 * X[2];
        Scalar const a44 = 5 * X[1];
        Scalar const a45 = a44 + a7;
        Scalar const a46 = a13 * a15;
        Scalar const a47 = 3 * a45 * a46;
        Scalar const a48 = std::sqrt(210);
        Scalar const a49 = ((X[2]) * (X[2]));
        Scalar const a50 = 18 * X[1];
        Scalar const a51 = a31 * X[1];
        Scalar const a52 = a51 + a6 * X[2];
        Scalar const a53 = a25 + a33 - 2 * X[1];
        Scalar const a54 = 6 * a49 + a5 * X[2] + a52 + a53 - 6 * X[2];
        Scalar const a55 = a48 * a54;
        Scalar const a56 = -3 * a17;
        Scalar const a57 = std::numbers::sqrt2_v<Scalar>;
        Scalar const a58 = -189 * a17;
        Scalar const a59 = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a60 = 168 * a59;
        Scalar const a61 = a57 * (a58 + a60 + 54 * X[0] - 3);
        Scalar const a62 = 168 * a17 - 84 * X[0] + 6;
        Scalar const a63 = a62 * a8;
        Scalar const a64 = 1680 * X[0];
        Scalar const a65 = a17 * X[1];
        Scalar const a66 = a27 + 16800 * a65 - 4200 * X[0] * X[1];
        Scalar const a67 = (1.0 / 70.0) * a0;
        Scalar const a68 = 8 * X[0] - 1;
        Scalar const a69 = 3 * a68;
        Scalar const a70 = a0 * a34 * a69;
        Scalar const a71 = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a72 = 56 * a71;
        Scalar const a73 = a25 * X[0];
        Scalar const a74 = 840 * X[0];
        Scalar const a75 = -840 * a17 + 280 * a59 + a74;
        Scalar const a76 = -735 * a25 + 5880 * a73 + a75;
        Scalar const a77 = 45 * a25;
        Scalar const a78 =
            6 * a57 *
            (a18 * X[1] - 30 * a28 + a41 + a56 + a59 + 35 * a71 + a77 * X[0] - a77 + 3 * X[0] - 1);
        Scalar const a79  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a80  = 560 * a59;
        Scalar const a81  = a79 * (420 * a17 - a80 - 60 * X[0] + 1);
        Scalar const a82  = a13 * a62 * a79;
        Scalar const a83  = a17 * X[2];
        Scalar const a84  = 9 * a13 * a45 * a57 * a68;
        Scalar const a85  = -3780 * a17 + 2240 * a59 + a74;
        Scalar const a86  = -168 * a42;
        Scalar const a87  = a25 * X[2];
        Scalar const a88  = 336 * a87 - 5;
        Scalar const a89  = a28 * X[2];
        Scalar const a90  = 12 * X[1];
        Scalar const a91  = 21 * a25;
        Scalar const a92  = 6 * a0 * a13 * (a33 + a90 * X[0] - a90 + a91);
        Scalar const a93  = std::sqrt(30);
        Scalar const a94  = a54 * a69 * a93;
        Scalar const a95  = 3360 * a65 - 1680 * X[0] * X[2];
        Scalar const a96  = a49 * X[1];
        Scalar const a97  = a49 * X[0];
        Scalar const a98  = 840 * a17;
        Scalar const a99  = 6 * a1 * a54 * (a7 + 7 * X[1]);
        Scalar const a100 = (3.0 / 4.0) * a15;
        Scalar const a101 = 24 * X[0];
        Scalar const a102 = 6 * a46 * (a16 * X[2] + a24 * X[2] + 10 * a49 + a51 + a53 - 10 * X[2]);
        P[0]              = a0;
        P[1]              = 0;
        P[2]              = 0;
        P[3]              = 0;
        P[4]              = a0;
        P[5]              = 0;
        P[6]              = 0;
        P[7]              = 0;
        P[8]              = a0;
        P[9]              = 0;
        P[10]             = a3;
        P[11]             = 0;
        P[12]             = 0;
        P[13]             = 0;
        P[14]             = a3;
        P[15]             = a4 * (a5 - 3.0 / 2.0);
        P[16]             = 0;
        P[17]             = 0;
        P[18]             = a4 * (3.0 / 2.0 - a6);
        P[19]             = a10;
        P[20]             = 0;
        P[21]             = 0;
        P[22]             = 0;
        P[23]             = a10;
        P[24]             = (1.0 / 2.0) * a11 * (4 * X[1] + 8 * X[2] - 3);
        P[25]             = 0;
        P[26]             = 0;
        P[27]             = (1.0 / 2.0) * a12;
        P[28]             = a14;
        P[29]             = 0;
        P[30]             = a12;
        P[31]             = 0;
        P[32]             = a14;
        P[33]             = 0;
        P[34]             = a19;
        P[35]             = 0;
        P[36]             = 0;
        P[37]             = 0;
        P[38]             = a19;
        P[39]             = a20 * (-18 * a17 + a6 + 3.0 / 10.0);
        P[40]             = a22;
        P[41]             = 0;
        P[42]             = 0;
        P[43]             = 0;
        P[44]             = a22;
        P[45]             = (2.0 / 3.0) * a23 * (-a24 + a26 + 1);
        P[46]             = 0;
        P[47]             = 0;
        P[48]             = (1.0 / 30.0) * a23 * (-120 * a17 - a29 + 240 * X[0] - 43);
        P[49]             = a35;
        P[50]             = 0;
        P[51]             = 0;
        P[52]             = 0;
        P[53]             = a35;
        P[54]             = (1.0 / 10.0) * a38;
        P[55]             = a40;
        P[56]             = 0;
        P[57]             = (1.0 / 5.0) * a38;
        P[58]             = 0;
        P[59]             = a40;
        P[60]             = a15 * (a26 - a41 + 30 * a42 - 5 * X[2] + 2);
        P[61]             = 0;
        P[62]             = 0;
        P[63]             = (1.0 / 10.0) * a15 *
                (-90 * a17 - 300 * a28 - a43 - 300 * X[0] * X[2] + 180 * X[0] - 31);
        P[64]  = a47;
        P[65]  = 0;
        P[66]  = (1.0 / 20.0) * a15 * (-a29 - a37 + 120 * X[0] - 19);
        P[67]  = 0;
        P[68]  = a47;
        P[69]  = (1.0 / 3.0) * a48 * (3 * a25 - a44 + 18 * a49 + a50 * X[2] - 15 * X[2] + 2);
        P[70]  = 0;
        P[71]  = 0;
        P[72]  = a48 * (-a32 - a52 + (1.0 / 3.0) * X[1] + X[2] - 1.0 / 3.0);
        P[73]  = a55;
        P[74]  = 0;
        P[75]  = a48 * (-a39 - a56 - X[0] - 1.0 / 20.0);
        P[76]  = 0;
        P[77]  = a55;
        P[78]  = 0;
        P[79]  = a61;
        P[80]  = 0;
        P[81]  = 0;
        P[82]  = 0;
        P[83]  = a61;
        P[84]  = 126 * a17 - a60 - 18 * X[0] + 3.0 / 10.0;
        P[85]  = a63;
        P[86]  = 0;
        P[87]  = 0;
        P[88]  = 0;
        P[89]  = a63;
        P[90]  = a67 * (7560 * a17 - 4480 * a59 - a64 - a66 - 67);
        P[91]  = a70;
        P[92]  = 0;
        P[93]  = 0;
        P[94]  = 0;
        P[95]  = a70;
        P[96]  = (15.0 / 4.0) * a57 * (-63 * a25 + a50 + a72 - 1);
        P[97]  = 0;
        P[98]  = 0;
        P[99]  = (3.0 / 28.0) * a57 * (-2520 * a65 - a76 + 5040 * X[0] * X[1] - 650 * X[1] + 117);
        P[100] = a78;
        P[101] = 0;
        P[102] = 0;
        P[103] = 0;
        P[104] = a78;
        P[105] = (1.0 / 10.0) * a81;
        P[106] = a82;
        P[107] = 0;
        P[108] = (1.0 / 5.0) * a81;
        P[109] = 0;
        P[110] = a82;
        P[111] = (3.0 / 70.0) * a57 *
                 (5670 * a17 - a43 - 3360 * a59 - 8400 * a65 - 8400 * a83 + 2100 * X[0] * X[1] +
                  2100 * X[0] * X[2] - 1260 * X[0] - 39);
        P[112] = a84;
        P[113] = 0;
        P[114] = (3.0 / 140.0) * a57 * (-a66 - a85 - 11);
        P[115] = 0;
        P[116] = a84;
        P[117] = (3.0 / 4.0) * a0 * (-231 * a25 + 168 * a71 + a86 + a88 + 78 * X[1] + 12 * X[2]);
        P[118] = 0;
        P[119] = 0;
        P[120] = (1.0 / 28.0) * a0 *
                 (2184 * a17 + 1323 * a25 - 728 * a59 - 5544 * a65 - 10584 * a73 - 2016 * a83 -
                  14112 * a89 + 11088 * X[0] * X[1] + 4032 * X[0] * X[2] - 2184 * X[0] +
                  1764 * X[1] * X[2] - 1416 * X[1] - 534 * X[2] + 295);
        P[121] = a92;
        P[122] = 0;
        P[123] = a67 * (2205 * a25 - 5040 * a65 - 17640 * a73 - a75 + 10080 * X[0] * X[1] -
                        1230 * X[1] + 101);
        P[124] = 0;
        P[125] = a92;
        P[126] = (1.0 / 7.0) * a93 *
                 (-504 * a17 * X[2] - a58 - 112 * a59 - 168 * a65 + 42 * X[0] * X[1] +
                  126 * X[0] * X[2] - 42 * X[0] + X[1] + 3 * X[2] - 1);
        P[127] = a94;
        P[128] = 0;
        P[129] = (3.0 / 140.0) * a93 *
                 (-6720 * a83 - a85 - a95 + 840 * X[0] * X[1] + 20 * X[1] + 40 * X[2] - 29);
        P[130] = 0;
        P[131] = a94;
        P[132] =
            (3.0 / 4.0) * a1 *
            (-105 * a25 - 336 * a42 - 42 * a49 + a72 + a88 + 336 * a96 + 54 * X[1] + 36 * X[2]);
        P[133] = 0;
        P[134] = 0;
        P[135] = (3.0 / 28.0) * a1 *
                 (504 * a17 + 147 * a25 + 294 * a49 - a60 - 1176 * a73 - 1344 * a83 - 4704 * a89 -
                  2352 * a97 - a98 * X[1] + 1680 * X[0] * X[1] + 2688 * X[0] * X[2] - 504 * X[0] +
                  588 * X[1] * X[2] - 212 * X[1] - 342 * X[2] + 65);
        P[136] = a99;
        P[137] = 0;
        P[138] = (3.0 / 70.0) * a1 *
                 (-a76 - 11760 * a89 - a95 - a98 * X[2] + 6720 * X[0] * X[1] + 1470 * X[1] * X[2] -
                  855 * X[1] - 205 * X[2] + 104);
        P[139] = 0;
        P[140] = a99;
        P[141] = a100 * (-210 * a49 + a50 + 8 * a71 + a86 + 96 * a87 - a91 + 240 * a96 +
                         160 * ((X[2]) * (X[2]) * (X[2])) + 72 * X[2] - 5);
        P[142] = 0;
        P[143] = 0;
        P[144] = a100 * (-a101 * a25 - a101 - 24 * a17 * X[1] + 24 * a17 + 3 * a25 -
                         240 * a49 * X[0] + 30 * a49 - a5 - 8 * a59 - 96 * a83 - 192 * a89 +
                         48 * X[0] * X[1] + 192 * X[0] * X[2] + 24 * X[1] * X[2] - 24 * X[2] + 3);
        P[145] = a102;
        P[146] = 0;
        P[147] = (3.0 / 70.0) * a15 *
                 (-1680 * a17 * X[1] + 1680 * a17 - a25 * a64 + 210 * a25 + 1050 * a49 - a64 - a80 -
                  4200 * a83 - 8400 * a89 - 8400 * a97 + 3360 * X[0] * X[1] + 8400 * X[0] * X[2] +
                  1050 * X[1] * X[2] - 435 * X[1] - 1075 * X[2] + 227);
        P[148] = 0;
        P[149] = a102;
        return Pm;
    }
};

template <>
class DivergenceFreeBasis<3, 4>
{
  public:
    inline static constexpr std::size_t kDims  = 3;
    inline static constexpr std::size_t kOrder = 4;
    inline static constexpr std::size_t kSize  = 85;

    [[maybe_unused]] Matrix<kSize, kDims> eval([[maybe_unused]] Vector<kDims> const& X) const
    {
        Matrix<kSize, kDims> Pm;
        Scalar* P         = Pm.data();
        Scalar const a0   = std::sqrt(6);
        Scalar const a1   = std::sqrt(10);
        Scalar const a2   = 4 * X[0];
        Scalar const a3   = a2 - 1;
        Scalar const a4   = a1 * a3;
        Scalar const a5   = std::sqrt(5);
        Scalar const a6   = 6 * X[1];
        Scalar const a7   = 6 * X[0];
        Scalar const a8   = X[0] - 1;
        Scalar const a9   = a8 + 3 * X[1];
        Scalar const a10  = 2 * a9;
        Scalar const a11  = a10 * a5;
        Scalar const a12  = std::sqrt(15);
        Scalar const a13  = 4 * X[1];
        Scalar const a14  = 8 * X[2];
        Scalar const a15  = -a12 * a3;
        Scalar const a16  = a8 + X[1] + 2 * X[2];
        Scalar const a17  = 2 * a16;
        Scalar const a18  = a12 * a17;
        Scalar const a19  = std::sqrt(14);
        Scalar const a20  = 10 * X[0];
        Scalar const a21  = ((X[0]) * (X[0]));
        Scalar const a22  = 15 * a21;
        Scalar const a23  = a19 * (-a20 + a22 + 1);
        Scalar const a24  = std::sqrt(7);
        Scalar const a25  = a7 - 1;
        Scalar const a26  = a10 * a24 * a25;
        Scalar const a27  = std::sqrt(42);
        Scalar const a28  = 10 * X[1];
        Scalar const a29  = ((X[1]) * (X[1]));
        Scalar const a30  = 15 * a29;
        Scalar const a31  = -100 * X[1];
        Scalar const a32  = X[0] * X[1];
        Scalar const a33  = a31 + 600 * a32;
        Scalar const a34  = 8 * X[1];
        Scalar const a35  = 2 * X[0];
        Scalar const a36  = a21 - a35;
        Scalar const a37  = a36 + 1;
        Scalar const a38  = 10 * a29 + a34 * X[0] - a34 + a37;
        Scalar const a39  = a27 * a38;
        Scalar const a40  = std::sqrt(21);
        Scalar const a41  = 20 * X[0];
        Scalar const a42  = 60 * a21;
        Scalar const a43  = a40 * (a41 - a42 + 1);
        Scalar const a44  = a16 * a25;
        Scalar const a45  = 2 * a40 * a44;
        Scalar const a46  = 15 * X[1];
        Scalar const a47  = 90 * a21;
        Scalar const a48  = X[0] * X[2];
        Scalar const a49  = 300 * a48;
        Scalar const a50  = -50 * X[1] - 50 * X[2];
        Scalar const a51  = a16 * a19;
        Scalar const a52  = 5 * X[1];
        Scalar const a53  = a52 + a8;
        Scalar const a54  = 3 * a53;
        Scalar const a55  = a51 * a54;
        Scalar const a56  = std::sqrt(210);
        Scalar const a57  = ((X[2]) * (X[2]));
        Scalar const a58  = 18 * X[1];
        Scalar const a59  = a35 * X[1];
        Scalar const a60  = a59 + a7 * X[2];
        Scalar const a61  = 2 * X[1];
        Scalar const a62  = a29 + a37 - a61;
        Scalar const a63  = 6 * a57 + a6 * X[2] + a60 + a62 - 6 * X[2];
        Scalar const a64  = a56 * a63;
        Scalar const a65  = -3 * a21;
        Scalar const a66  = std::numbers::sqrt2_v<Scalar>;
        Scalar const a67  = -189 * a21;
        Scalar const a68  = ((X[0]) * (X[0]) * (X[0]));
        Scalar const a69  = 168 * a68;
        Scalar const a70  = a66 * (a67 + a69 + 54 * X[0] - 3);
        Scalar const a71  = 168 * a21 - 84 * X[0] + 6;
        Scalar const a72  = a71 * a9;
        Scalar const a73  = 1680 * X[0];
        Scalar const a74  = a21 * X[1];
        Scalar const a75  = a31 + 16800 * a74 - 4200 * X[0] * X[1];
        Scalar const a76  = (1.0 / 70.0) * a0;
        Scalar const a77  = 8 * X[0] - 1;
        Scalar const a78  = 3 * a77;
        Scalar const a79  = a0 * a38 * a78;
        Scalar const a80  = ((X[1]) * (X[1]) * (X[1]));
        Scalar const a81  = 56 * a80;
        Scalar const a82  = -735 * a29;
        Scalar const a83  = a29 * X[0];
        Scalar const a84  = 840 * X[0];
        Scalar const a85  = -840 * a21 + 280 * a68 + a84;
        Scalar const a86  = a82 + 5880 * a83 + a85;
        Scalar const a87  = 45 * a29;
        Scalar const a88  = 3 * X[0] - 1;
        Scalar const a89  = a65 + a68 + a88;
        Scalar const a90  = a22 * X[1] - 30 * a32 + a46 + 35 * a80 + a87 * X[0] - a87 + a89;
        Scalar const a91  = 6 * a66 * a90;
        Scalar const a92  = std::numbers::sqrt3_v<Scalar>;
        Scalar const a93  = 60 * X[0];
        Scalar const a94  = 560 * a68;
        Scalar const a95  = a92 * (420 * a21 - a93 - a94 + 1);
        Scalar const a96  = a16 * a71 * a92;
        Scalar const a97  = a21 * X[2];
        Scalar const a98  = 9 * a16 * a53 * a66 * a77;
        Scalar const a99  = -3780 * a21 + 2240 * a68 + a84;
        Scalar const a100 = X[1] * X[2];
        Scalar const a101 = 168 * a100;
        Scalar const a102 = -a101;
        Scalar const a103 = a29 * X[2];
        Scalar const a104 = 336 * a103 - 5;
        Scalar const a105 = 10584 * a83;
        Scalar const a106 = a32 * X[2];
        Scalar const a107 = 12 * X[1];
        Scalar const a108 = 21 * a29;
        Scalar const a109 = a107 * X[0];
        Scalar const a110 = -a107 + a108 + a109 + a37;
        Scalar const a111 = 6 * a0 * a110 * a16;
        Scalar const a112 = std::sqrt(30);
        Scalar const a113 = 42 * X[0];
        Scalar const a114 = -a113 * X[1];
        Scalar const a115 = 168 * a21;
        Scalar const a116 = -X[1] - 3 * X[2] + 1;
        Scalar const a117 = a112 * a63 * a78;
        Scalar const a118 = -20 * X[1] - 40 * X[2];
        Scalar const a119 = 3360 * a74 - 1680 * X[0] * X[2];
        Scalar const a120 = a57 * X[1];
        Scalar const a121 = 342 * X[2];
        Scalar const a122 = 504 * X[0];
        Scalar const a123 = 840 * a21;
        Scalar const a124 = a57 * X[0];
        Scalar const a125 = -588 * X[1] * X[2];
        Scalar const a126 = a125 - 147 * a29 - 294 * a57;
        Scalar const a127 = a63 * (a8 + 7 * X[1]);
        Scalar const a128 = 6 * a1 * a127;
        Scalar const a129 = -1470 * X[1] * X[2];
        Scalar const a130 = ((X[2]) * (X[2]) * (X[2]));
        Scalar const a131 = (3.0 / 4.0) * a19;
        Scalar const a132 = 24 * X[0];
        Scalar const a133 = a20 * X[2] + a28 * X[2] + 10 * a57 + a59 + a62 - 10 * X[2];
        Scalar const a134 = 6 * a133 * a51;
        Scalar const a135 = 1680 * a21;
        Scalar const a136 = -a135;
        Scalar const a137 = 8400 * a48;
        Scalar const a138 = 3360 * X[0] * X[1];
        Scalar const a139 = 8400 * a32;
        Scalar const a140 = -210 * a29 - 1050 * a57 - 1050 * X[1] * X[2];
        Scalar const a141 = std::sqrt(22);
        Scalar const a142 = ((X[0]) * (X[0]) * (X[0]) * (X[0]));
        Scalar const a143 = a141 * (a115 + 210 * a142 - 336 * a68 - 28 * X[0] + 1);
        Scalar const a144 = std::sqrt(11);
        Scalar const a145 = 72 * a21;
        Scalar const a146 = 180 * a142;
        Scalar const a147 = a132 - 108 * a21 + 120 * a68 - 1;
        Scalar const a148 = a10 * a144 * a147;
        Scalar const a149 = std::sqrt(66);
        Scalar const a150 = a68 * X[1];
        Scalar const a151 = 252000 * a150 - 151200 * a21 * X[1] + a31 + 16800 * a32;
        Scalar const a152 = a88 * (15 * X[0] - 1);
        Scalar const a153 = a149 * a152 * a38;
        Scalar const a154 = 45360 * a21;
        Scalar const a155 = a21 * a29;
        Scalar const a156 = 264600 * a155 - 52920 * a29 * X[0] + a82;
        Scalar const a157 = 18900 * a142 + a154 + a156 - 52920 * a68 - 7560 * X[0];
        Scalar const a158 = a20 - 1;
        Scalar const a159 = 2 * a158;
        Scalar const a160 = a141 * a159 * a90;
        Scalar const a161 = ((X[1]) * (X[1]) * (X[1]) * (X[1]));
        Scalar const a162 = std::sqrt(110);
        Scalar const a163 = (3.0 / 5.0) * a162;
        Scalar const a164 = 90720 * a21;
        Scalar const a165 = a80 * X[0];
        Scalar const a166 = 224 * a80;
        Scalar const a167 = 24 * X[1];
        Scalar const a168 = 6 * a21;
        Scalar const a169 = a142 + a168 - a2 - 4 * a68 + 1;
        Scalar const a170 =
            a162 * (-a145 * X[1] + 126 * a161 + a166 * X[0] - a166 + a167 * a68 - a167 + a169 +
                    126 * a21 * a29 + 126 * a29 - 252 * a83 + 72 * X[0] * X[1]);
        Scalar const a171 = std::sqrt(33);
        Scalar const a172 = 140 * X[0];
        Scalar const a173 = a171 * (a136 - 4200 * a142 + a172 + 5040 * a68 + 1);
        Scalar const a174 = a147 * a17 * a171;
        Scalar const a175 = -105840 * a68;
        Scalar const a176 = a68 * X[2];
        Scalar const a177 = a141 * a152 * a16 * a54;
        Scalar const a178 = 37800 * a142;
        Scalar const a179 = a178 + 31920 * a21 - 70560 * a68 - 3360 * X[0];
        Scalar const a180 = 110880 * a150;
        Scalar const a181 = 211680 * a74 * X[2] - 42336 * X[0] * X[1] * X[2];
        Scalar const a182 = a158 * a17;
        Scalar const a183 = a110 * a149 * a182;
        Scalar const a184 = 15120 * a21;
        Scalar const a185 = 30240 * X[0] * X[1];
        Scalar const a186 = a184 + a185;
        Scalar const a187 = 100800 * a150 - 166320 * a21 * X[1];
        Scalar const a188 = std::sqrt(330);
        Scalar const a189 = a80 * X[2];
        Scalar const a190 = a74 * X[2];
        Scalar const a191 = a29 * a48;
        Scalar const a192 = 846720 * a165 - 84672 * a80;
        Scalar const a193 = 84 * a29;
        Scalar const a194 = 21 * X[1];
        Scalar const a195 =
            a16 * a188 * (a114 + a193 * X[0] - a193 + a194 * a21 + a194 + 84 * a80 + a89);
        Scalar const a196 = 211680 * a32;
        Scalar const a197 = 2520 * a142 - 10080 * a68 - 10080 * X[0];
        Scalar const a198 = a152 * a188 * a63;
        Scalar const a199 = -60480 * a21 * X[2];
        Scalar const a200 = -30240 * a21 * X[1];
        Scalar const a201 = 50400 * a150;
        Scalar const a202 = 15120 * X[0];
        Scalar const a203 = a21 * a57;
        Scalar const a204 = 105840 * a203;
        Scalar const a205 = a127 * a159 * a162;
        Scalar const a206 = 60480 * X[0];
        Scalar const a207 = 50400 * a176 - 332640 * a21 * X[1];
        Scalar const a208 = a29 * a57;
        Scalar const a209 = a32 * a57;
        Scalar const a210 = 16 * X[1];
        Scalar const a211 = 5 * a141 * a63 * (a210 * X[0] - a210 + 36 * a29 + a37);
        Scalar const a212 = 7560 * a142 + a154 - 30240 * a68 - 30240 * X[0];
        Scalar const a213 = std::sqrt(154);
        Scalar const a214 = a133 * a182 * a213;
        Scalar const a215 = std::sqrt(770);
        Scalar const a216 = a130 * X[1];
        Scalar const a217 = 450 * a208 - 14 * X[1];
        Scalar const a218 = a130 * X[0];
        Scalar const a219 = a133 * a16 * a215 * (a8 + 9 * X[1]);
        Scalar const a220 = ((X[2]) * (X[2]) * (X[2]) * (X[2]));
        Scalar const a221 = 30 * a21;
        Scalar const a222 = 90 * a57;
        Scalar const a223 = 60 * a100 + a222 + 6 * a29;
        Scalar const a224 = 140 * a130;
        Scalar const a225 = 20 * X[2];
        Scalar const a226 = 180 * X[0];
        Scalar const a227 = 3 * a162 *
                            (a100 * a42 - 120 * a100 * X[0] + a103 * a93 - 60 * a103 - a107 * a21 +
                             a109 + a120 * a226 - 180 * a120 + a13 * a68 - a13 + a130 * a172 +
                             a161 + a168 * a29 + a169 + a2 * a80 + 70 * a220 + a222 * a29 + a223 +
                             a224 * X[1] - a224 + a225 * a80 - a225 - a226 * a57 - a42 * X[2] +
                             a47 * a57 + 20 * a68 * X[2] - 4 * a80 - 12 * a83 + a93 * X[2]);
        Scalar const a228 = 90720 * a48;
        P[0]              = a0;
        P[1]              = 0;
        P[2]              = 0;
        P[3]              = 0;
        P[4]              = a0;
        P[5]              = 0;
        P[6]              = 0;
        P[7]              = 0;
        P[8]              = a0;
        P[9]              = 0;
        P[10]             = a4;
        P[11]             = 0;
        P[12]             = 0;
        P[13]             = 0;
        P[14]             = a4;
        P[15]             = a5 * (a6 - 3.0 / 2.0);
        P[16]             = 0;
        P[17]             = 0;
        P[18]             = a5 * (3.0 / 2.0 - a7);
        P[19]             = a11;
        P[20]             = 0;
        P[21]             = 0;
        P[22]             = 0;
        P[23]             = a11;
        P[24]             = (1.0 / 2.0) * a12 * (a13 + a14 - 3);
        P[25]             = 0;
        P[26]             = 0;
        P[27]             = (1.0 / 2.0) * a15;
        P[28]             = a18;
        P[29]             = 0;
        P[30]             = a15;
        P[31]             = 0;
        P[32]             = a18;
        P[33]             = 0;
        P[34]             = a23;
        P[35]             = 0;
        P[36]             = 0;
        P[37]             = 0;
        P[38]             = a23;
        P[39]             = a24 * (-18 * a21 + a7 + 3.0 / 10.0);
        P[40]             = a26;
        P[41]             = 0;
        P[42]             = 0;
        P[43]             = 0;
        P[44]             = a26;
        P[45]             = (2.0 / 3.0) * a27 * (-a28 + a30 + 1);
        P[46]             = 0;
        P[47]             = 0;
        P[48]             = (1.0 / 30.0) * a27 * (-120 * a21 - a33 + 240 * X[0] - 43);
        P[49]             = a39;
        P[50]             = 0;
        P[51]             = 0;
        P[52]             = 0;
        P[53]             = a39;
        P[54]             = (1.0 / 10.0) * a43;
        P[55]             = a45;
        P[56]             = 0;
        P[57]             = (1.0 / 5.0) * a43;
        P[58]             = 0;
        P[59]             = a45;
        P[60]             = a19 * (a30 - a46 + 30 * X[1] * X[2] - 5 * X[2] + 2);
        P[61]             = 0;
        P[62]             = 0;
        P[63]             = (1.0 / 10.0) * a19 * (-300 * a32 - a47 - a49 - a50 + 180 * X[0] - 31);
        P[64]             = a55;
        P[65]             = 0;
        P[66]             = (1.0 / 20.0) * a19 * (-a33 - a42 + 120 * X[0] - 19);
        P[67]             = 0;
        P[68]             = a55;
        P[69]  = (1.0 / 3.0) * a56 * (3 * a29 - a52 + 18 * a57 + a58 * X[2] - 15 * X[2] + 2);
        P[70]  = 0;
        P[71]  = 0;
        P[72]  = a56 * (-a36 - a60 + (1.0 / 3.0) * X[1] + X[2] - 1.0 / 3.0);
        P[73]  = a64;
        P[74]  = 0;
        P[75]  = a56 * (-a44 - a65 - X[0] - 1.0 / 20.0);
        P[76]  = 0;
        P[77]  = a64;
        P[78]  = 0;
        P[79]  = a70;
        P[80]  = 0;
        P[81]  = 0;
        P[82]  = 0;
        P[83]  = a70;
        P[84]  = 126 * a21 - a69 - 18 * X[0] + 3.0 / 10.0;
        P[85]  = a72;
        P[86]  = 0;
        P[87]  = 0;
        P[88]  = 0;
        P[89]  = a72;
        P[90]  = a76 * (7560 * a21 - 4480 * a68 - a73 - a75 - 67);
        P[91]  = a79;
        P[92]  = 0;
        P[93]  = 0;
        P[94]  = 0;
        P[95]  = a79;
        P[96]  = (15.0 / 4.0) * a66 * (-63 * a29 + a58 + a81 - 1);
        P[97]  = 0;
        P[98]  = 0;
        P[99]  = (3.0 / 28.0) * a66 * (-2520 * a74 - a86 + 5040 * X[0] * X[1] - 650 * X[1] + 117);
        P[100] = a91;
        P[101] = 0;
        P[102] = 0;
        P[103] = 0;
        P[104] = a91;
        P[105] = (1.0 / 10.0) * a95;
        P[106] = a96;
        P[107] = 0;
        P[108] = (1.0 / 5.0) * a95;
        P[109] = 0;
        P[110] = a96;
        P[111] = (3.0 / 70.0) * a66 *
                 (5670 * a21 - a50 - 3360 * a68 - 8400 * a74 - 8400 * a97 + 2100 * X[0] * X[1] +
                  2100 * X[0] * X[2] - 1260 * X[0] - 39);
        P[112] = a98;
        P[113] = 0;
        P[114] = (3.0 / 140.0) * a66 * (-a75 - a99 - 11);
        P[115] = 0;
        P[116] = a98;
        P[117] = (3.0 / 4.0) * a0 * (a102 + a104 - 231 * a29 + 168 * a80 + 78 * X[1] + 12 * X[2]);
        P[118] = 0;
        P[119] = 0;
        P[120] = (1.0 / 28.0) * a0 *
                 (-a105 - 14112 * a106 + 2184 * a21 + 1323 * a29 - 728 * a68 - 5544 * a74 -
                  2016 * a97 + 11088 * X[0] * X[1] + 4032 * X[0] * X[2] - 2184 * X[0] +
                  1764 * X[1] * X[2] - 1416 * X[1] - 534 * X[2] + 295);
        P[121] = a111;
        P[122] = 0;
        P[123] = a76 * (2205 * a29 - 5040 * a74 - 17640 * a83 - a85 + 10080 * X[0] * X[1] -
                        1230 * X[1] + 101);
        P[124] = 0;
        P[125] = a111;
        P[126] = (1.0 / 7.0) * a112 *
                 (-a113 - a114 - a115 * X[1] - a116 - 504 * a21 * X[2] - a67 - 112 * a68 +
                  126 * X[0] * X[2]);
        P[127] = a117;
        P[128] = 0;
        P[129] = (3.0 / 140.0) * a112 * (-a118 - a119 - 6720 * a97 - a99 + 840 * X[0] * X[1] - 29);
        P[130] = 0;
        P[131] = a117;
        P[132] =
            (3.0 / 4.0) * a1 *
            (-336 * a100 + a104 + 336 * a120 - 105 * a29 - 42 * a57 + a81 + 54 * X[1] + 36 * X[2]);
        P[133] = 0;
        P[134] = 0;
        P[135] =
            (3.0 / 28.0) * a1 *
            (-4704 * a106 - a121 - a122 - a123 * X[1] - 2352 * a124 - a126 + 504 * a21 - a69 -
             1176 * a83 - 1344 * a97 + 1680 * X[0] * X[1] + 2688 * X[0] * X[2] - 212 * X[1] + 65);
        P[136] = a128;
        P[137] = 0;
        P[138] = (3.0 / 70.0) * a1 *
                 (-11760 * a106 - a119 - a123 * X[2] - a129 - a86 + 6720 * X[0] * X[1] -
                  855 * X[1] - 205 * X[2] + 104);
        P[139] = 0;
        P[140] = a128;
        P[141] = a131 * (a102 + 96 * a103 - a108 + 240 * a120 + 160 * a130 - 210 * a57 + a58 +
                         8 * a80 + 72 * X[2] - 5);
        P[142] = 0;
        P[143] = 0;
        P[144] = a131 * (-192 * a106 - a132 * a29 - a132 - 24 * a21 * X[1] + 24 * a21 + 3 * a29 -
                         240 * a57 * X[0] + 30 * a57 - a6 - 8 * a68 - 96 * a97 + 48 * X[0] * X[1] +
                         192 * X[0] * X[2] + 24 * X[1] * X[2] - 24 * X[2] + 3);
        P[145] = a134;
        P[146] = 0;
        P[147] = (3.0 / 70.0) * a19 *
                 (-8400 * a124 - a135 * X[1] - a136 + a137 + a138 - a139 * X[2] - a140 - a29 * a73 -
                  a73 - a94 - 4200 * a97 - 435 * X[1] - 1075 * X[2] + 227);
        P[148] = 0;
        P[149] = a134;
        P[150] = 0;
        P[151] = a143;
        P[152] = 0;
        P[153] = 0;
        P[154] = 0;
        P[155] = a143;
        P[156] = a144 * (-a145 - a146 + 216 * a68 + a7 + 3.0 / 70.0);
        P[157] = a148;
        P[158] = 0;
        P[159] = 0;
        P[160] = 0;
        P[161] = a148;
        P[162] = (1.0 / 840.0) * a149 *
                 (-75600 * a142 - a151 - 63840 * a21 + 141120 * a68 + 6720 * X[0] - 97);
        P[163] = a153;
        P[164] = 0;
        P[165] = 0;
        P[166] = 0;
        P[167] = a153;
        P[168] = (1.0 / 252.0) * a141 *
                 (-151200 * a150 - a157 + 249480 * a21 * X[1] - 45360 * a32 - 850 * X[1] + 247);
        P[169] = a160;
        P[170] = 0;
        P[171] = 0;
        P[172] = 0;
        P[173] = a160;
        P[174] = a163 * (210 * a161 + 168 * a29 - 336 * a80 - 28 * X[1] + 1);
        P[175] = 0;
        P[176] = 0;
        P[177] = (1.0 / 2520.0) * a162 *
                 (-15120 * a142 - 211680 * a150 - 846720 * a155 - a164 - 1270080 * a165 +
                  635040 * a21 * X[1] + 1693440 * a29 * X[0] - 172284 * a29 - 635040 * a32 +
                  60480 * a68 + 127008 * a80 + 60480 * X[0] + 66724 * X[1] - 6883);
        P[178] = a170;
        P[179] = 0;
        P[180] = 0;
        P[181] = 0;
        P[182] = a170;
        P[183] = (1.0 / 70.0) * a173;
        P[184] = a174;
        P[185] = 0;
        P[186] = (1.0 / 35.0) * a173;
        P[187] = 0;
        P[188] = a174;
        P[189] = (1.0 / 280.0) * a141 *
                 (-a137 - a139 - 56700 * a142 - 126000 * a150 - a175 - 126000 * a176 +
                  75600 * a21 * X[1] + 75600 * a21 * X[2] - 47880 * a21 - a50 + 5040 * X[0] - 49);
        P[190] = a177;
        P[191] = 0;
        P[192] = (1.0 / 560.0) * a141 * (-a151 - a179 - 1);
        P[193] = 0;
        P[194] = a177;
        P[195] = (1.0 / 252.0) * a149 *
                 (-a125 - 16380 * a142 - 158760 * a155 - 40320 * a176 - a180 - a181 +
                  182952 * a21 * X[1] + 66528 * a21 * X[2] - 39312 * a21 + 31752 * a29 * X[0] +
                  441 * a29 - 33264 * a32 - 12096 * a48 + 45864 * a68 + 6552 * X[0] - 572 * X[1] -
                  278 * X[2] + 175);
        P[196] = a183;
        P[197] = 0;
        P[198] = (1.0 / 630.0) * a149 *
                 (-6300 * a142 - a156 - a186 - a187 + 17640 * a68 + 2520 * X[0] - 310 * X[1] + 22);
        P[199] = 0;
        P[200] = a183;
        P[201] = (1.0 / 5.0) * a188 *
                 (a101 - 756 * a103 + 420 * a161 + 840 * a189 + 420 * a29 - 756 * a80 - 77 * X[1] -
                  7 * X[2] + 3);
        P[202] = 0;
        P[203] = 0;
        P[204] = (1.0 / 2520.0) * a188 *
                 (-87024 * a100 - 13860 * a142 - 176400 * a150 - 635040 * a155 - 35280 * a176 -
                  423360 * a190 - 1270080 * a191 - a192 + 529200 * a21 * X[1] +
                  105840 * a21 * X[2] - 83160 * a21 + 1270080 * a29 * X[0] + 127008 * a29 * X[2] -
                  128772 * a29 - 529200 * a32 - 105840 * a48 + 55440 * a68 +
                  846720 * X[0] * X[1] * X[2] + 55440 * X[0] + 55118 * X[1] + 11606 * X[2] - 6163);
        P[205] = a195;
        P[206] = 0;
        P[207] = (1.0 / 5040.0) * a188 *
                 (-423360 * a155 - a184 - a192 - a196 - a197 + 211680 * a21 * X[1] +
                  846720 * a29 * X[0] - 83496 * a29 - 70560 * a68 * X[1] + 20636 * X[1] - 971);
        P[208] = 0;
        P[209] = a195;
        P[210] =
            (1.0 / 84.0) * a188 *
            (-a116 - a122 * X[2] - 1890 * a142 - 2520 * a150 - 7560 * a176 + 1512 * a21 * X[1] +
             4536 * a21 * X[2] - 1596 * a21 + 3528 * a68 - 168 * X[0] * X[1] + 168 * X[0]);
        P[211] = a198;
        P[212] = 0;
        P[213] =
            (1.0 / 560.0) * a188 *
            (-a118 - a138 - 100800 * a176 - a179 - a199 - a200 - a201 - 6720 * X[0] * X[2] - 39);
        P[214] = 0;
        P[215] = a198;
        P[216] =
            (1.0 / 252.0) * a162 *
            (a105 - a126 - 11340 * a142 - 52920 * a155 - 80640 * a176 - a181 - a201 - a202 * X[1] -
             a204 + 83160 * a21 * X[1] + 133056 * a21 * X[2] - 27216 * a21 - 24192 * a48 +
             21168 * a57 * X[0] + 31752 * a68 + 4536 * X[0] - 232 * X[1] - 402 * X[2] + 85);
        P[217] = a205;
        P[218] = 0;
        P[219] =
            (1.0 / 630.0) * a162 *
            (-a129 - 201600 * a150 - a157 - a202 * X[2] - a206 * X[1] - a207 + 83160 * a21 * X[2] -
             529200 * a74 * X[2] + 105840 * X[0] * X[1] * X[2] - 1005 * X[1] - 155 * X[2] + 89);
        P[220] = 0;
        P[221] = a205;
        P[222] =
            a141 * (408 * a100 - 1404 * a103 - 432 * a120 + 180 * a161 + 1080 * a189 + 1080 * a208 +
                    276 * a29 + 24 * a57 - 396 * a80 - 63 * X[1] - 21 * X[2] + 3);
        P[223] = 0;
        P[224] = 0;
        P[225] = (1.0 / 36.0) * a141 *
                 (-11400 * a100 - 810 * a142 - 8280 * a150 - 23760 * a155 - 25920 * a165 -
                  6120 * a176 - 56160 * a190 - 116640 * a191 - 8640 * a203 - 77760 * a209 +
                  24840 * a21 * X[1] + 18360 * a21 * X[2] - 4860 * a21 + 47520 * a29 * X[0] +
                  11664 * a29 * X[2] - 4794 * a29 - 24840 * a32 - 18360 * a48 + 17280 * a57 * X[0] +
                  7776 * a57 * X[1] - 1812 * a57 + 3240 * a68 + 2592 * a80 +
                  112320 * X[0] * X[1] * X[2] + 3240 * X[0] + 2549 * X[1] + 1947 * X[2] - 347);
        P[226] = a211;
        P[227] = 0;
        P[228] = (1.0 / 1008.0) * a141 *
                 (-95424 * a100 - 171360 * a150 - 786240 * a155 - 1088640 * a165 - 20160 * a176 -
                  483840 * a190 - 2177280 * a191 - a199 - a206 * X[2] + 514080 * a21 * X[1] - a212 +
                  1572480 * a29 * X[0] + 217728 * a29 * X[2] - 158088 * a29 - 514080 * a32 +
                  108864 * a80 + 967680 * X[0] * X[1] * X[2] + 51204 * X[1] + 5896 * X[2] - 2983);
        P[229] = 0;
        P[230] = a211;
        P[231] = (1.0 / 12.0) * a213 *
                 (-a14 - a146 - 480 * a150 - 360 * a155 - 1920 * a176 - 2880 * a190 - 3600 * a203 +
                  792 * a21 * X[1] + 3168 * a21 * X[2] - 432 * a21 + 72 * a29 * X[0] + a29 -
                  144 * a32 - 576 * a48 + 720 * a57 * X[0] + 10 * a57 - a61 + 504 * a68 +
                  576 * X[0] * X[1] * X[2] + 72 * X[0] + 8 * X[1] * X[2] + 1);
        P[232] = a214;
        P[233] = 0;
        P[234] = (1.0 / 630.0) * a213 *
                 (-a140 - 75600 * a155 - a164 - a175 - 252000 * a176 - a178 - a185 - a187 -
                  378000 * a190 - 378000 * a203 + 415800 * a21 * X[2] + 15120 * a29 * X[0] -
                  75600 * a48 + 75600 * a57 * X[0] + 75600 * X[0] * X[1] * X[2] + 15120 * X[0] -
                  585 * X[1] - 1325 * X[2] + 407);
        P[235] = 0;
        P[236] = a214;
        P[237] = (3.0 / 5.0) * a215 *
                 (176 * a100 - 450 * a120 - a121 * a29 - 30 * a130 + 15 * a161 + 180 * a189 +
                  300 * a216 + a217 + 40 * a29 + 40 * a57 - 42 * a80 - 14 * X[2] + 1);
        P[238] = 0;
        P[239] = 0;
        P[240] = (1.0 / 30.0) * a215 *
                 (-1376 * a100 + 540 * a130 - 90 * a142 - 600 * a150 - 1260 * a155 - 1080 * a165 -
                  1320 * a176 - 6840 * a190 - 9720 * a191 - 4500 * a203 - 16200 * a209 +
                  1800 * a21 * X[1] + 3960 * a21 * X[2] - 540 * a21 - 5400 * a218 +
                  2520 * a29 * X[0] + 972 * a29 * X[2] - 253 * a29 - 1800 * a32 - 3960 * a48 +
                  9000 * a57 * X[0] + 1620 * a57 * X[1] - 910 * a57 + 360 * a68 + 108 * a80 +
                  13680 * X[0] * X[1] * X[2] + 360 * X[0] + 182 * X[1] + 404 * X[2] - 37);
        P[241] = a219;
        P[242] = 0;
        P[243] = (1.0 / 2520.0) * a215 *
                 (-152880 * a100 - 287280 * a155 - 272160 * a165 - a180 - 756000 * a190 -
                  1360800 * a191 - 75600 * a203 - a207 - 1360800 * a209 + 151200 * a21 * X[2] -
                  a212 + 574560 * a29 * X[0] + 136080 * a29 * X[2] - 58548 * a29 - 332640 * a32 -
                  151200 * a48 + 151200 * a57 * X[0] + 136080 * a57 * X[1] - 14910 * a57 +
                  27216 * a80 + 1512000 * X[0] * X[1] * X[2] + 34488 * X[1] + 15080 * X[2] - 3083);
        P[244] = 0;
        P[245] = a219;
        P[246] = a163 *
                 (240 * a100 - 270 * a103 - 810 * a120 - 630 * a130 + 5 * a161 + 100 * a189 +
                  700 * a216 + a217 + 350 * a220 + 24 * a29 + 360 * a57 - 18 * a80 - 70 * X[2] + 3);
        P[247] = 0;
        P[248] = 0;
        P[249] =
            a163 * (70 * a130 - 5 * a142 - 100 * a176 - 300 * a190 - 450 * a203 - 900 * a209 +
                    60 * a21 * X[1] + 300 * a21 * X[2] - 700 * a218 - a221 * a29 - a221 - a223 -
                    a29 * a49 + 60 * a29 * X[0] + 30 * a29 * X[2] - a41 * a80 - a49 +
                    900 * a57 * X[0] + 90 * a57 * X[1] + a6 - 20 * a68 * X[1] + 20 * a68 + 2 * a80 -
                    a93 * X[1] + 600 * X[0] * X[1] * X[2] + 20 * X[0] + 30 * X[2] - 2);
        P[250] = a227;
        P[251] = 0;
        P[252] = (1.0 / 168.0) * a162 *
                 (-a100 * a164 - 18480 * a100 + 14112 * a130 - a184 * a29 - a186 - a196 * a57 -
                  a197 - a200 - a204 + 90720 * a21 * X[2] - 141120 * a218 - a228 * a29 - a228 +
                  30240 * a29 * X[0] + 9072 * a29 * X[2] - 3108 * a29 + 211680 * a57 * X[0] +
                  21168 * a57 * X[1] - 21462 * a57 - 10080 * a68 * X[1] - 30240 * a68 * X[2] -
                  10080 * a80 * X[0] + 1008 * a80 + 181440 * X[0] * X[1] * X[2] + 3208 * X[1] +
                  9432 * X[2] - 1111);
        P[253] = 0;
        P[254] = a227;
        return Pm;
    }
};

} // namespace detail
} // namespace polynomial
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_POLYNOMIAL_BASIS_H
