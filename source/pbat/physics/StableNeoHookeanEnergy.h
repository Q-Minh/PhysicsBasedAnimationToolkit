/**
 * @file StableNeoHookeanEnergy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Stable Neo-Hookean \cite smith2018snh hyperelastic energy
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup physics
 */

#ifndef PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
#define PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <cmath>

namespace pbat {
namespace physics {

template <int Dims>
struct StableNeoHookeanEnergy;

/**
 * @brief Stable Neo-Hookean hyperelastic energy for 1D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct StableNeoHookeanEnergy<1>
{
  public:
    template <class TScalar, int M, int N>
    using SMatrix = pbat::math::linalg::mini::SMatrix<TScalar, M, N>; ///< Scalar matrix type

    template <class TScalar, int M>
    using SVector = pbat::math::linalg::mini::SVector<TScalar, M>; ///< Scalar vector type

    static auto constexpr kDims = 1; ///< Dimension of the space

    /**
     * @brief Evaluate the elastic energy
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 1>
    grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 1, 1>
    hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return ScalarType Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGrad(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF) const;

    /**
     * @brief Evaluate the elastic energy with its gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     * @return ScalarType Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;

    /**
     * @brief Evaluate the elastic energy gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE void gradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    psi = (1.0 / 2.0) * lambda * ((F[0] - 1 - mu / lambda) * (F[0] - 1 - mu / lambda)) +
          (1.0 / 2.0) * mu * (((F[0]) * (F[0])) - 1);
    return psi;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<1>::SVector<typename TMatrix::ScalarType, 1>
StableNeoHookeanEnergy<1>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 1> G;
    G[0] = (1.0 / 2.0) * lambda * (2 * F[0] - 2 - 2 * mu / lambda) + mu * F[0];
    return G;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<1>::SMatrix<typename TMatrix::ScalarType, 1, 1>
StableNeoHookeanEnergy<1>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 1, 1> H;
    H[0] = lambda + mu;
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::evalWithGrad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF) const
{
    static_assert(
        TMatrixGF::kRows == 1 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 1x1");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = mu / lambda;
    ScalarType const a1 = (1.0 / 2.0) * lambda;
    psi   = a1 * ((-a0 + F[0] - 1) * (-a0 + F[0] - 1)) + (1.0 / 2.0) * mu * (((F[0]) * (F[0])) - 1);
    gF[0] = a1 * (-2 * a0 + 2 * F[0] - 2) + mu * F[0];
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::evalWithGradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 1 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 1x1");
    static_assert(
        TMatrixHF::kRows == 1 and TMatrixHF::kCols == 1,
        "Hessian w.r.t. F must have dimensions 1x1");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = mu / lambda;
    ScalarType const a1 = (1.0 / 2.0) * lambda;
    psi   = a1 * ((-a0 + F[0] - 1) * (-a0 + F[0] - 1)) + (1.0 / 2.0) * mu * (((F[0]) * (F[0])) - 1);
    gF[0] = a1 * (-2 * a0 + 2 * F[0] - 2) + mu * F[0];
    HF[0] = lambda + mu;
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<1>::gradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 1 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 1x1");
    static_assert(
        TMatrixHF::kRows == 1 and TMatrixHF::kCols == 1,
        "Hessian w.r.t. F must have dimensions 1x1");
    using ScalarType = typename TMatrix::ScalarType;
    gF[0]            = (1.0 / 2.0) * lambda * (2 * F[0] - 2 - 2 * mu / lambda) + mu * F[0];
    HF[0]            = lambda + mu;
}

/**
 * @brief Stable Neo-Hookean hyperelastic energy for 2D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct StableNeoHookeanEnergy<2>
{
  public:
    template <class TScalar, int M, int N>
    using SMatrix = pbat::math::linalg::mini::SMatrix<TScalar, M, N>; ///< Scalar matrix type

    template <class TScalar, int M>
    using SVector = pbat::math::linalg::mini::SVector<TScalar, M>; ///< Scalar vector type

    static auto constexpr kDims = 2; ///< Dimension of the space

    /**
     * @brief Evaluate the elastic energy
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 4>
    grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 4, 4>
    hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return ScalarType Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGrad(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF) const;

    /**
     * @brief Evaluate the elastic energy with its gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     * @return ScalarType Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;

    /**
     * @brief Evaluate the elastic energy gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE void gradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    psi = (1.0 / 2.0) * lambda *
              ((F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda) *
               (F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda)) +
          (1.0 / 2.0) * mu *
              (((F[0]) * (F[0])) + ((F[1]) * (F[1])) + ((F[2]) * (F[2])) + ((F[3]) * (F[3])) - 2);
    return psi;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<2>::SVector<typename TMatrix::ScalarType, 4>
StableNeoHookeanEnergy<2>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 4> G;
    ScalarType const a0 = lambda * (F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda);
    G[0]                = a0 * F[3] + mu * F[0];
    G[1]                = -a0 * F[2] + mu * F[1];
    G[2]                = -a0 * F[1] + mu * F[2];
    G[3]                = a0 * F[0] + mu * F[3];
    return G;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<2>::SMatrix<typename TMatrix::ScalarType, 4, 4>
StableNeoHookeanEnergy<2>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 4, 4> H;
    ScalarType const a0 = lambda * F[3];
    ScalarType const a1 = -a0 * F[2];
    ScalarType const a2 = -a0 * F[1];
    ScalarType const a3 = lambda * (F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda);
    ScalarType const a4 = a3 + lambda * F[0] * F[3];
    ScalarType const a5 = -a3 + lambda * F[1] * F[2];
    ScalarType const a6 = lambda * F[0];
    ScalarType const a7 = -a6 * F[2];
    ScalarType const a8 = -a6 * F[1];
    H[0]                = lambda * ((F[3]) * (F[3])) + mu;
    H[1]                = a1;
    H[2]                = a2;
    H[3]                = a4;
    H[4]                = a1;
    H[5]                = lambda * ((F[2]) * (F[2])) + mu;
    H[6]                = a5;
    H[7]                = a7;
    H[8]                = a2;
    H[9]                = a5;
    H[10]               = lambda * ((F[1]) * (F[1])) + mu;
    H[11]               = a8;
    H[12]               = a4;
    H[13]               = a7;
    H[14]               = a8;
    H[15]               = lambda * ((F[0]) * (F[0])) + mu;
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::evalWithGrad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF) const
{
    static_assert(
        TMatrixGF::kRows == 4 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 4x1");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda;
    ScalarType const a1 = a0 * lambda;
    psi                 = (1.0 / 2.0) * ((a0) * (a0)) * lambda +
          (1.0 / 2.0) * mu *
              (((F[0]) * (F[0])) + ((F[1]) * (F[1])) + ((F[2]) * (F[2])) + ((F[3]) * (F[3])) - 2);
    gF[0] = a1 * F[3] + mu * F[0];
    gF[1] = -a1 * F[2] + mu * F[1];
    gF[2] = -a1 * F[1] + mu * F[2];
    gF[3] = a1 * F[0] + mu * F[3];
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::evalWithGradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 4 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 4x1");
    static_assert(
        TMatrixHF::kRows == 4 and TMatrixHF::kCols == 4,
        "Hessian w.r.t. F must have dimensions 4x4");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0  = ((F[0]) * (F[0]));
    ScalarType const a1  = ((F[1]) * (F[1]));
    ScalarType const a2  = ((F[2]) * (F[2]));
    ScalarType const a3  = ((F[3]) * (F[3]));
    ScalarType const a4  = F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda;
    ScalarType const a5  = a4 * lambda;
    ScalarType const a6  = lambda * F[3];
    ScalarType const a7  = -a6 * F[2];
    ScalarType const a8  = -a6 * F[1];
    ScalarType const a9  = a5 + lambda * F[0] * F[3];
    ScalarType const a10 = -a5 + lambda * F[1] * F[2];
    ScalarType const a11 = lambda * F[0];
    ScalarType const a12 = -a11 * F[2];
    ScalarType const a13 = -a11 * F[1];
    psi    = (1.0 / 2.0) * ((a4) * (a4)) * lambda + (1.0 / 2.0) * mu * (a0 + a1 + a2 + a3 - 2);
    gF[0]  = a5 * F[3] + mu * F[0];
    gF[1]  = -a5 * F[2] + mu * F[1];
    gF[2]  = -a5 * F[1] + mu * F[2];
    gF[3]  = a5 * F[0] + mu * F[3];
    HF[0]  = a3 * lambda + mu;
    HF[1]  = a7;
    HF[2]  = a8;
    HF[3]  = a9;
    HF[4]  = a7;
    HF[5]  = a2 * lambda + mu;
    HF[6]  = a10;
    HF[7]  = a12;
    HF[8]  = a8;
    HF[9]  = a10;
    HF[10] = a1 * lambda + mu;
    HF[11] = a13;
    HF[12] = a9;
    HF[13] = a12;
    HF[14] = a13;
    HF[15] = a0 * lambda + mu;
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<2>::gradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 4 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 4x1");
    static_assert(
        TMatrixHF::kRows == 4 and TMatrixHF::kCols == 4,
        "Hessian w.r.t. F must have dimensions 4x4");
    using ScalarType    = typename TMatrix::ScalarType;
    ScalarType const a0 = lambda * (F[0] * F[3] - F[1] * F[2] - 1 - mu / lambda);
    ScalarType const a1 = lambda * F[3];
    ScalarType const a2 = -a1 * F[2];
    ScalarType const a3 = -a1 * F[1];
    ScalarType const a4 = a0 + lambda * F[0] * F[3];
    ScalarType const a5 = -a0 + lambda * F[1] * F[2];
    ScalarType const a6 = lambda * F[0];
    ScalarType const a7 = -a6 * F[2];
    ScalarType const a8 = -a6 * F[1];
    gF[0]               = a0 * F[3] + mu * F[0];
    gF[1]               = -a0 * F[2] + mu * F[1];
    gF[2]               = -a0 * F[1] + mu * F[2];
    gF[3]               = a0 * F[0] + mu * F[3];
    HF[0]               = lambda * ((F[3]) * (F[3])) + mu;
    HF[1]               = a2;
    HF[2]               = a3;
    HF[3]               = a4;
    HF[4]               = a2;
    HF[5]               = lambda * ((F[2]) * (F[2])) + mu;
    HF[6]               = a5;
    HF[7]               = a7;
    HF[8]               = a3;
    HF[9]               = a5;
    HF[10]              = lambda * ((F[1]) * (F[1])) + mu;
    HF[11]              = a8;
    HF[12]              = a4;
    HF[13]              = a7;
    HF[14]              = a8;
    HF[15]              = lambda * ((F[0]) * (F[0])) + mu;
}

/**
 * @brief Stable Neo-Hookean hyperelastic energy for 3D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct StableNeoHookeanEnergy<3>
{
  public:
    template <class TScalar, int M, int N>
    using SMatrix = pbat::math::linalg::mini::SMatrix<TScalar, M, N>; ///< Scalar matrix type

    template <class TScalar, int M>
    using SVector = pbat::math::linalg::mini::SVector<TScalar, M>; ///< Scalar vector type

    static auto constexpr kDims = 3; ///< Dimension of the space

    /**
     * @brief Evaluate the elastic energy
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 9>
    grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return ScalarType Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 9, 9>
    hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return ScalarType Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGrad(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF) const;

    /**
     * @brief Evaluate the elastic energy with its gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     * @return ScalarType Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType evalWithGradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;

    /**
     * @brief Evaluate the elastic energy gradient and hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @param HF Hessian w.r.t. F
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE void gradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    psi = (1.0 / 2.0) * lambda *
              ((F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] + F[1] * F[5] * F[6] +
                F[2] * F[3] * F[7] - F[2] * F[4] * F[6] - 1 - mu / lambda) *
               (F[0] * F[4] * F[8] - F[0] * F[5] * F[7] - F[1] * F[3] * F[8] + F[1] * F[5] * F[6] +
                F[2] * F[3] * F[7] - F[2] * F[4] * F[6] - 1 - mu / lambda)) +
          (1.0 / 2.0) * mu *
              (((F[0]) * (F[0])) + ((F[1]) * (F[1])) + ((F[2]) * (F[2])) + ((F[3]) * (F[3])) +
               ((F[4]) * (F[4])) + ((F[5]) * (F[5])) + ((F[6]) * (F[6])) + ((F[7]) * (F[7])) +
               ((F[8]) * (F[8])) - 3);
    return psi;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<3>::SVector<typename TMatrix::ScalarType, 9>
StableNeoHookeanEnergy<3>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 9> G;
    ScalarType const a0 = F[5] * F[7];
    ScalarType const a1 = F[3] * F[8];
    ScalarType const a2 = F[4] * F[6];
    ScalarType const a3 = (1.0 / 2.0) * lambda *
                          (-a0 * F[0] - a1 * F[1] - a2 * F[2] + F[0] * F[4] * F[8] +
                           F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - 1 - mu / lambda);
    ScalarType const a4 = 2 * F[8];
    ScalarType const a5 = 2 * F[2];
    ScalarType const a6 = 2 * F[0];
    ScalarType const a7 = 2 * F[1];
    G[0]                = a3 * (-2 * a0 + 2 * F[4] * F[8]) + mu * F[0];
    G[1]                = a3 * (-2 * a1 + 2 * F[5] * F[6]) + mu * F[1];
    G[2]                = a3 * (-2 * a2 + 2 * F[3] * F[7]) + mu * F[2];
    G[3]                = a3 * (-a4 * F[1] + 2 * F[2] * F[7]) + mu * F[3];
    G[4]                = a3 * (a4 * F[0] - a5 * F[6]) + mu * F[4];
    G[5]                = a3 * (-a6 * F[7] + 2 * F[1] * F[6]) + mu * F[5];
    G[6]                = a3 * (-a5 * F[4] + a7 * F[5]) + mu * F[6];
    G[7]                = a3 * (-a6 * F[5] + 2 * F[2] * F[3]) + mu * F[7];
    G[8]                = a3 * (a6 * F[4] - a7 * F[3]) + mu * F[8];
    return G;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return PBAT_HOST_DEVICE
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<3>::SMatrix<typename TMatrix::ScalarType, 9, 9>
StableNeoHookeanEnergy<3>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 9, 9> H;
    ScalarType const a0  = F[4] * F[8];
    ScalarType const a1  = F[5] * F[7];
    ScalarType const a2  = a0 - a1;
    ScalarType const a3  = (1.0 / 2.0) * lambda;
    ScalarType const a4  = a3 * (2 * a0 - 2 * a1);
    ScalarType const a5  = F[3] * F[8];
    ScalarType const a6  = -a5 + F[5] * F[6];
    ScalarType const a7  = F[3] * F[7];
    ScalarType const a8  = F[4] * F[6];
    ScalarType const a9  = a7 - a8;
    ScalarType const a10 = F[1] * F[8];
    ScalarType const a11 = -a10 + F[2] * F[7];
    ScalarType const a12 = F[0] * F[8];
    ScalarType const a13 = F[2] * F[6];
    ScalarType const a14 = a12 - a13;
    ScalarType const a15 = lambda * (-a1 * F[0] - a5 * F[1] - a8 * F[2] + F[0] * F[4] * F[8] +
                                     F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - 1 - mu / lambda);
    ScalarType const a16 = a15 * F[8];
    ScalarType const a17 = F[0] * F[7];
    ScalarType const a18 = -a17 + F[1] * F[6];
    ScalarType const a19 = a15 * F[7];
    ScalarType const a20 = -a19;
    ScalarType const a21 = F[1] * F[5];
    ScalarType const a22 = F[2] * F[4];
    ScalarType const a23 = a21 - a22;
    ScalarType const a24 = F[0] * F[5];
    ScalarType const a25 = -a24 + F[2] * F[3];
    ScalarType const a26 = a15 * F[5];
    ScalarType const a27 = -a26;
    ScalarType const a28 = F[0] * F[4];
    ScalarType const a29 = F[1] * F[3];
    ScalarType const a30 = a28 - a29;
    ScalarType const a31 = a15 * F[4];
    ScalarType const a32 = a3 * (-2 * a5 + 2 * F[5] * F[6]);
    ScalarType const a33 = -a16;
    ScalarType const a34 = a15 * F[6];
    ScalarType const a35 = a15 * F[3];
    ScalarType const a36 = -a35;
    ScalarType const a37 = a3 * (2 * a7 - 2 * a8);
    ScalarType const a38 = -a34;
    ScalarType const a39 = -a31;
    ScalarType const a40 = a3 * (-2 * a10 + 2 * F[2] * F[7]);
    ScalarType const a41 = a15 * F[2];
    ScalarType const a42 = a15 * F[1];
    ScalarType const a43 = -a42;
    ScalarType const a44 = a3 * (2 * a12 - 2 * a13);
    ScalarType const a45 = -a41;
    ScalarType const a46 = a15 * F[0];
    ScalarType const a47 = a3 * (-2 * a17 + 2 * F[1] * F[6]);
    ScalarType const a48 = -a46;
    ScalarType const a49 = a3 * (2 * a21 - 2 * a22);
    ScalarType const a50 = a3 * (-2 * a24 + 2 * F[2] * F[3]);
    ScalarType const a51 = a3 * (2 * a28 - 2 * a29);
    H[0]                 = a2 * a4 + mu;
    H[1]                 = a4 * a6;
    H[2]                 = a4 * a9;
    H[3]                 = a11 * a4;
    H[4]                 = a14 * a4 + a16;
    H[5]                 = a18 * a4 + a20;
    H[6]                 = a23 * a4;
    H[7]                 = a25 * a4 + a27;
    H[8]                 = a30 * a4 + a31;
    H[9]                 = a2 * a32;
    H[10]                = a32 * a6 + mu;
    H[11]                = a32 * a9;
    H[12]                = a11 * a32 + a33;
    H[13]                = a14 * a32;
    H[14]                = a18 * a32 + a34;
    H[15]                = a23 * a32 + a26;
    H[16]                = a25 * a32;
    H[17]                = a30 * a32 + a36;
    H[18]                = a2 * a37;
    H[19]                = a37 * a6;
    H[20]                = a37 * a9 + mu;
    H[21]                = a11 * a37 + a19;
    H[22]                = a14 * a37 + a38;
    H[23]                = a18 * a37;
    H[24]                = a23 * a37 + a39;
    H[25]                = a25 * a37 + a35;
    H[26]                = a30 * a37;
    H[27]                = a2 * a40;
    H[28]                = a33 + a40 * a6;
    H[29]                = a19 + a40 * a9;
    H[30]                = a11 * a40 + mu;
    H[31]                = a14 * a40;
    H[32]                = a18 * a40;
    H[33]                = a23 * a40;
    H[34]                = a25 * a40 + a41;
    H[35]                = a30 * a40 + a43;
    H[36]                = a16 + a2 * a44;
    H[37]                = a44 * a6;
    H[38]                = a38 + a44 * a9;
    H[39]                = a11 * a44;
    H[40]                = a14 * a44 + mu;
    H[41]                = a18 * a44;
    H[42]                = a23 * a44 + a45;
    H[43]                = a25 * a44;
    H[44]                = a30 * a44 + a46;
    H[45]                = a2 * a47 + a20;
    H[46]                = a34 + a47 * a6;
    H[47]                = a47 * a9;
    H[48]                = a11 * a47;
    H[49]                = a14 * a47;
    H[50]                = a18 * a47 + mu;
    H[51]                = a23 * a47 + a42;
    H[52]                = a25 * a47 + a48;
    H[53]                = a30 * a47;
    H[54]                = a2 * a49;
    H[55]                = a26 + a49 * a6;
    H[56]                = a39 + a49 * a9;
    H[57]                = a11 * a49;
    H[58]                = a14 * a49 + a45;
    H[59]                = a18 * a49 + a42;
    H[60]                = a23 * a49 + mu;
    H[61]                = a25 * a49;
    H[62]                = a30 * a49;
    H[63]                = a2 * a50 + a27;
    H[64]                = a50 * a6;
    H[65]                = a35 + a50 * a9;
    H[66]                = a11 * a50 + a41;
    H[67]                = a14 * a50;
    H[68]                = a18 * a50 + a48;
    H[69]                = a23 * a50;
    H[70]                = a25 * a50 + mu;
    H[71]                = a30 * a50;
    H[72]                = a2 * a51 + a31;
    H[73]                = a36 + a51 * a6;
    H[74]                = a51 * a9;
    H[75]                = a11 * a51 + a43;
    H[76]                = a14 * a51 + a46;
    H[77]                = a18 * a51;
    H[78]                = a23 * a51;
    H[79]                = a25 * a51;
    H[80]                = a30 * a51 + mu;
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::evalWithGrad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF) const
{
    static_assert(
        TMatrixGF::kRows == 9 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 9x1");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = F[5] * F[7];
    ScalarType const a1 = F[3] * F[8];
    ScalarType const a2 = F[4] * F[6];
    ScalarType const a3 = -a0 * F[0] - a1 * F[1] - a2 * F[2] + F[0] * F[4] * F[8] +
                          F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - 1 - mu / lambda;
    ScalarType const a4 = (1.0 / 2.0) * lambda;
    ScalarType const a5 = a3 * a4;
    ScalarType const a6 = 2 * F[8];
    ScalarType const a7 = 2 * F[2];
    ScalarType const a8 = 2 * F[0];
    ScalarType const a9 = 2 * F[1];
    psi                 = ((a3) * (a3)) * a4 + (1.0 / 2.0) * mu *
                                   (((F[0]) * (F[0])) + ((F[1]) * (F[1])) + ((F[2]) * (F[2])) +
                                    ((F[3]) * (F[3])) + ((F[4]) * (F[4])) + ((F[5]) * (F[5])) +
                                    ((F[6]) * (F[6])) + ((F[7]) * (F[7])) + ((F[8]) * (F[8])) - 3);
    gF[0] = a5 * (-2 * a0 + 2 * F[4] * F[8]) + mu * F[0];
    gF[1] = a5 * (-2 * a1 + 2 * F[5] * F[6]) + mu * F[1];
    gF[2] = a5 * (-2 * a2 + 2 * F[3] * F[7]) + mu * F[2];
    gF[3] = a5 * (-a6 * F[1] + 2 * F[2] * F[7]) + mu * F[3];
    gF[4] = a5 * (a6 * F[0] - a7 * F[6]) + mu * F[4];
    gF[5] = a5 * (-a8 * F[7] + 2 * F[1] * F[6]) + mu * F[5];
    gF[6] = a5 * (-a7 * F[4] + a9 * F[5]) + mu * F[6];
    gF[7] = a5 * (-a8 * F[5] + 2 * F[2] * F[3]) + mu * F[7];
    gF[8] = a5 * (a8 * F[4] - a9 * F[3]) + mu * F[8];
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::evalWithGradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 9 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 9x1");
    static_assert(
        TMatrixHF::kRows == 9 and TMatrixHF::kCols == 9,
        "Hessian w.r.t. F must have dimensions 9x9");
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = F[5] * F[7];
    ScalarType const a1 = F[3] * F[8];
    ScalarType const a2 = F[4] * F[6];
    ScalarType const a3 = -a0 * F[0] - a1 * F[1] - a2 * F[2] + F[0] * F[4] * F[8] +
                          F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - 1 - mu / lambda;
    ScalarType const a4  = (1.0 / 2.0) * lambda;
    ScalarType const a5  = F[4] * F[8];
    ScalarType const a6  = -2 * a0 + 2 * a5;
    ScalarType const a7  = a3 * a4;
    ScalarType const a8  = -2 * a1 + 2 * F[5] * F[6];
    ScalarType const a9  = F[3] * F[7];
    ScalarType const a10 = -2 * a2 + 2 * a9;
    ScalarType const a11 = F[1] * F[8];
    ScalarType const a12 = -2 * a11 + 2 * F[2] * F[7];
    ScalarType const a13 = F[0] * F[8];
    ScalarType const a14 = F[2] * F[6];
    ScalarType const a15 = 2 * a13 - 2 * a14;
    ScalarType const a16 = F[0] * F[7];
    ScalarType const a17 = -2 * a16 + 2 * F[1] * F[6];
    ScalarType const a18 = F[1] * F[5];
    ScalarType const a19 = F[2] * F[4];
    ScalarType const a20 = 2 * a18 - 2 * a19;
    ScalarType const a21 = F[0] * F[5];
    ScalarType const a22 = -2 * a21 + 2 * F[2] * F[3];
    ScalarType const a23 = F[0] * F[4];
    ScalarType const a24 = F[1] * F[3];
    ScalarType const a25 = 2 * a23 - 2 * a24;
    ScalarType const a26 = a4 * (-a0 + a5);
    ScalarType const a27 = a3 * lambda;
    ScalarType const a28 = a27 * F[8];
    ScalarType const a29 = a27 * F[7];
    ScalarType const a30 = -a29;
    ScalarType const a31 = a27 * F[5];
    ScalarType const a32 = -a31;
    ScalarType const a33 = a27 * F[4];
    ScalarType const a34 = a4 * (-a1 + F[5] * F[6]);
    ScalarType const a35 = -a28;
    ScalarType const a36 = a27 * F[6];
    ScalarType const a37 = a27 * F[3];
    ScalarType const a38 = -a37;
    ScalarType const a39 = a4 * (-a2 + a9);
    ScalarType const a40 = -a36;
    ScalarType const a41 = -a33;
    ScalarType const a42 = a4 * (-a11 + F[2] * F[7]);
    ScalarType const a43 = a27 * F[2];
    ScalarType const a44 = a27 * F[1];
    ScalarType const a45 = -a44;
    ScalarType const a46 = a4 * (a13 - a14);
    ScalarType const a47 = -a43;
    ScalarType const a48 = a27 * F[0];
    ScalarType const a49 = a4 * (-a16 + F[1] * F[6]);
    ScalarType const a50 = -a48;
    ScalarType const a51 = a4 * (a18 - a19);
    ScalarType const a52 = a4 * (-a21 + F[2] * F[3]);
    ScalarType const a53 = a4 * (a23 - a24);
    psi                  = ((a3) * (a3)) * a4 + (1.0 / 2.0) * mu *
                                   (((F[0]) * (F[0])) + ((F[1]) * (F[1])) + ((F[2]) * (F[2])) +
                                    ((F[3]) * (F[3])) + ((F[4]) * (F[4])) + ((F[5]) * (F[5])) +
                                    ((F[6]) * (F[6])) + ((F[7]) * (F[7])) + ((F[8]) * (F[8])) - 3);
    gF[0]  = a6 * a7 + mu * F[0];
    gF[1]  = a7 * a8 + mu * F[1];
    gF[2]  = a10 * a7 + mu * F[2];
    gF[3]  = a12 * a7 + mu * F[3];
    gF[4]  = a15 * a7 + mu * F[4];
    gF[5]  = a17 * a7 + mu * F[5];
    gF[6]  = a20 * a7 + mu * F[6];
    gF[7]  = a22 * a7 + mu * F[7];
    gF[8]  = a25 * a7 + mu * F[8];
    HF[0]  = a26 * a6 + mu;
    HF[1]  = a26 * a8;
    HF[2]  = a10 * a26;
    HF[3]  = a12 * a26;
    HF[4]  = a15 * a26 + a28;
    HF[5]  = a17 * a26 + a30;
    HF[6]  = a20 * a26;
    HF[7]  = a22 * a26 + a32;
    HF[8]  = a25 * a26 + a33;
    HF[9]  = a34 * a6;
    HF[10] = a34 * a8 + mu;
    HF[11] = a10 * a34;
    HF[12] = a12 * a34 + a35;
    HF[13] = a15 * a34;
    HF[14] = a17 * a34 + a36;
    HF[15] = a20 * a34 + a31;
    HF[16] = a22 * a34;
    HF[17] = a25 * a34 + a38;
    HF[18] = a39 * a6;
    HF[19] = a39 * a8;
    HF[20] = a10 * a39 + mu;
    HF[21] = a12 * a39 + a29;
    HF[22] = a15 * a39 + a40;
    HF[23] = a17 * a39;
    HF[24] = a20 * a39 + a41;
    HF[25] = a22 * a39 + a37;
    HF[26] = a25 * a39;
    HF[27] = a42 * a6;
    HF[28] = a35 + a42 * a8;
    HF[29] = a10 * a42 + a29;
    HF[30] = a12 * a42 + mu;
    HF[31] = a15 * a42;
    HF[32] = a17 * a42;
    HF[33] = a20 * a42;
    HF[34] = a22 * a42 + a43;
    HF[35] = a25 * a42 + a45;
    HF[36] = a28 + a46 * a6;
    HF[37] = a46 * a8;
    HF[38] = a10 * a46 + a40;
    HF[39] = a12 * a46;
    HF[40] = a15 * a46 + mu;
    HF[41] = a17 * a46;
    HF[42] = a20 * a46 + a47;
    HF[43] = a22 * a46;
    HF[44] = a25 * a46 + a48;
    HF[45] = a30 + a49 * a6;
    HF[46] = a36 + a49 * a8;
    HF[47] = a10 * a49;
    HF[48] = a12 * a49;
    HF[49] = a15 * a49;
    HF[50] = a17 * a49 + mu;
    HF[51] = a20 * a49 + a44;
    HF[52] = a22 * a49 + a50;
    HF[53] = a25 * a49;
    HF[54] = a51 * a6;
    HF[55] = a31 + a51 * a8;
    HF[56] = a10 * a51 + a41;
    HF[57] = a12 * a51;
    HF[58] = a15 * a51 + a47;
    HF[59] = a17 * a51 + a44;
    HF[60] = a20 * a51 + mu;
    HF[61] = a22 * a51;
    HF[62] = a25 * a51;
    HF[63] = a32 + a52 * a6;
    HF[64] = a52 * a8;
    HF[65] = a10 * a52 + a37;
    HF[66] = a12 * a52 + a43;
    HF[67] = a15 * a52;
    HF[68] = a17 * a52 + a50;
    HF[69] = a20 * a52;
    HF[70] = a22 * a52 + mu;
    HF[71] = a25 * a52;
    HF[72] = a33 + a53 * a6;
    HF[73] = a38 + a53 * a8;
    HF[74] = a10 * a53;
    HF[75] = a12 * a53 + a45;
    HF[76] = a15 * a53 + a48;
    HF[77] = a17 * a53;
    HF[78] = a20 * a53;
    HF[79] = a22 * a53;
    HF[80] = a25 * a53 + mu;
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<3>::gradAndHessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF,
    TMatrixHF& HF) const
{
    static_assert(
        TMatrixGF::kRows == 9 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 9x1");
    static_assert(
        TMatrixHF::kRows == 9 and TMatrixHF::kCols == 9,
        "Hessian w.r.t. F must have dimensions 9x9");
    using ScalarType     = typename TMatrix::ScalarType;
    ScalarType const a0  = F[4] * F[8];
    ScalarType const a1  = F[5] * F[7];
    ScalarType const a2  = 2 * a0 - 2 * a1;
    ScalarType const a3  = F[3] * F[8];
    ScalarType const a4  = F[4] * F[6];
    ScalarType const a5  = lambda * (-a1 * F[0] - a3 * F[1] - a4 * F[2] + F[0] * F[4] * F[8] +
                                    F[1] * F[5] * F[6] + F[2] * F[3] * F[7] - 1 - mu / lambda);
    ScalarType const a6  = (1.0 / 2.0) * a5;
    ScalarType const a7  = -2 * a3 + 2 * F[5] * F[6];
    ScalarType const a8  = F[3] * F[7];
    ScalarType const a9  = -2 * a4 + 2 * a8;
    ScalarType const a10 = F[1] * F[8];
    ScalarType const a11 = -2 * a10 + 2 * F[2] * F[7];
    ScalarType const a12 = F[0] * F[8];
    ScalarType const a13 = F[2] * F[6];
    ScalarType const a14 = 2 * a12 - 2 * a13;
    ScalarType const a15 = F[0] * F[7];
    ScalarType const a16 = -2 * a15 + 2 * F[1] * F[6];
    ScalarType const a17 = F[1] * F[5];
    ScalarType const a18 = F[2] * F[4];
    ScalarType const a19 = 2 * a17 - 2 * a18;
    ScalarType const a20 = F[0] * F[5];
    ScalarType const a21 = -2 * a20 + 2 * F[2] * F[3];
    ScalarType const a22 = F[0] * F[4];
    ScalarType const a23 = F[1] * F[3];
    ScalarType const a24 = 2 * a22 - 2 * a23;
    ScalarType const a25 = (1.0 / 2.0) * lambda;
    ScalarType const a26 = a25 * (a0 - a1);
    ScalarType const a27 = a5 * F[8];
    ScalarType const a28 = a5 * F[7];
    ScalarType const a29 = -a28;
    ScalarType const a30 = a5 * F[5];
    ScalarType const a31 = -a30;
    ScalarType const a32 = a5 * F[4];
    ScalarType const a33 = a25 * (-a3 + F[5] * F[6]);
    ScalarType const a34 = -a27;
    ScalarType const a35 = a5 * F[6];
    ScalarType const a36 = a5 * F[3];
    ScalarType const a37 = -a36;
    ScalarType const a38 = a25 * (-a4 + a8);
    ScalarType const a39 = -a35;
    ScalarType const a40 = -a32;
    ScalarType const a41 = a25 * (-a10 + F[2] * F[7]);
    ScalarType const a42 = a5 * F[2];
    ScalarType const a43 = a5 * F[1];
    ScalarType const a44 = -a43;
    ScalarType const a45 = a25 * (a12 - a13);
    ScalarType const a46 = -a42;
    ScalarType const a47 = a5 * F[0];
    ScalarType const a48 = a25 * (-a15 + F[1] * F[6]);
    ScalarType const a49 = -a47;
    ScalarType const a50 = a25 * (a17 - a18);
    ScalarType const a51 = a25 * (-a20 + F[2] * F[3]);
    ScalarType const a52 = a25 * (a22 - a23);
    gF[0]                = a2 * a6 + mu * F[0];
    gF[1]                = a6 * a7 + mu * F[1];
    gF[2]                = a6 * a9 + mu * F[2];
    gF[3]                = a11 * a6 + mu * F[3];
    gF[4]                = a14 * a6 + mu * F[4];
    gF[5]                = a16 * a6 + mu * F[5];
    gF[6]                = a19 * a6 + mu * F[6];
    gF[7]                = a21 * a6 + mu * F[7];
    gF[8]                = a24 * a6 + mu * F[8];
    HF[0]                = a2 * a26 + mu;
    HF[1]                = a26 * a7;
    HF[2]                = a26 * a9;
    HF[3]                = a11 * a26;
    HF[4]                = a14 * a26 + a27;
    HF[5]                = a16 * a26 + a29;
    HF[6]                = a19 * a26;
    HF[7]                = a21 * a26 + a31;
    HF[8]                = a24 * a26 + a32;
    HF[9]                = a2 * a33;
    HF[10]               = a33 * a7 + mu;
    HF[11]               = a33 * a9;
    HF[12]               = a11 * a33 + a34;
    HF[13]               = a14 * a33;
    HF[14]               = a16 * a33 + a35;
    HF[15]               = a19 * a33 + a30;
    HF[16]               = a21 * a33;
    HF[17]               = a24 * a33 + a37;
    HF[18]               = a2 * a38;
    HF[19]               = a38 * a7;
    HF[20]               = a38 * a9 + mu;
    HF[21]               = a11 * a38 + a28;
    HF[22]               = a14 * a38 + a39;
    HF[23]               = a16 * a38;
    HF[24]               = a19 * a38 + a40;
    HF[25]               = a21 * a38 + a36;
    HF[26]               = a24 * a38;
    HF[27]               = a2 * a41;
    HF[28]               = a34 + a41 * a7;
    HF[29]               = a28 + a41 * a9;
    HF[30]               = a11 * a41 + mu;
    HF[31]               = a14 * a41;
    HF[32]               = a16 * a41;
    HF[33]               = a19 * a41;
    HF[34]               = a21 * a41 + a42;
    HF[35]               = a24 * a41 + a44;
    HF[36]               = a2 * a45 + a27;
    HF[37]               = a45 * a7;
    HF[38]               = a39 + a45 * a9;
    HF[39]               = a11 * a45;
    HF[40]               = a14 * a45 + mu;
    HF[41]               = a16 * a45;
    HF[42]               = a19 * a45 + a46;
    HF[43]               = a21 * a45;
    HF[44]               = a24 * a45 + a47;
    HF[45]               = a2 * a48 + a29;
    HF[46]               = a35 + a48 * a7;
    HF[47]               = a48 * a9;
    HF[48]               = a11 * a48;
    HF[49]               = a14 * a48;
    HF[50]               = a16 * a48 + mu;
    HF[51]               = a19 * a48 + a43;
    HF[52]               = a21 * a48 + a49;
    HF[53]               = a24 * a48;
    HF[54]               = a2 * a50;
    HF[55]               = a30 + a50 * a7;
    HF[56]               = a40 + a50 * a9;
    HF[57]               = a11 * a50;
    HF[58]               = a14 * a50 + a46;
    HF[59]               = a16 * a50 + a43;
    HF[60]               = a19 * a50 + mu;
    HF[61]               = a21 * a50;
    HF[62]               = a24 * a50;
    HF[63]               = a2 * a51 + a31;
    HF[64]               = a51 * a7;
    HF[65]               = a36 + a51 * a9;
    HF[66]               = a11 * a51 + a42;
    HF[67]               = a14 * a51;
    HF[68]               = a16 * a51 + a49;
    HF[69]               = a19 * a51;
    HF[70]               = a21 * a51 + mu;
    HF[71]               = a24 * a51;
    HF[72]               = a2 * a52 + a32;
    HF[73]               = a37 + a52 * a7;
    HF[74]               = a52 * a9;
    HF[75]               = a11 * a52 + a44;
    HF[76]               = a14 * a52 + a47;
    HF[77]               = a16 * a52;
    HF[78]               = a19 * a52;
    HF[79]               = a21 * a52;
    HF[80]               = a24 * a52 + mu;
}

} // namespace physics
} // namespace pbat

#endif // PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
