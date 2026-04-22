/**
 * @file StableNeoHookeanEnergy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Stable Neo-Hookean \cite smith2018snh hyperelastic energy
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
#define PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Determinant.h"
#include "pbat/math/linalg/mini/Flatten.h"
#include "pbat/math/linalg/mini/Geometry.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/linalg/mini/Reductions.h"
#include "pbat/math/linalg/mini/Reshape.h"

#include <cmath>

// clang-format off
#include "pbat/warning/Push.h"
#include "pbat/warning/FloatConversion.h"
// clang-format on

namespace pbat {
namespace physics {

template <int Dims>
struct StableNeoHookeanEnergy;

/**
 * @brief Stable Neo-Hookean hyperelastic energy for 1D
 *
 * @tparam Dims Dimension of the space
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
     * @return Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    Eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 1>
    Grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 1, 1>
    Hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGrad(
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
     * @return Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGradAndHessian(
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
    PBAT_HOST_DEVICE void GradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::Eval(
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
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<1>::SVector<typename TMatrix::ScalarType, 1>
StableNeoHookeanEnergy<1>::Grad(
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
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<1>::SMatrix<typename TMatrix::ScalarType, 1, 1>
StableNeoHookeanEnergy<1>::Hessian(
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::EvalWithGrad(
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<1>::EvalWithGradAndHessian(
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
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<1>::GradAndHessian(
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
    gF[0] = (1.0 / 2.0) * lambda * (2 * F[0] - 2 - 2 * mu / lambda) + mu * F[0];
    HF[0] = lambda + mu;
}

/**
 * @brief Stable Neo-Hookean hyperelastic energy for 2D
 *
 * @tparam Dims Dimension of the space
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
     * @return Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    Eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 4>
    Grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 4, 4>
    Hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGrad(
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
     * @return Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGradAndHessian(
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
    PBAT_HOST_DEVICE void GradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::Eval(
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
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<2>::SVector<typename TMatrix::ScalarType, 4>
StableNeoHookeanEnergy<2>::Grad(
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
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<2>::SMatrix<typename TMatrix::ScalarType, 4, 4>
StableNeoHookeanEnergy<2>::Hessian(
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::EvalWithGrad(
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
    psi   = (1.0 / 2.0) * ((a0) * (a0)) * lambda +
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<2>::EvalWithGradAndHessian(
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
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<2>::GradAndHessian(
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
     * @return Energy
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType
    Eval(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy gradient
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SVector<typename TMatrix::ScalarType, 9>
    Grad(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy hessian
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @return Energy hessian
     */
    template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
    PBAT_HOST_DEVICE SMatrix<typename TMatrix::ScalarType, 9, 9>
    Hessian(TMatrix const& F, typename TMatrix::ScalarType mu, typename TMatrix::ScalarType lambda)
        const;

    /**
     * @brief Evaluate the elastic energy and its gradient
     *
     * @tparam TMatrix Matrix type
     * @param F Deformation gradient
     * @param mu First Lame coefficient
     * @param lambda Second Lame coefficient
     * @param gF Gradient w.r.t. F
     * @return Energy and its gradient
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGrad(
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
     * @return Energy and its gradient and hessian
     */
    template <
        math::linalg::mini::CReadableVectorizedMatrix TMatrix,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
        math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
    PBAT_HOST_DEVICE typename TMatrix::ScalarType EvalWithGradAndHessian(
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
    PBAT_HOST_DEVICE void GradAndHessian(
        TMatrix const& F,
        typename TMatrix::ScalarType mu,
        typename TMatrix::ScalarType lambda,
        TMatrixGF& gF,
        TMatrixHF& HF) const;
};

template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::Eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType  = typename TMatrix::ScalarType;
    ScalarType I3     = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I2     = Dot(F, F);
    ScalarType I3min1 = I3 - 1;
    ScalarType psi =
        ScalarType(0.5) * mu * (I2 - 3) - mu * I3min1 + ScalarType(0.5) * lambda * I3min1 * I3min1;
    return psi;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<3>::SVector<typename TMatrix::ScalarType, 9>
StableNeoHookeanEnergy<3>::Grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 9> G;
    ScalarType I3         = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I2         = Dot(F, F);
    ScalarType I3minAlpha = I3 - 1 - mu / lambda;
    SMatrix<ScalarType, 3, 3> Fcross;
    Fcross.Col(0) = Cross(F.Col(1), F.Col(2));
    Fcross.Col(1) = Cross(F.Col(2), F.Col(0));
    Fcross.Col(2) = Cross(F.Col(0), F.Col(1));
    G             = mu * Flatten(F) + lambda * I3minAlpha * Flatten(Fcross);
    return G;
}

/**
 * @brief
 *
 * @tparam TMatrix
 * @param F
 * @param mu
 * @param lambda
 * @return
 */
template <math::linalg::mini::CReadableVectorizedMatrix TMatrix>
PBAT_HOST_DEVICE StableNeoHookeanEnergy<3>::SMatrix<typename TMatrix::ScalarType, 9, 9>
StableNeoHookeanEnergy<3>::Hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 9, 9> H;
    ScalarType I3         = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I3minAlpha = I3 - 1 - mu / lambda;
    SMatrix<ScalarType, 3, 3> Fcross;
    auto f0       = F.Col(0);
    auto f1       = F.Col(1);
    auto f2       = F.Col(2);
    Fcross.Col(0) = Cross(f1, f2);
    Fcross.Col(1) = Cross(f2, f0);
    Fcross.Col(2) = Cross(f0, f1);
    auto H00      = H.template Slice<3, 3>(0, 0);
    auto H10      = H.template Slice<3, 3>(3, 0);
    auto H20      = H.template Slice<3, 3>(6, 0);
    auto H01      = H.template Slice<3, 3>(0, 3);
    auto H11      = H.template Slice<3, 3>(3, 3);
    auto H21      = H.template Slice<3, 3>(6, 3);
    auto H02      = H.template Slice<3, 3>(0, 6);
    auto H12      = H.template Slice<3, 3>(3, 6);
    auto H22      = H.template Slice<3, 3>(6, 6);
    using math::linalg::mini::Zeros;
    H00 = Zeros<ScalarType, 3, 3>();
    H11 = Zeros<ScalarType, 3, 3>();
    H22 = Zeros<ScalarType, 3, 3>();
    ToSkewSymmetricMatrix(f2, H10);
    ToSkewSymmetricMatrix(f0, H21);
    ToSkewSymmetricMatrix(f1, H02);
    H10 *= lambda * I3minAlpha;
    H21 *= lambda * I3minAlpha;
    H02 *= lambda * I3minAlpha;
    H01 = -H10;
    H12 = -H21;
    H20 = -H02;
    H += lambda * Flatten(Fcross) * Flatten(Fcross).Transpose();
    using math::linalg::mini::Ones;
    Diag(H) += mu * Ones<ScalarType, 9, 1>();
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::EvalWithGrad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda,
    TMatrixGF& gF) const
{
    static_assert(
        TMatrixGF::kRows == 9 and TMatrixGF::kCols == 1,
        "Grad w.r.t. F must have dimensions 9x1");
    using ScalarType      = typename TMatrix::ScalarType;
    ScalarType I3         = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I2         = Dot(F, F);
    ScalarType I3min1     = I3 - 1;
    ScalarType I3minAlpha = I3min1 - mu / lambda;
    ScalarType psi =
        ScalarType(0.5) * mu * (I2 - 3) - mu * I3min1 + ScalarType(0.5) * lambda * I3min1 * I3min1;
    SMatrix<ScalarType, 3, 3> Fcross;
    Fcross.Col(0) = Cross(F.Col(1), F.Col(2));
    Fcross.Col(1) = Cross(F.Col(2), F.Col(0));
    Fcross.Col(2) = Cross(F.Col(0), F.Col(1));
    gF            = mu * Flatten(F) + lambda * I3minAlpha * Flatten(Fcross);
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType StableNeoHookeanEnergy<3>::EvalWithGradAndHessian(
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
    using ScalarType      = typename TMatrix::ScalarType;
    ScalarType I3         = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I2         = Dot(F, F);
    ScalarType I3min1     = I3 - 1;
    ScalarType I3minAlpha = I3min1 - mu / lambda;
    ScalarType psi =
        ScalarType(0.5) * mu * (I2 - 3) - mu * I3min1 + ScalarType(0.5) * lambda * I3min1 * I3min1;
    SMatrix<ScalarType, 3, 3> Fcross;
    auto f0       = F.Col(0);
    auto f1       = F.Col(1);
    auto f2       = F.Col(2);
    Fcross.Col(0) = Cross(f1, f2);
    Fcross.Col(1) = Cross(f2, f0);
    Fcross.Col(2) = Cross(f0, f1);
    gF            = mu * Flatten(F) + lambda * I3minAlpha * Flatten(Fcross);
    auto H00      = HF.template Slice<3, 3>(0, 0);
    auto H10      = HF.template Slice<3, 3>(3, 0);
    auto H20      = HF.template Slice<3, 3>(6, 0);
    auto H01      = HF.template Slice<3, 3>(0, 3);
    auto H11      = HF.template Slice<3, 3>(3, 3);
    auto H21      = HF.template Slice<3, 3>(6, 3);
    auto H02      = HF.template Slice<3, 3>(0, 6);
    auto H12      = HF.template Slice<3, 3>(3, 6);
    auto H22      = HF.template Slice<3, 3>(6, 6);
    using math::linalg::mini::Zeros;
    H00 = Zeros<ScalarType, 3, 3>();
    H11 = Zeros<ScalarType, 3, 3>();
    H22 = Zeros<ScalarType, 3, 3>();
    ToSkewSymmetricMatrix(f2, H10);
    ToSkewSymmetricMatrix(f0, H21);
    ToSkewSymmetricMatrix(f1, H02);
    H10 *= lambda * I3minAlpha;
    H21 *= lambda * I3minAlpha;
    H02 *= lambda * I3minAlpha;
    H01 = -H10;
    H12 = -H21;
    H20 = -H02;
    HF += lambda * Flatten(Fcross) * Flatten(Fcross).Transpose();
    using math::linalg::mini::Ones;
    Diag(HF) += mu * Ones<ScalarType, 9, 1>();
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void StableNeoHookeanEnergy<3>::GradAndHessian(
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
    using ScalarType      = typename TMatrix::ScalarType;
    ScalarType I3         = TMatrix::kRows == 3 ? Determinant(F) : Determinant(Reshape<3, 3>(F));
    ScalarType I3min1     = I3 - 1;
    ScalarType I3minAlpha = I3min1 - mu / lambda;
    SMatrix<ScalarType, 3, 3> Fcross;
    auto f0       = F.Col(0);
    auto f1       = F.Col(1);
    auto f2       = F.Col(2);
    Fcross.Col(0) = Cross(f1, f2);
    Fcross.Col(1) = Cross(f2, f0);
    Fcross.Col(2) = Cross(f0, f1);
    gF            = mu * Flatten(F) + lambda * I3minAlpha * Flatten(Fcross);
    auto H00      = HF.template Slice<3, 3>(0, 0);
    auto H10      = HF.template Slice<3, 3>(3, 0);
    auto H20      = HF.template Slice<3, 3>(6, 0);
    auto H01      = HF.template Slice<3, 3>(0, 3);
    auto H11      = HF.template Slice<3, 3>(3, 3);
    auto H21      = HF.template Slice<3, 3>(6, 3);
    auto H02      = HF.template Slice<3, 3>(0, 6);
    auto H12      = HF.template Slice<3, 3>(3, 6);
    auto H22      = HF.template Slice<3, 3>(6, 6);
    using math::linalg::mini::Zeros;
    H00 = Zeros<ScalarType, 3, 3>();
    H11 = Zeros<ScalarType, 3, 3>();
    H22 = Zeros<ScalarType, 3, 3>();
    ToSkewSymmetricMatrix(f2, H10);
    ToSkewSymmetricMatrix(f0, H21);
    ToSkewSymmetricMatrix(f1, H02);
    H10 *= lambda * I3minAlpha;
    H21 *= lambda * I3minAlpha;
    H02 *= lambda * I3minAlpha;
    H01 = -H10;
    H12 = -H21;
    H20 = -H02;
    HF += lambda * Flatten(Fcross) * Flatten(Fcross).Transpose();
    using math::linalg::mini::Ones;
    Diag(HF) += mu * Ones<ScalarType, 9, 1>();
}

} // namespace physics
} // namespace pbat

#include "pbat/warning/Pop.h"

#endif // PBAT_PHYSICS_STABLENEOHOOKEANENERGY_H
