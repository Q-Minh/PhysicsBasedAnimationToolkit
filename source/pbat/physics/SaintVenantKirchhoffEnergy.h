/**
 * @file SaintVenantKirchhoffEnergy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Saint-Venant Kirchhoff hyperelastic energy
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup physics
 */

#ifndef PBAT_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
#define PBAT_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <cmath>

namespace pbat {
namespace physics {

template <int Dims>
struct SaintVenantKirchhoffEnergy;

/**
 * @brief Saint-Venant Kirchhoff hyperelastic energy for 1D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct SaintVenantKirchhoffEnergy<1>
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<1>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 =
        (((1.0 / 2.0) * ((F[0]) * (F[0])) - 1.0 / 2.0) *
         ((1.0 / 2.0) * ((F[0]) * (F[0])) - 1.0 / 2.0));
    psi = (1.0 / 2.0) * a0 * lambda + a0 * mu;
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<1>::SVector<typename TMatrix::ScalarType, 1>
SaintVenantKirchhoffEnergy<1>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 1> G;
    ScalarType const a0 = ((1.0 / 2.0) * ((F[0]) * (F[0])) - 1.0 / 2.0) * F[0];
    G[0]                = a0 * lambda + 2 * a0 * mu;
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<1>::SMatrix<typename TMatrix::ScalarType, 1, 1>
SaintVenantKirchhoffEnergy<1>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 1, 1> H;
    ScalarType const a0 = ((F[0]) * (F[0]));
    ScalarType const a1 = 2 * mu;
    ScalarType const a2 = (1.0 / 2.0) * a0 - 1.0 / 2.0;
    H[0]                = a0 * a1 + a0 * lambda + a1 * a2 + a2 * lambda;
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<1>::evalWithGrad(
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
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) - 1.0 / 2.0;
    ScalarType const a1 = ((a0) * (a0));
    ScalarType const a2 = a0 * F[0];
    psi                 = (1.0 / 2.0) * a1 * lambda + a1 * mu;
    gF[0]               = a2 * lambda + 2 * a2 * mu;
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<1>::evalWithGradAndHessian(
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
    ScalarType const a0 = ((F[0]) * (F[0]));
    ScalarType const a1 = (1.0 / 2.0) * a0 - 1.0 / 2.0;
    ScalarType const a2 = ((a1) * (a1));
    ScalarType const a3 = a1 * lambda;
    ScalarType const a4 = 2 * mu;
    ScalarType const a5 = a1 * a4;
    psi                 = (1.0 / 2.0) * a2 * lambda + a2 * mu;
    gF[0]               = a3 * F[0] + a5 * F[0];
    HF[0]               = a0 * a4 + a0 * lambda + a3 + a5;
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void SaintVenantKirchhoffEnergy<1>::gradAndHessian(
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
    using ScalarType    = typename TMatrix::ScalarType;
    ScalarType const a0 = ((F[0]) * (F[0]));
    ScalarType const a1 = (1.0 / 2.0) * a0 - 1.0 / 2.0;
    ScalarType const a2 = a1 * lambda;
    ScalarType const a3 = 2 * mu;
    ScalarType const a4 = a1 * a3;
    gF[0]               = a2 * F[0] + a4 * F[0];
    HF[0]               = a0 * a3 + a0 * lambda + a2 + a4;
}

/**
 * @brief Saint-Venant Kirchhoff hyperelastic energy for 2D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct SaintVenantKirchhoffEnergy<2>
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<2>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[2]) * (F[2])) + (1.0 / 2.0) * ((F[3]) * (F[3]));
    psi                 = (1.0 / 2.0) * lambda * ((a0 + a1 - 1) * (a0 + a1 - 1)) +
          mu * (((a0 - 1.0 / 2.0) * (a0 - 1.0 / 2.0)) + ((a1 - 1.0 / 2.0) * (a1 - 1.0 / 2.0)) +
                2 * (((1.0 / 2.0) * F[0] * F[2] + (1.0 / 2.0) * F[1] * F[3]) *
                     ((1.0 / 2.0) * F[0] * F[2] + (1.0 / 2.0) * F[1] * F[3])));
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<2>::SVector<typename TMatrix::ScalarType, 4>
SaintVenantKirchhoffEnergy<2>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 4> G;
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[2]) * (F[2])) + (1.0 / 2.0) * ((F[3]) * (F[3]));
    ScalarType const a2 = lambda * (a0 + a1 - 1);
    ScalarType const a3 = 2 * a0 - 1;
    ScalarType const a4 = F[0] * F[2] + F[1] * F[3];
    ScalarType const a5 = 2 * a1 - 1;
    G[0]                = a2 * F[0] + mu * (a3 * F[0] + a4 * F[2]);
    G[1]                = a2 * F[1] + mu * (a3 * F[1] + a4 * F[3]);
    G[2]                = a2 * F[2] + mu * (a4 * F[0] + a5 * F[2]);
    G[3]                = a2 * F[3] + mu * (a4 * F[1] + a5 * F[3]);
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<2>::SMatrix<typename TMatrix::ScalarType, 4, 4>
SaintVenantKirchhoffEnergy<2>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 4, 4> H;
    ScalarType const a0 = ((F[0]) * (F[0]));
    ScalarType const a1 = ((F[1]) * (F[1]));
    ScalarType const a2 = ((F[2]) * (F[2]));
    ScalarType const a3 = a1 + a2 - 1;
    ScalarType const a4 = ((F[3]) * (F[3]));
    ScalarType const a5 =
        lambda * ((1.0 / 2.0) * a0 + (1.0 / 2.0) * a1 + (1.0 / 2.0) * a2 + (1.0 / 2.0) * a4 - 1);
    ScalarType const a6  = F[0] * F[1];
    ScalarType const a7  = F[2] * F[3];
    ScalarType const a8  = a6 * lambda + mu * (2 * a6 + a7);
    ScalarType const a9  = F[0] * F[2];
    ScalarType const a10 = F[1] * F[3];
    ScalarType const a11 = a9 * lambda + mu * (a10 + 2 * a9);
    ScalarType const a12 = F[0] * F[3];
    ScalarType const a13 = F[1] * F[2];
    ScalarType const a14 = a12 * lambda + a13 * mu;
    ScalarType const a15 = a0 + a4 - 1;
    ScalarType const a16 = a12 * mu + a13 * lambda;
    ScalarType const a17 = a10 * lambda + mu * (2 * a10 + a9);
    ScalarType const a18 = a7 * lambda + mu * (a6 + 2 * a7);
    H[0]                 = a0 * lambda + a5 + mu * (3 * a0 + a3);
    H[1]                 = a8;
    H[2]                 = a11;
    H[3]                 = a14;
    H[4]                 = a8;
    H[5]                 = a1 * lambda + a5 + mu * (3 * a1 + a15);
    H[6]                 = a16;
    H[7]                 = a17;
    H[8]                 = a11;
    H[9]                 = a16;
    H[10]                = a2 * lambda + a5 + mu * (a15 + 3 * a2);
    H[11]                = a18;
    H[12]                = a14;
    H[13]                = a17;
    H[14]                = a18;
    H[15]                = a4 * lambda + a5 + mu * (a3 + 3 * a4);
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<2>::evalWithGrad(
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
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[2]) * (F[2])) + (1.0 / 2.0) * ((F[3]) * (F[3]));
    ScalarType const a2 = a0 + a1 - 1;
    ScalarType const a3 = a0 - 1.0 / 2.0;
    ScalarType const a4 = a1 - 1.0 / 2.0;
    ScalarType const a5 = (1.0 / 2.0) * F[0] * F[2] + (1.0 / 2.0) * F[1] * F[3];
    ScalarType const a6 = a2 * lambda;
    ScalarType const a7 = 2 * a3;
    ScalarType const a8 = 2 * a5;
    ScalarType const a9 = 2 * a4;
    psi                 = (1.0 / 2.0) * ((a2) * (a2)) * lambda +
          mu * (((a3) * (a3)) + ((a4) * (a4)) + 2 * ((a5) * (a5)));
    gF[0] = a6 * F[0] + mu * (a7 * F[0] + a8 * F[2]);
    gF[1] = a6 * F[1] + mu * (a7 * F[1] + a8 * F[3]);
    gF[2] = a6 * F[2] + mu * (a8 * F[0] + a9 * F[2]);
    gF[3] = a6 * F[3] + mu * (a8 * F[1] + a9 * F[3]);
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<2>::evalWithGradAndHessian(
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
    ScalarType const a2  = (1.0 / 2.0) * a0 + (1.0 / 2.0) * a1;
    ScalarType const a3  = ((F[2]) * (F[2]));
    ScalarType const a4  = ((F[3]) * (F[3]));
    ScalarType const a5  = (1.0 / 2.0) * a3 + (1.0 / 2.0) * a4;
    ScalarType const a6  = a2 + a5 - 1;
    ScalarType const a7  = a2 - 1.0 / 2.0;
    ScalarType const a8  = a5 - 1.0 / 2.0;
    ScalarType const a9  = F[0] * F[2];
    ScalarType const a10 = F[1] * F[3];
    ScalarType const a11 = (1.0 / 2.0) * a10 + (1.0 / 2.0) * a9;
    ScalarType const a12 = a6 * lambda;
    ScalarType const a13 = 2 * a7;
    ScalarType const a14 = 2 * a11;
    ScalarType const a15 = 2 * a8;
    ScalarType const a16 = a1 + a3 - 1;
    ScalarType const a17 = F[0] * F[1];
    ScalarType const a18 = F[2] * F[3];
    ScalarType const a19 = a17 * lambda + mu * (2 * a17 + a18);
    ScalarType const a20 = a9 * lambda + mu * (a10 + 2 * a9);
    ScalarType const a21 = F[0] * F[3];
    ScalarType const a22 = F[1] * F[2];
    ScalarType const a23 = a21 * lambda + a22 * mu;
    ScalarType const a24 = a0 + a4 - 1;
    ScalarType const a25 = a21 * mu + a22 * lambda;
    ScalarType const a26 = a10 * lambda + mu * (2 * a10 + a9);
    ScalarType const a27 = a18 * lambda + mu * (a17 + 2 * a18);
    psi                  = (1.0 / 2.0) * ((a6) * (a6)) * lambda +
          mu * (2 * ((a11) * (a11)) + ((a7) * (a7)) + ((a8) * (a8)));
    gF[0]  = a12 * F[0] + mu * (a13 * F[0] + a14 * F[2]);
    gF[1]  = a12 * F[1] + mu * (a13 * F[1] + a14 * F[3]);
    gF[2]  = a12 * F[2] + mu * (a14 * F[0] + a15 * F[2]);
    gF[3]  = a12 * F[3] + mu * (a14 * F[1] + a15 * F[3]);
    HF[0]  = a0 * lambda + a12 + mu * (3 * a0 + a16);
    HF[1]  = a19;
    HF[2]  = a20;
    HF[3]  = a23;
    HF[4]  = a19;
    HF[5]  = a1 * lambda + a12 + mu * (3 * a1 + a24);
    HF[6]  = a25;
    HF[7]  = a26;
    HF[8]  = a20;
    HF[9]  = a25;
    HF[10] = a12 + a3 * lambda + mu * (a24 + 3 * a3);
    HF[11] = a27;
    HF[12] = a23;
    HF[13] = a26;
    HF[14] = a27;
    HF[15] = a12 + a4 * lambda + mu * (a16 + 3 * a4);
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void SaintVenantKirchhoffEnergy<2>::gradAndHessian(
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
    using ScalarType     = typename TMatrix::ScalarType;
    ScalarType const a0  = ((F[0]) * (F[0]));
    ScalarType const a1  = ((F[1]) * (F[1]));
    ScalarType const a2  = (1.0 / 2.0) * a0 + (1.0 / 2.0) * a1;
    ScalarType const a3  = ((F[2]) * (F[2]));
    ScalarType const a4  = ((F[3]) * (F[3]));
    ScalarType const a5  = (1.0 / 2.0) * a3 + (1.0 / 2.0) * a4;
    ScalarType const a6  = lambda * (a2 + a5 - 1);
    ScalarType const a7  = 2 * a2 - 1;
    ScalarType const a8  = F[0] * F[2];
    ScalarType const a9  = F[1] * F[3];
    ScalarType const a10 = a8 + a9;
    ScalarType const a11 = 2 * a5 - 1;
    ScalarType const a12 = a1 + a3 - 1;
    ScalarType const a13 = F[0] * F[1];
    ScalarType const a14 = F[2] * F[3];
    ScalarType const a15 = a13 * lambda + mu * (2 * a13 + a14);
    ScalarType const a16 = a8 * lambda + mu * (2 * a8 + a9);
    ScalarType const a17 = F[0] * F[3];
    ScalarType const a18 = F[1] * F[2];
    ScalarType const a19 = a17 * lambda + a18 * mu;
    ScalarType const a20 = a0 + a4 - 1;
    ScalarType const a21 = a17 * mu + a18 * lambda;
    ScalarType const a22 = a9 * lambda + mu * (a8 + 2 * a9);
    ScalarType const a23 = a14 * lambda + mu * (a13 + 2 * a14);
    gF[0]                = a6 * F[0] + mu * (a10 * F[2] + a7 * F[0]);
    gF[1]                = a6 * F[1] + mu * (a10 * F[3] + a7 * F[1]);
    gF[2]                = a6 * F[2] + mu * (a10 * F[0] + a11 * F[2]);
    gF[3]                = a6 * F[3] + mu * (a10 * F[1] + a11 * F[3]);
    HF[0]                = a0 * lambda + a6 + mu * (3 * a0 + a12);
    HF[1]                = a15;
    HF[2]                = a16;
    HF[3]                = a19;
    HF[4]                = a15;
    HF[5]                = a1 * lambda + a6 + mu * (3 * a1 + a20);
    HF[6]                = a21;
    HF[7]                = a22;
    HF[8]                = a16;
    HF[9]                = a21;
    HF[10]               = a3 * lambda + a6 + mu * (a20 + 3 * a3);
    HF[11]               = a23;
    HF[12]               = a19;
    HF[13]               = a22;
    HF[14]               = a23;
    HF[15]               = a4 * lambda + a6 + mu * (a12 + 3 * a4);
}

/**
 * @brief Saint-Venant Kirchhoff hyperelastic energy for 3D
 *
 * @tparam Dims Dimension of the space
 * @ingroup physics
 */
template <>
struct SaintVenantKirchhoffEnergy<3>
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
     * @return ScalarType Energy and gradient
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
     * @return ScalarType Energy and gradient and hessian
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
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<3>::eval(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    ScalarType psi;
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1])) +
                          (1.0 / 2.0) * ((F[2]) * (F[2]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[3]) * (F[3])) + (1.0 / 2.0) * ((F[4]) * (F[4])) +
                          (1.0 / 2.0) * ((F[5]) * (F[5]));
    ScalarType const a2 = (1.0 / 2.0) * ((F[6]) * (F[6])) + (1.0 / 2.0) * ((F[7]) * (F[7])) +
                          (1.0 / 2.0) * ((F[8]) * (F[8]));
    ScalarType const a3 = (1.0 / 2.0) * F[0];
    ScalarType const a4 = (1.0 / 2.0) * F[1];
    ScalarType const a5 = (1.0 / 2.0) * F[2];
    psi = (1.0 / 2.0) * lambda * ((a0 + a1 + a2 - 3.0 / 2.0) * (a0 + a1 + a2 - 3.0 / 2.0)) +
          mu * (((a0 - 1.0 / 2.0) * (a0 - 1.0 / 2.0)) + ((a1 - 1.0 / 2.0) * (a1 - 1.0 / 2.0)) +
                ((a2 - 1.0 / 2.0) * (a2 - 1.0 / 2.0)) +
                2 * ((a3 * F[3] + a4 * F[4] + a5 * F[5]) * (a3 * F[3] + a4 * F[4] + a5 * F[5])) +
                2 * ((a3 * F[6] + a4 * F[7] + a5 * F[8]) * (a3 * F[6] + a4 * F[7] + a5 * F[8])) +
                2 * (((1.0 / 2.0) * F[3] * F[6] + (1.0 / 2.0) * F[4] * F[7] +
                      (1.0 / 2.0) * F[5] * F[8]) *
                     ((1.0 / 2.0) * F[3] * F[6] + (1.0 / 2.0) * F[4] * F[7] +
                      (1.0 / 2.0) * F[5] * F[8])));
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<3>::SVector<typename TMatrix::ScalarType, 9>
SaintVenantKirchhoffEnergy<3>::grad(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SVector<ScalarType, 9> G;
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1])) +
                          (1.0 / 2.0) * ((F[2]) * (F[2]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[3]) * (F[3])) + (1.0 / 2.0) * ((F[4]) * (F[4])) +
                          (1.0 / 2.0) * ((F[5]) * (F[5]));
    ScalarType const a2 = (1.0 / 2.0) * ((F[6]) * (F[6])) + (1.0 / 2.0) * ((F[7]) * (F[7])) +
                          (1.0 / 2.0) * ((F[8]) * (F[8]));
    ScalarType const a3  = lambda * (a0 + a1 + a2 - 3.0 / 2.0);
    ScalarType const a4  = 2 * a0 - 1;
    ScalarType const a5  = (1.0 / 2.0) * F[0];
    ScalarType const a6  = (1.0 / 2.0) * F[1];
    ScalarType const a7  = (1.0 / 2.0) * F[2];
    ScalarType const a8  = 2 * a5 * F[3] + 2 * a6 * F[4] + 2 * a7 * F[5];
    ScalarType const a9  = 2 * a5 * F[6] + 2 * a6 * F[7] + 2 * a7 * F[8];
    ScalarType const a10 = 2 * a1 - 1;
    ScalarType const a11 = F[3] * F[6] + F[4] * F[7] + F[5] * F[8];
    ScalarType const a12 = 2 * a2 - 1;
    G[0]                 = a3 * F[0] + mu * (a4 * F[0] + a8 * F[3] + a9 * F[6]);
    G[1]                 = a3 * F[1] + mu * (a4 * F[1] + a8 * F[4] + a9 * F[7]);
    G[2]                 = a3 * F[2] + mu * (a4 * F[2] + a8 * F[5] + a9 * F[8]);
    G[3]                 = a3 * F[3] + mu * (a10 * F[3] + a11 * F[6] + a8 * F[0]);
    G[4]                 = a3 * F[4] + mu * (a10 * F[4] + a11 * F[7] + a8 * F[1]);
    G[5]                 = a3 * F[5] + mu * (a10 * F[5] + a11 * F[8] + a8 * F[2]);
    G[6]                 = a3 * F[6] + mu * (a11 * F[3] + a12 * F[6] + a9 * F[0]);
    G[7]                 = a3 * F[7] + mu * (a11 * F[4] + a12 * F[7] + a9 * F[1]);
    G[8]                 = a3 * F[8] + mu * (a11 * F[5] + a12 * F[8] + a9 * F[2]);
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
PBAT_HOST_DEVICE SaintVenantKirchhoffEnergy<3>::SMatrix<typename TMatrix::ScalarType, 9, 9>
SaintVenantKirchhoffEnergy<3>::hessian(
    [[maybe_unused]] TMatrix const& F,
    [[maybe_unused]] typename TMatrix::ScalarType mu,
    [[maybe_unused]] typename TMatrix::ScalarType lambda) const
{
    using ScalarType = typename TMatrix::ScalarType;
    SMatrix<ScalarType, 9, 9> H;
    ScalarType const a0  = ((F[0]) * (F[0]));
    ScalarType const a1  = ((F[1]) * (F[1]));
    ScalarType const a2  = ((F[3]) * (F[3]));
    ScalarType const a3  = a1 + a2;
    ScalarType const a4  = ((F[6]) * (F[6]));
    ScalarType const a5  = ((F[2]) * (F[2]));
    ScalarType const a6  = a5 - 1;
    ScalarType const a7  = a4 + a6;
    ScalarType const a8  = ((F[4]) * (F[4]));
    ScalarType const a9  = ((F[5]) * (F[5]));
    ScalarType const a10 = ((F[7]) * (F[7]));
    ScalarType const a11 = ((F[8]) * (F[8]));
    ScalarType const a12 =
        lambda * ((1.0 / 2.0) * a0 + (1.0 / 2.0) * a1 + (1.0 / 2.0) * a10 + (1.0 / 2.0) * a11 +
                  (1.0 / 2.0) * a2 + (1.0 / 2.0) * a4 + (1.0 / 2.0) * a5 + (1.0 / 2.0) * a8 +
                  (1.0 / 2.0) * a9 - 3.0 / 2.0);
    ScalarType const a13 = F[0] * F[1];
    ScalarType const a14 = F[3] * F[4];
    ScalarType const a15 = F[6] * F[7];
    ScalarType const a16 = a13 * lambda + mu * (2 * a13 + a14 + a15);
    ScalarType const a17 = F[0] * F[2];
    ScalarType const a18 = F[3] * F[5];
    ScalarType const a19 = F[6] * F[8];
    ScalarType const a20 = a17 * lambda + mu * (2 * a17 + a18 + a19);
    ScalarType const a21 = F[0] * F[3];
    ScalarType const a22 = F[1] * F[4];
    ScalarType const a23 = F[2] * F[5];
    ScalarType const a24 = a21 * lambda + mu * (2 * a21 + a22 + a23);
    ScalarType const a25 = lambda * F[0];
    ScalarType const a26 = mu * F[3];
    ScalarType const a27 = a25 * F[4] + a26 * F[1];
    ScalarType const a28 = a25 * F[5] + a26 * F[2];
    ScalarType const a29 = F[0] * F[6];
    ScalarType const a30 = F[1] * F[7];
    ScalarType const a31 = F[2] * F[8];
    ScalarType const a32 = a29 * lambda + mu * (2 * a29 + a30 + a31);
    ScalarType const a33 = mu * F[6];
    ScalarType const a34 = a25 * F[7] + a33 * F[1];
    ScalarType const a35 = a25 * F[8] + a33 * F[2];
    ScalarType const a36 = a0 + a8;
    ScalarType const a37 = F[1] * F[2];
    ScalarType const a38 = F[4] * F[5];
    ScalarType const a39 = F[7] * F[8];
    ScalarType const a40 = a37 * lambda + mu * (2 * a37 + a38 + a39);
    ScalarType const a41 = lambda * F[1];
    ScalarType const a42 = mu * F[4];
    ScalarType const a43 = a41 * F[3] + a42 * F[0];
    ScalarType const a44 = a22 * lambda + mu * (a21 + 2 * a22 + a23);
    ScalarType const a45 = a41 * F[5] + a42 * F[2];
    ScalarType const a46 = mu * F[7];
    ScalarType const a47 = a41 * F[6] + a46 * F[0];
    ScalarType const a48 = a30 * lambda + mu * (a29 + 2 * a30 + a31);
    ScalarType const a49 = a41 * F[8] + a46 * F[2];
    ScalarType const a50 = a9 - 1;
    ScalarType const a51 = a0 + a11;
    ScalarType const a52 = lambda * F[2];
    ScalarType const a53 = mu * F[5];
    ScalarType const a54 = a52 * F[3] + a53 * F[0];
    ScalarType const a55 = a52 * F[4] + a53 * F[1];
    ScalarType const a56 = a23 * lambda + mu * (a21 + a22 + 2 * a23);
    ScalarType const a57 = mu * F[8];
    ScalarType const a58 = a52 * F[6] + a57 * F[0];
    ScalarType const a59 = a52 * F[7] + a57 * F[1];
    ScalarType const a60 = a31 * lambda + mu * (a29 + a30 + 2 * a31);
    ScalarType const a61 = a14 * lambda + mu * (a13 + 2 * a14 + a15);
    ScalarType const a62 = a18 * lambda + mu * (a17 + 2 * a18 + a19);
    ScalarType const a63 = F[3] * F[6];
    ScalarType const a64 = F[4] * F[7];
    ScalarType const a65 = F[5] * F[8];
    ScalarType const a66 = a63 * lambda + mu * (2 * a63 + a64 + a65);
    ScalarType const a67 = lambda * F[3];
    ScalarType const a68 = a33 * F[4] + a67 * F[7];
    ScalarType const a69 = a33 * F[5] + a67 * F[8];
    ScalarType const a70 = a38 * lambda + mu * (a37 + 2 * a38 + a39);
    ScalarType const a71 = lambda * F[4];
    ScalarType const a72 = a26 * F[7] + a71 * F[6];
    ScalarType const a73 = a64 * lambda + mu * (a63 + 2 * a64 + a65);
    ScalarType const a74 = a46 * F[5] + a71 * F[8];
    ScalarType const a75 = a11 + a8;
    ScalarType const a76 = lambda * F[5];
    ScalarType const a77 = a26 * F[8] + a76 * F[6];
    ScalarType const a78 = a42 * F[8] + a76 * F[7];
    ScalarType const a79 = a65 * lambda + mu * (a63 + a64 + 2 * a65);
    ScalarType const a80 = a15 * lambda + mu * (a13 + a14 + 2 * a15);
    ScalarType const a81 = a19 * lambda + mu * (a17 + a18 + 2 * a19);
    ScalarType const a82 = a39 * lambda + mu * (a37 + a38 + 2 * a39);
    H[0]                 = a0 * lambda + a12 + mu * (3 * a0 + a3 + a7);
    H[1]                 = a16;
    H[2]                 = a20;
    H[3]                 = a24;
    H[4]                 = a27;
    H[5]                 = a28;
    H[6]                 = a32;
    H[7]                 = a34;
    H[8]                 = a35;
    H[9]                 = a16;
    H[10]                = a1 * lambda + a12 + mu * (3 * a1 + a10 + a36 + a6);
    H[11]                = a40;
    H[12]                = a43;
    H[13]                = a44;
    H[14]                = a45;
    H[15]                = a47;
    H[16]                = a48;
    H[17]                = a49;
    H[18]                = a20;
    H[19]                = a40;
    H[20]                = a12 + a5 * lambda + mu * (a1 + 3 * a5 + a50 + a51);
    H[21]                = a54;
    H[22]                = a55;
    H[23]                = a56;
    H[24]                = a58;
    H[25]                = a59;
    H[26]                = a60;
    H[27]                = a24;
    H[28]                = a43;
    H[29]                = a54;
    H[30]                = a12 + a2 * lambda + mu * (3 * a2 + a36 + a4 + a50);
    H[31]                = a61;
    H[32]                = a62;
    H[33]                = a66;
    H[34]                = a68;
    H[35]                = a69;
    H[36]                = a27;
    H[37]                = a44;
    H[38]                = a55;
    H[39]                = a61;
    H[40]                = a12 + a8 * lambda + mu * (a10 + a3 + a50 + 3 * a8);
    H[41]                = a70;
    H[42]                = a72;
    H[43]                = a73;
    H[44]                = a74;
    H[45]                = a28;
    H[46]                = a45;
    H[47]                = a56;
    H[48]                = a62;
    H[49]                = a70;
    H[50]                = a12 + a9 * lambda + mu * (a2 + a6 + a75 + 3 * a9);
    H[51]                = a77;
    H[52]                = a78;
    H[53]                = a79;
    H[54]                = a32;
    H[55]                = a47;
    H[56]                = a58;
    H[57]                = a66;
    H[58]                = a72;
    H[59]                = a77;
    H[60]                = a12 + a4 * lambda + mu * (a10 + a2 + 3 * a4 + a51 - 1);
    H[61]                = a80;
    H[62]                = a81;
    H[63]                = a34;
    H[64]                = a48;
    H[65]                = a59;
    H[66]                = a68;
    H[67]                = a73;
    H[68]                = a78;
    H[69]                = a80;
    H[70]                = a10 * lambda + a12 + mu * (a1 + 3 * a10 + a4 + a75 - 1);
    H[71]                = a82;
    H[72]                = a35;
    H[73]                = a49;
    H[74]                = a60;
    H[75]                = a69;
    H[76]                = a74;
    H[77]                = a79;
    H[78]                = a81;
    H[79]                = a82;
    H[80]                = a11 * lambda + a12 + mu * (a10 + 3 * a11 + a7 + a9);
    return H;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<3>::evalWithGrad(
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
    ScalarType const a0 = (1.0 / 2.0) * ((F[0]) * (F[0])) + (1.0 / 2.0) * ((F[1]) * (F[1])) +
                          (1.0 / 2.0) * ((F[2]) * (F[2]));
    ScalarType const a1 = (1.0 / 2.0) * ((F[3]) * (F[3])) + (1.0 / 2.0) * ((F[4]) * (F[4])) +
                          (1.0 / 2.0) * ((F[5]) * (F[5]));
    ScalarType const a2 = (1.0 / 2.0) * ((F[6]) * (F[6])) + (1.0 / 2.0) * ((F[7]) * (F[7])) +
                          (1.0 / 2.0) * ((F[8]) * (F[8]));
    ScalarType const a3  = a0 + a1 + a2 - 3.0 / 2.0;
    ScalarType const a4  = a0 - 1.0 / 2.0;
    ScalarType const a5  = a1 - 1.0 / 2.0;
    ScalarType const a6  = a2 - 1.0 / 2.0;
    ScalarType const a7  = (1.0 / 2.0) * F[0];
    ScalarType const a8  = (1.0 / 2.0) * F[1];
    ScalarType const a9  = (1.0 / 2.0) * F[2];
    ScalarType const a10 = a7 * F[3] + a8 * F[4] + a9 * F[5];
    ScalarType const a11 = a7 * F[6] + a8 * F[7] + a9 * F[8];
    ScalarType const a12 =
        (1.0 / 2.0) * F[3] * F[6] + (1.0 / 2.0) * F[4] * F[7] + (1.0 / 2.0) * F[5] * F[8];
    ScalarType const a13 = a3 * lambda;
    ScalarType const a14 = 2 * a4;
    ScalarType const a15 = 2 * a10;
    ScalarType const a16 = 2 * a11;
    ScalarType const a17 = 2 * a5;
    ScalarType const a18 = 2 * a12;
    ScalarType const a19 = 2 * a6;
    psi                  = (1.0 / 2.0) * ((a3) * (a3)) * lambda +
          mu * (2 * ((a10) * (a10)) + 2 * ((a11) * (a11)) + 2 * ((a12) * (a12)) + ((a4) * (a4)) +
                ((a5) * (a5)) + ((a6) * (a6)));
    gF[0] = a13 * F[0] + mu * (a14 * F[0] + a15 * F[3] + a16 * F[6]);
    gF[1] = a13 * F[1] + mu * (a14 * F[1] + a15 * F[4] + a16 * F[7]);
    gF[2] = a13 * F[2] + mu * (a14 * F[2] + a15 * F[5] + a16 * F[8]);
    gF[3] = a13 * F[3] + mu * (a15 * F[0] + a17 * F[3] + a18 * F[6]);
    gF[4] = a13 * F[4] + mu * (a15 * F[1] + a17 * F[4] + a18 * F[7]);
    gF[5] = a13 * F[5] + mu * (a15 * F[2] + a17 * F[5] + a18 * F[8]);
    gF[6] = a13 * F[6] + mu * (a16 * F[0] + a18 * F[3] + a19 * F[6]);
    gF[7] = a13 * F[7] + mu * (a16 * F[1] + a18 * F[4] + a19 * F[7]);
    gF[8] = a13 * F[8] + mu * (a16 * F[2] + a18 * F[5] + a19 * F[8]);
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE typename TMatrix::ScalarType SaintVenantKirchhoffEnergy<3>::evalWithGradAndHessian(
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
    ScalarType const a0  = ((F[0]) * (F[0]));
    ScalarType const a1  = ((F[1]) * (F[1]));
    ScalarType const a2  = ((F[2]) * (F[2]));
    ScalarType const a3  = (1.0 / 2.0) * a0 + (1.0 / 2.0) * a1 + (1.0 / 2.0) * a2;
    ScalarType const a4  = ((F[3]) * (F[3]));
    ScalarType const a5  = ((F[4]) * (F[4]));
    ScalarType const a6  = ((F[5]) * (F[5]));
    ScalarType const a7  = (1.0 / 2.0) * a4 + (1.0 / 2.0) * a5 + (1.0 / 2.0) * a6;
    ScalarType const a8  = ((F[6]) * (F[6]));
    ScalarType const a9  = ((F[7]) * (F[7]));
    ScalarType const a10 = ((F[8]) * (F[8]));
    ScalarType const a11 = (1.0 / 2.0) * a10 + (1.0 / 2.0) * a8 + (1.0 / 2.0) * a9;
    ScalarType const a12 = a11 + a3 + a7 - 3.0 / 2.0;
    ScalarType const a13 = a3 - 1.0 / 2.0;
    ScalarType const a14 = a7 - 1.0 / 2.0;
    ScalarType const a15 = a11 - 1.0 / 2.0;
    ScalarType const a16 = F[0] * F[3];
    ScalarType const a17 = F[1] * F[4];
    ScalarType const a18 = F[2] * F[5];
    ScalarType const a19 = (1.0 / 2.0) * a16 + (1.0 / 2.0) * a17 + (1.0 / 2.0) * a18;
    ScalarType const a20 = F[0] * F[6];
    ScalarType const a21 = F[1] * F[7];
    ScalarType const a22 = F[2] * F[8];
    ScalarType const a23 = (1.0 / 2.0) * a20 + (1.0 / 2.0) * a21 + (1.0 / 2.0) * a22;
    ScalarType const a24 = F[3] * F[6];
    ScalarType const a25 = F[4] * F[7];
    ScalarType const a26 = F[5] * F[8];
    ScalarType const a27 = (1.0 / 2.0) * a24 + (1.0 / 2.0) * a25 + (1.0 / 2.0) * a26;
    ScalarType const a28 = a12 * lambda;
    ScalarType const a29 = 2 * a13;
    ScalarType const a30 = 2 * a19;
    ScalarType const a31 = 2 * a23;
    ScalarType const a32 = 2 * a14;
    ScalarType const a33 = 2 * a27;
    ScalarType const a34 = 2 * a15;
    ScalarType const a35 = a1 + a4;
    ScalarType const a36 = a2 - 1;
    ScalarType const a37 = a36 + a8;
    ScalarType const a38 = F[0] * F[1];
    ScalarType const a39 = F[3] * F[4];
    ScalarType const a40 = F[6] * F[7];
    ScalarType const a41 = a38 * lambda + mu * (2 * a38 + a39 + a40);
    ScalarType const a42 = F[0] * F[2];
    ScalarType const a43 = F[3] * F[5];
    ScalarType const a44 = F[6] * F[8];
    ScalarType const a45 = a42 * lambda + mu * (2 * a42 + a43 + a44);
    ScalarType const a46 = a16 * lambda + mu * (2 * a16 + a17 + a18);
    ScalarType const a47 = lambda * F[0];
    ScalarType const a48 = mu * F[3];
    ScalarType const a49 = a47 * F[4] + a48 * F[1];
    ScalarType const a50 = a47 * F[5] + a48 * F[2];
    ScalarType const a51 = a20 * lambda + mu * (2 * a20 + a21 + a22);
    ScalarType const a52 = mu * F[6];
    ScalarType const a53 = a47 * F[7] + a52 * F[1];
    ScalarType const a54 = a47 * F[8] + a52 * F[2];
    ScalarType const a55 = a0 + a5;
    ScalarType const a56 = F[1] * F[2];
    ScalarType const a57 = F[4] * F[5];
    ScalarType const a58 = F[7] * F[8];
    ScalarType const a59 = a56 * lambda + mu * (2 * a56 + a57 + a58);
    ScalarType const a60 = lambda * F[1];
    ScalarType const a61 = mu * F[4];
    ScalarType const a62 = a60 * F[3] + a61 * F[0];
    ScalarType const a63 = a17 * lambda + mu * (a16 + 2 * a17 + a18);
    ScalarType const a64 = a60 * F[5] + a61 * F[2];
    ScalarType const a65 = mu * F[7];
    ScalarType const a66 = a60 * F[6] + a65 * F[0];
    ScalarType const a67 = a21 * lambda + mu * (a20 + 2 * a21 + a22);
    ScalarType const a68 = a60 * F[8] + a65 * F[2];
    ScalarType const a69 = a6 - 1;
    ScalarType const a70 = a0 + a10;
    ScalarType const a71 = lambda * F[2];
    ScalarType const a72 = mu * F[5];
    ScalarType const a73 = a71 * F[3] + a72 * F[0];
    ScalarType const a74 = a71 * F[4] + a72 * F[1];
    ScalarType const a75 = a18 * lambda + mu * (a16 + a17 + 2 * a18);
    ScalarType const a76 = mu * F[8];
    ScalarType const a77 = a71 * F[6] + a76 * F[0];
    ScalarType const a78 = a71 * F[7] + a76 * F[1];
    ScalarType const a79 = a22 * lambda + mu * (a20 + a21 + 2 * a22);
    ScalarType const a80 = a39 * lambda + mu * (a38 + 2 * a39 + a40);
    ScalarType const a81 = a43 * lambda + mu * (a42 + 2 * a43 + a44);
    ScalarType const a82 = a24 * lambda + mu * (2 * a24 + a25 + a26);
    ScalarType const a83 = lambda * F[3];
    ScalarType const a84 = a52 * F[4] + a83 * F[7];
    ScalarType const a85 = a52 * F[5] + a83 * F[8];
    ScalarType const a86 = a57 * lambda + mu * (a56 + 2 * a57 + a58);
    ScalarType const a87 = lambda * F[4];
    ScalarType const a88 = a48 * F[7] + a87 * F[6];
    ScalarType const a89 = a25 * lambda + mu * (a24 + 2 * a25 + a26);
    ScalarType const a90 = a65 * F[5] + a87 * F[8];
    ScalarType const a91 = a10 + a5;
    ScalarType const a92 = lambda * F[5];
    ScalarType const a93 = a48 * F[8] + a92 * F[6];
    ScalarType const a94 = a61 * F[8] + a92 * F[7];
    ScalarType const a95 = a26 * lambda + mu * (a24 + a25 + 2 * a26);
    ScalarType const a96 = a40 * lambda + mu * (a38 + a39 + 2 * a40);
    ScalarType const a97 = a44 * lambda + mu * (a42 + a43 + 2 * a44);
    ScalarType const a98 = a58 * lambda + mu * (a56 + a57 + 2 * a58);
    psi                  = (1.0 / 2.0) * ((a12) * (a12)) * lambda +
          mu * (((a13) * (a13)) + ((a14) * (a14)) + ((a15) * (a15)) + 2 * ((a19) * (a19)) +
                2 * ((a23) * (a23)) + 2 * ((a27) * (a27)));
    gF[0]  = a28 * F[0] + mu * (a29 * F[0] + a30 * F[3] + a31 * F[6]);
    gF[1]  = a28 * F[1] + mu * (a29 * F[1] + a30 * F[4] + a31 * F[7]);
    gF[2]  = a28 * F[2] + mu * (a29 * F[2] + a30 * F[5] + a31 * F[8]);
    gF[3]  = a28 * F[3] + mu * (a30 * F[0] + a32 * F[3] + a33 * F[6]);
    gF[4]  = a28 * F[4] + mu * (a30 * F[1] + a32 * F[4] + a33 * F[7]);
    gF[5]  = a28 * F[5] + mu * (a30 * F[2] + a32 * F[5] + a33 * F[8]);
    gF[6]  = a28 * F[6] + mu * (a31 * F[0] + a33 * F[3] + a34 * F[6]);
    gF[7]  = a28 * F[7] + mu * (a31 * F[1] + a33 * F[4] + a34 * F[7]);
    gF[8]  = a28 * F[8] + mu * (a31 * F[2] + a33 * F[5] + a34 * F[8]);
    HF[0]  = a0 * lambda + a28 + mu * (3 * a0 + a35 + a37);
    HF[1]  = a41;
    HF[2]  = a45;
    HF[3]  = a46;
    HF[4]  = a49;
    HF[5]  = a50;
    HF[6]  = a51;
    HF[7]  = a53;
    HF[8]  = a54;
    HF[9]  = a41;
    HF[10] = a1 * lambda + a28 + mu * (3 * a1 + a36 + a55 + a9);
    HF[11] = a59;
    HF[12] = a62;
    HF[13] = a63;
    HF[14] = a64;
    HF[15] = a66;
    HF[16] = a67;
    HF[17] = a68;
    HF[18] = a45;
    HF[19] = a59;
    HF[20] = a2 * lambda + a28 + mu * (a1 + 3 * a2 + a69 + a70);
    HF[21] = a73;
    HF[22] = a74;
    HF[23] = a75;
    HF[24] = a77;
    HF[25] = a78;
    HF[26] = a79;
    HF[27] = a46;
    HF[28] = a62;
    HF[29] = a73;
    HF[30] = a28 + a4 * lambda + mu * (3 * a4 + a55 + a69 + a8);
    HF[31] = a80;
    HF[32] = a81;
    HF[33] = a82;
    HF[34] = a84;
    HF[35] = a85;
    HF[36] = a49;
    HF[37] = a63;
    HF[38] = a74;
    HF[39] = a80;
    HF[40] = a28 + a5 * lambda + mu * (a35 + 3 * a5 + a69 + a9);
    HF[41] = a86;
    HF[42] = a88;
    HF[43] = a89;
    HF[44] = a90;
    HF[45] = a50;
    HF[46] = a64;
    HF[47] = a75;
    HF[48] = a81;
    HF[49] = a86;
    HF[50] = a28 + a6 * lambda + mu * (a36 + a4 + 3 * a6 + a91);
    HF[51] = a93;
    HF[52] = a94;
    HF[53] = a95;
    HF[54] = a51;
    HF[55] = a66;
    HF[56] = a77;
    HF[57] = a82;
    HF[58] = a88;
    HF[59] = a93;
    HF[60] = a28 + a8 * lambda + mu * (a4 + a70 + 3 * a8 + a9 - 1);
    HF[61] = a96;
    HF[62] = a97;
    HF[63] = a53;
    HF[64] = a67;
    HF[65] = a78;
    HF[66] = a84;
    HF[67] = a89;
    HF[68] = a94;
    HF[69] = a96;
    HF[70] = a28 + a9 * lambda + mu * (a1 + a8 + 3 * a9 + a91 - 1);
    HF[71] = a98;
    HF[72] = a54;
    HF[73] = a68;
    HF[74] = a79;
    HF[75] = a85;
    HF[76] = a90;
    HF[77] = a95;
    HF[78] = a97;
    HF[79] = a98;
    HF[80] = a10 * lambda + a28 + mu * (3 * a10 + a37 + a6 + a9);
    return psi;
}

template <
    math::linalg::mini::CReadableVectorizedMatrix TMatrix,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixGF,
    math::linalg::mini::CWriteableVectorizedMatrix TMatrixHF>
PBAT_HOST_DEVICE void SaintVenantKirchhoffEnergy<3>::gradAndHessian(
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
    ScalarType const a0  = ((F[0]) * (F[0]));
    ScalarType const a1  = ((F[1]) * (F[1]));
    ScalarType const a2  = ((F[2]) * (F[2]));
    ScalarType const a3  = (1.0 / 2.0) * a0 + (1.0 / 2.0) * a1 + (1.0 / 2.0) * a2;
    ScalarType const a4  = ((F[3]) * (F[3]));
    ScalarType const a5  = ((F[4]) * (F[4]));
    ScalarType const a6  = ((F[5]) * (F[5]));
    ScalarType const a7  = (1.0 / 2.0) * a4 + (1.0 / 2.0) * a5 + (1.0 / 2.0) * a6;
    ScalarType const a8  = ((F[6]) * (F[6]));
    ScalarType const a9  = ((F[7]) * (F[7]));
    ScalarType const a10 = ((F[8]) * (F[8]));
    ScalarType const a11 = (1.0 / 2.0) * a10 + (1.0 / 2.0) * a8 + (1.0 / 2.0) * a9;
    ScalarType const a12 = lambda * (a11 + a3 + a7 - 3.0 / 2.0);
    ScalarType const a13 = 2 * a3 - 1;
    ScalarType const a14 = F[0] * F[3];
    ScalarType const a15 = F[1] * F[4];
    ScalarType const a16 = F[2] * F[5];
    ScalarType const a17 = a14 + a15 + a16;
    ScalarType const a18 = F[0] * F[6];
    ScalarType const a19 = F[1] * F[7];
    ScalarType const a20 = F[2] * F[8];
    ScalarType const a21 = a18 + a19 + a20;
    ScalarType const a22 = 2 * a7 - 1;
    ScalarType const a23 = F[3] * F[6];
    ScalarType const a24 = F[4] * F[7];
    ScalarType const a25 = F[5] * F[8];
    ScalarType const a26 = a23 + a24 + a25;
    ScalarType const a27 = 2 * a11 - 1;
    ScalarType const a28 = a1 + a4;
    ScalarType const a29 = a2 - 1;
    ScalarType const a30 = a29 + a8;
    ScalarType const a31 = F[0] * F[1];
    ScalarType const a32 = F[3] * F[4];
    ScalarType const a33 = F[6] * F[7];
    ScalarType const a34 = a31 * lambda + mu * (2 * a31 + a32 + a33);
    ScalarType const a35 = F[0] * F[2];
    ScalarType const a36 = F[3] * F[5];
    ScalarType const a37 = F[6] * F[8];
    ScalarType const a38 = a35 * lambda + mu * (2 * a35 + a36 + a37);
    ScalarType const a39 = a14 * lambda + mu * (2 * a14 + a15 + a16);
    ScalarType const a40 = lambda * F[0];
    ScalarType const a41 = mu * F[3];
    ScalarType const a42 = a40 * F[4] + a41 * F[1];
    ScalarType const a43 = a40 * F[5] + a41 * F[2];
    ScalarType const a44 = a18 * lambda + mu * (2 * a18 + a19 + a20);
    ScalarType const a45 = mu * F[6];
    ScalarType const a46 = a40 * F[7] + a45 * F[1];
    ScalarType const a47 = a40 * F[8] + a45 * F[2];
    ScalarType const a48 = a0 + a5;
    ScalarType const a49 = F[1] * F[2];
    ScalarType const a50 = F[4] * F[5];
    ScalarType const a51 = F[7] * F[8];
    ScalarType const a52 = a49 * lambda + mu * (2 * a49 + a50 + a51);
    ScalarType const a53 = lambda * F[1];
    ScalarType const a54 = mu * F[4];
    ScalarType const a55 = a53 * F[3] + a54 * F[0];
    ScalarType const a56 = a15 * lambda + mu * (a14 + 2 * a15 + a16);
    ScalarType const a57 = a53 * F[5] + a54 * F[2];
    ScalarType const a58 = mu * F[7];
    ScalarType const a59 = a53 * F[6] + a58 * F[0];
    ScalarType const a60 = a19 * lambda + mu * (a18 + 2 * a19 + a20);
    ScalarType const a61 = a53 * F[8] + a58 * F[2];
    ScalarType const a62 = a6 - 1;
    ScalarType const a63 = a0 + a10;
    ScalarType const a64 = lambda * F[2];
    ScalarType const a65 = mu * F[5];
    ScalarType const a66 = a64 * F[3] + a65 * F[0];
    ScalarType const a67 = a64 * F[4] + a65 * F[1];
    ScalarType const a68 = a16 * lambda + mu * (a14 + a15 + 2 * a16);
    ScalarType const a69 = mu * F[8];
    ScalarType const a70 = a64 * F[6] + a69 * F[0];
    ScalarType const a71 = a64 * F[7] + a69 * F[1];
    ScalarType const a72 = a20 * lambda + mu * (a18 + a19 + 2 * a20);
    ScalarType const a73 = a32 * lambda + mu * (a31 + 2 * a32 + a33);
    ScalarType const a74 = a36 * lambda + mu * (a35 + 2 * a36 + a37);
    ScalarType const a75 = a23 * lambda + mu * (2 * a23 + a24 + a25);
    ScalarType const a76 = lambda * F[3];
    ScalarType const a77 = a45 * F[4] + a76 * F[7];
    ScalarType const a78 = a45 * F[5] + a76 * F[8];
    ScalarType const a79 = a50 * lambda + mu * (a49 + 2 * a50 + a51);
    ScalarType const a80 = lambda * F[4];
    ScalarType const a81 = a41 * F[7] + a80 * F[6];
    ScalarType const a82 = a24 * lambda + mu * (a23 + 2 * a24 + a25);
    ScalarType const a83 = a58 * F[5] + a80 * F[8];
    ScalarType const a84 = a10 + a5;
    ScalarType const a85 = lambda * F[5];
    ScalarType const a86 = a41 * F[8] + a85 * F[6];
    ScalarType const a87 = a54 * F[8] + a85 * F[7];
    ScalarType const a88 = a25 * lambda + mu * (a23 + a24 + 2 * a25);
    ScalarType const a89 = a33 * lambda + mu * (a31 + a32 + 2 * a33);
    ScalarType const a90 = a37 * lambda + mu * (a35 + a36 + 2 * a37);
    ScalarType const a91 = a51 * lambda + mu * (a49 + a50 + 2 * a51);
    gF[0]                = a12 * F[0] + mu * (a13 * F[0] + a17 * F[3] + a21 * F[6]);
    gF[1]                = a12 * F[1] + mu * (a13 * F[1] + a17 * F[4] + a21 * F[7]);
    gF[2]                = a12 * F[2] + mu * (a13 * F[2] + a17 * F[5] + a21 * F[8]);
    gF[3]                = a12 * F[3] + mu * (a17 * F[0] + a22 * F[3] + a26 * F[6]);
    gF[4]                = a12 * F[4] + mu * (a17 * F[1] + a22 * F[4] + a26 * F[7]);
    gF[5]                = a12 * F[5] + mu * (a17 * F[2] + a22 * F[5] + a26 * F[8]);
    gF[6]                = a12 * F[6] + mu * (a21 * F[0] + a26 * F[3] + a27 * F[6]);
    gF[7]                = a12 * F[7] + mu * (a21 * F[1] + a26 * F[4] + a27 * F[7]);
    gF[8]                = a12 * F[8] + mu * (a21 * F[2] + a26 * F[5] + a27 * F[8]);
    HF[0]                = a0 * lambda + a12 + mu * (3 * a0 + a28 + a30);
    HF[1]                = a34;
    HF[2]                = a38;
    HF[3]                = a39;
    HF[4]                = a42;
    HF[5]                = a43;
    HF[6]                = a44;
    HF[7]                = a46;
    HF[8]                = a47;
    HF[9]                = a34;
    HF[10]               = a1 * lambda + a12 + mu * (3 * a1 + a29 + a48 + a9);
    HF[11]               = a52;
    HF[12]               = a55;
    HF[13]               = a56;
    HF[14]               = a57;
    HF[15]               = a59;
    HF[16]               = a60;
    HF[17]               = a61;
    HF[18]               = a38;
    HF[19]               = a52;
    HF[20]               = a12 + a2 * lambda + mu * (a1 + 3 * a2 + a62 + a63);
    HF[21]               = a66;
    HF[22]               = a67;
    HF[23]               = a68;
    HF[24]               = a70;
    HF[25]               = a71;
    HF[26]               = a72;
    HF[27]               = a39;
    HF[28]               = a55;
    HF[29]               = a66;
    HF[30]               = a12 + a4 * lambda + mu * (3 * a4 + a48 + a62 + a8);
    HF[31]               = a73;
    HF[32]               = a74;
    HF[33]               = a75;
    HF[34]               = a77;
    HF[35]               = a78;
    HF[36]               = a42;
    HF[37]               = a56;
    HF[38]               = a67;
    HF[39]               = a73;
    HF[40]               = a12 + a5 * lambda + mu * (a28 + 3 * a5 + a62 + a9);
    HF[41]               = a79;
    HF[42]               = a81;
    HF[43]               = a82;
    HF[44]               = a83;
    HF[45]               = a43;
    HF[46]               = a57;
    HF[47]               = a68;
    HF[48]               = a74;
    HF[49]               = a79;
    HF[50]               = a12 + a6 * lambda + mu * (a29 + a4 + 3 * a6 + a84);
    HF[51]               = a86;
    HF[52]               = a87;
    HF[53]               = a88;
    HF[54]               = a44;
    HF[55]               = a59;
    HF[56]               = a70;
    HF[57]               = a75;
    HF[58]               = a81;
    HF[59]               = a86;
    HF[60]               = a12 + a8 * lambda + mu * (a4 + a63 + 3 * a8 + a9 - 1);
    HF[61]               = a89;
    HF[62]               = a90;
    HF[63]               = a46;
    HF[64]               = a60;
    HF[65]               = a71;
    HF[66]               = a77;
    HF[67]               = a82;
    HF[68]               = a87;
    HF[69]               = a89;
    HF[70]               = a12 + a9 * lambda + mu * (a1 + a8 + a84 + 3 * a9 - 1);
    HF[71]               = a91;
    HF[72]               = a47;
    HF[73]               = a61;
    HF[74]               = a72;
    HF[75]               = a78;
    HF[76]               = a83;
    HF[77]               = a88;
    HF[78]               = a90;
    HF[79]               = a91;
    HF[80]               = a10 * lambda + a12 + mu * (3 * a10 + a30 + a6 + a9);
}

} // namespace physics
} // namespace pbat

#endif // PBAT_PHYSICS_SAINTVENANTKIRCHHOFFENERGY_H
