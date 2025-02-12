/**
 * @file DeformationGradient.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Functions to compute deformation gradient and its derivatives.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_DEFORMATIONGRADIENT_H
#define PBAT_FEM_DEFORMATIONGRADIENT_H

#include "Concepts.h"
#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Concepts.h"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat {
namespace fem {

/**
 * @brief Computes the deformation gradient \f$ \frac{\partial \mathbf{x}(X)}{\partial X} \f$ of the
 * deformation map \f$ \mathbf{x}(X) \f$.
 *
 * If the problem is discretized with displacement coefficients \f$ \mathbf{u} = \mathbf{x}(X) - X
 * \f$, then simply feed this function with argument \f$ \mathbf{x} = X + \mathbf{u} \f$.
 *
 * @tparam TDerivedU Eigen matrix expression
 * @tparam TDerivedX Eigen matrix expression
 * @tparam TElement FEM element type
 * @param x Matrix of column-wise position nodal coefficients
 * @param GP Basis function gradients
 * @return Deformation gradient matrix
 *
 */
template <CElement TElement, class TDerivedx, class TDerivedX>
auto DeformationGradient(
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedX> const& GP) -> Matrix<TDerivedx::RowsAtCompileTime, TElement::kDims>
{
    return x * GP;
}

/**
 * @brief Computes \f$ \frac{\partial \Psi}{\partial \mathbf{x}_i} \in \mathbb{R}^d \f$, i.e. the
 * gradient of a scalar function \f$ \Psi \f$ w.r.t. the \f$ i^{\text{th}}
 * \f$ node's degrees of freedom.
 *
 * @tparam TElement FEM element type
 * @tparam Dims Problem dimensionality
 * @tparam TMatrixGF Mini matrix type
 * @tparam TMatrixGP Mini matrix type
 * @tparam Scalar Coefficient type of returned gradient
 * @param GF \f$ d^2 \times 1 \f$ gradient of \f$ \Psi \f$ w.r.t. vectorized jacobian \f$
 * \text{vec}(\mathbf{F}) \f$
 * @param GP \f$ * \times d \f$ basis function gradients \f$ \nabla \mathbf{N}_i \f$
 * @param i Basis function index
 * @return \f$ d \times 1 \f$ vector \f$ \frac{\partial \Psi}{\partial \mathbf{x}_i} \f$
 *
 */
template <
    CElement TElement,
    int Dims,
    math::linalg::mini::CMatrix TMatrixGF,
    math::linalg::mini::CMatrix TMatrixGP,
    class ScalarType = typename TMatrixGF::ScalarType>
auto GradientSegmentWrtDofs(TMatrixGF const& GF, TMatrixGP const& GP, auto i)
    -> math::linalg::mini::SVector<ScalarType, Dims>
{
    using namespace math::linalg::mini;
    SVector<ScalarType, Dims> dPsidx = Zeros<ScalarType, Dims, 1>{};
    for (auto k = 0; k < Dims; ++k)
    {
        dPsidx += GP(i, k) * GF.template Slice<Dims, 1>(k * Dims, 0);
    }
    return dPsidx;
}

/**
 * @brief Computes gradient w.r.t. FEM degrees of freedom \f$ x \f$ of scalar function \f$
 * \Psi(\mathbf{F}) \f$, where \f$ F \f$ is the jacobian of \f$ x \f$, via chain rule. This is
 * effectively a rank-3 to rank-1 tensor contraction.
 *
 * Let \f$ \mathbf{g}_i \f$ be the \f$ i^{\text{th}} \f$ basis function gradient, then
 *
 * \f[
 * \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i} = \frac{\partial}{\partial \mathbf{x}_i}
 * \mathbf{x}_i \mathbf{g}_i^T, \mathbf{x}_i \in R^d, \mathbf{g}_i \in R^d
 * \f]
 *
 * Thus,
 * \f[
 * \frac{\partial \text{vec}(\mathbf{F})}{\partial \mathbf{x}_i} = \mathbf{g}_i \otimes
 * \mathbf{I}_{d x d} \in R^{d^2 \times d}
 * \f]
 *
 * With \f$ \mathbf{G}_F = \nabla_{\text{vec}(\mathbf{F})} \Psi \in \mathbb{R}^{d^2} \f$,
 * and \f$ \mathbf{G}_F^k \f$ the \f$ k^\text{th} \; d \times 1 \f$ block of \f$ \mathbf{G}_F \f$,
 * then
 *
 * \f[
 * \frac{\partial \Psi}{\partial \mathbf{x}_i}
 * = \frac{\partial \Psi}{\partial \text{vec}(\mathbf{F})} \frac{\partial
 * \text{vec}(\mathbf{F})}{\partial \mathbf{x}_i}
 * = \sum_{k=1}^{d} \mathbf{G}_F^k \mathbf{g}_{ik}
 * \f]
 *
 * @tparam TElement FEM element type
 * @tparam Dims Problem dimensionality
 * @tparam TMatrixGF Mini matrix type
 * @tparam TMatrixGP Mini matrix type
 * @tparam Scalar Coefficient type of returned gradient
 * @param GF \f$ d^2 \times 1 \f$ gradient of \f$ \Psi \f$ w.r.t. vectorized jacobian \f$
 * \text{vec}(\mathbf{F}) \f$
 * @param GP \f$ * \times d \f$ basis function gradients \f$ \nabla \mathbf{N}_i \f$
 * @return \f$ d \times 1 \f$ vector \f$ \frac{\partial \Psi}{\partial \mathbf{x}_i} \f$
 *
 */
template <
    CElement TElement,
    int Dims,
    math::linalg::mini::CMatrix TMatrixGF,
    math::linalg::mini::CMatrix TMatrixGP,
    class ScalarType = typename TMatrixGF::ScalarType>
auto GradientWrtDofs(TMatrixGF const& GF, TMatrixGP const& GP)
    -> math::linalg::mini::SVector<ScalarType, TElement::kNodes * Dims>
{
    auto constexpr kRows = TElement::kNodes * Dims;
    using namespace math::linalg::mini;
    SVector<ScalarType, kRows> dPsidx = Zeros<ScalarType, kRows, 1>{};
    for (auto k = 0; k < Dims; ++k)
    {
        for (auto i = 0; i < TElement::kNodes; ++i)
        {
            dPsidx.template Slice<Dims, 1>(i * Dims, 0) +=
                GP(i, k) * GF.template Slice<Dims, 1>(k * Dims, 0);
        }
    }
    return dPsidx;
}

/**
 * @brief Computes \f$ \frac{\partial^2 \Psi}{\partial \mathbf{x}_i \partial \mathbf{x}_j} \f$, i.e.
 * the hessian of a scalar function \f$ \Psi \f$ w.r.t. the \f$ i^{\text{th}} \f$ and \f$
 * j^{\text{th}} \f$ node's degrees of freedom.
 *
 * @tparam TElement FEM element type
 * @tparam Dims Problem dimensionality
 * @tparam TMatrixHF Mini matrix type
 * @tparam TMatrixGP Mini matrix type
 * @tparam Scalar Coefficient type of returned hessian
 * @param HF \f$ d^2 \times d^2 \f$ hessian of \f$ \Psi \f$ w.r.t. vectorized jacobian \f$
 * \text{vec}(\mathbf{F}) \f$, i.e. \f$ \frac{\partial^2 \Psi}{\partial \text{vec}(\mathbf{F})^2}
 * \f$
 * @param GP \f$ * \times d \f$ basis function gradients \f$ \nabla \mathbf{N}_i \f$
 * @param i Basis function index
 * @param j Basis function index
 * @return \f$ d \times d \f$ matrix \f$ \frac{\partial^2 \Psi}{\partial \mathbf{x}_i \partial
 * \mathbf{x}_j} \f$
 *
 */
template <
    CElement TElement,
    int Dims,
    math::linalg::mini::CMatrix TMatrixHF,
    math::linalg::mini::CMatrix TMatrixGP,
    class ScalarType = typename TMatrixHF::ScalarType>
auto HessianBlockWrtDofs(TMatrixHF const& HF, TMatrixGP const& GP, auto i, auto j)
    -> math::linalg::mini::SMatrix<ScalarType, Dims, Dims>
{
    using namespace math::linalg::mini;
    SMatrix<ScalarType, Dims, Dims> d2Psidx2 = Zeros<ScalarType, Dims, Dims>{};
    for (auto kj = 0; kj < Dims; ++kj)
    {
        for (auto ki = 0; ki < Dims; ++ki)
        {
            d2Psidx2 += GP(i, ki) * GP(j, kj) * HF.template Slice<Dims, Dims>(ki * Dims, kj * Dims);
        }
    }
    return d2Psidx2;
}

/**
 * @brief Computes hessian w.r.t. FEM degrees of freedom \f$ x \f$ of scalar function \f$
 * \Psi(\mathbf{F}) \f$, where \f$ F \f$ is the jacobian of \f$ x \f$, via chain rule. This is
 * effectively a rank-4 to rank-2 tensor contraction.
 *
 * Let \f$ \mathbf{g}_i \f$ be the \f$ i^{\text{th}} \f$ basis function gradient, then
 *
 * \f[
 * \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i} = \frac{\partial}{\partial
 * \mathbf{x}_i} \mathbf{x}_i \mathbf{g}_i^T, \mathbf{x}_i \in R^d, \mathbf{g}_i \in R^d
 * \f]
 *
 * Thus,
 * \f[
 * \frac{\partial \text{vec}(\mathbf{F})}{\partial \mathbf{x}_i} = \mathbf{g}_i \otimes
 * \mathbf{I}_{d x d} \in R^{d^2 \times d}
 * \f]
 *
 * With \f$ \mathbf{H}_F = \nabla^2_{\text{vec}(\mathbf{F})} \Psi \in \mathbb{R}^{d^2 \times d^2}
 * \f$, and \f$ \mathbf{H}_F^{uv} \f$ the \f$ (u,v)^\text{th} \; d \times d \f$ block of \f$
 * \mathbf{H}_F \f$, then
 *
 * \f[
 * \frac{\partial^2 \Psi}{\partial \mathbf{x}_i \partial \mathbf{x}_j}
 * = \frac{\partial^2 \Psi}{\partial \text{vec}(\mathbf{F})^2} \frac{\partial^2
 * \text{vec}(\mathbf{F})}{\partial \mathbf{x}_i \partial \mathbf{x}_j}
 * = \sum_{u=1}^{d} \sum_{v=1}^{d} \mathbf{H}_F^{uv} \mathbf{g}_{iu} \mathbf{g}_{jv}
 * \f]
 *
 * @tparam TElement FEM element type
 * @tparam Dims Problem dimensionality
 * @tparam TMatrixHF Mini matrix type
 * @tparam TMatrixGP Mini matrix type
 * @tparam Scalar Coefficient type of returned hessian
 * @param HF \f$ d^2 \times d^2 \f$ hessian of \f$ \Psi \f$ w.r.t. vectorized jacobian \f$
 * \text{vec}(\mathbf{F}) \f$, i.e. \f$ \frac{\partial^2 \Psi}{\partial \text{vec}(\mathbf{F})^2}
 * \f$
 * @param GP \f$ * \times d \f$ basis function gradients \f$ \nabla \mathbf{N}_i \f$
 * @return \f$ d^2 \times d^2 \f$ matrix \f$ \frac{\partial^2 \Psi}{\partial \mathbf{x}^2} \f$
 *
 */
template <
    CElement TElement,
    int Dims,
    math::linalg::mini::CMatrix TMatrixHF,
    math::linalg::mini::CMatrix TMatrixGP,
    class ScalarType = typename TMatrixHF::ScalarType>
auto HessianWrtDofs(TMatrixHF const& HF, TMatrixGP const& GP)
    -> math::linalg::mini::SMatrix<ScalarType, TElement::kNodes * Dims, TElement::kNodes * Dims>
{
    auto constexpr kRows = TElement::kNodes * Dims;
    auto constexpr kCols = TElement::kNodes * Dims;
    using namespace math::linalg::mini;
    SMatrix<ScalarType, kRows, kCols> d2Psidx2 = Zeros<ScalarType, kRows, kCols>{};
    for (auto kj = 0; kj < Dims; ++kj)
    {
        for (auto ki = 0; ki < Dims; ++ki)
        {
            for (auto j = 0; j < TElement::kNodes; ++j)
            {
                for (auto i = 0; i < TElement::kNodes; ++i)
                {
                    d2Psidx2.template Slice<Dims, Dims>(i * Dims, j * Dims) +=
                        GP(i, ki) * GP(j, kj) * HF.template Slice<Dims, Dims>(ki * Dims, kj * Dims);
                }
            }
        }
    }
    return d2Psidx2;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_DEFORMATIONGRADIENT_H
