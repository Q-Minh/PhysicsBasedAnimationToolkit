#ifndef PBAT_FEM_DEFORMATION_GRADIENT_H
#define PBAT_FEM_DEFORMATION_GRADIENT_H

#include "Concepts.h"
#include "Jacobian.h"

#include <Eigen/SVD>
#include <pbat/aliases.h>

namespace pbat {
namespace fem {

/**
 * @brief Computes the deformation gradient dx(X)/dX of the deformation map x(X).
 *
 * If the problem is discretized with displacement coefficients u = x(X) - X,
 * then simply feed this function with argument x = X + u.
 *
 * @tparam TDerivedU
 * @tparam TDerivedX
 * @tparam TElement
 * @param x Matrix of column-wise position nodal coefficients
 * @param GP Basis function gradients
 * @return
 */
template <CElement TElement, class TDerivedx, class TDerivedX>
Matrix<TDerivedx::RowsAtCompileTime, TElement::kDims>
DeformationGradient(Eigen::MatrixBase<TDerivedx> const& x, Eigen::MatrixBase<TDerivedX> const& GP)
{
    return x * GP;
}

/**
 * @brief Computes gradient w.r.t. FEM degrees of freedom x of some function of deformation gradient
 * F, U(F(x)) via chain rule. This is effectively a rank-3 to rank-1 tensor contraction.
 *
 * dF/dxi = d/dxi xi gi^T, xi \in R^d, gi \in R^d
 *
 * dvec(F)/dxi = gi \kronecker I_{d x d} \in R^{d^2 x d}
 *
 * dPsi/dxi = dPsi/dvec(F) * dvec(F)/dxi
 *        = [ dp1_{d x 1} --- dpd_{d x 1} ] [  gi1 I_{d x d}
 *                                                  |
 *                                             gid I_{d x d} ]
 *        = \sum_{k=1}^{d} dpk_{d x 1} * gik
 *
 * @tparam TDerivedF
 * @tparam TDerivedGP
 * @tparam TElement
 * @tparam Dims Problem dimensionality
 * @param GF Gradient w.r.t. vectorized deformation gradient vec(F)
 * @param GP Basis function gradients
 * @return
 */
template <CElement TElement, int Dims, class TDerivedF, class TDerivedGP>
Vector<TElement::kNodes * Dims>
GradientWrtDofs(Eigen::DenseBase<TDerivedF> const& GF, Eigen::DenseBase<TDerivedGP> const& GP)
{
    auto const kRows     = TElement::kNodes * Dims;
    Vector<kRows> dPsidx = Vector<kRows>::Zero();
    for (auto k = 0; k < Dims; ++k)
        for (auto i = 0; i < TElement::kNodes; ++i)
            dPsidx.segment<Dims>(i * Dims) += GP(i, k) * GF.segment<Dims>(k * Dims);
    return dPsidx;
}

/**
 * @brief Computes hessian w.r.t. FEM degrees of freedom x of some function of deformation gradient
 * F, U(F(x)) via chain rule. This is effectively a rank-4 to rank-2 tensor contraction.
 *
 * dPsi/dxi = \sum_{k=1}^{d} dpk_{d x 1} * gik (see GradientWrtDofs)
 * d^2 Psi / dxi dxj = \sum_{ki=1}^{d} \sum_{kj=1}^{d} d^2 gi_{ki} p_{ki,kj}  gj_{kj}
 *
 * @tparam TDerivedF
 * @tparam TDerivedGP
 * @tparam TElement
 * @tparam Dims Problem dimensionality
 * @param HF Hessian of energy w.r.t. vectorized deformation gradient vec(F)
 * @param GP Basis function gradients
 * @return
 */
template <CElement TElement, int Dims, class TDerivedF, class TDerivedGP>
Matrix<TElement::kNodes * Dims, TElement::kNodes * Dims>
HessianWrtDofs(Eigen::DenseBase<TDerivedF> const& HF, Eigen::DenseBase<TDerivedGP> const& GP)
{
    auto const kRows              = TElement::kNodes * Dims;
    auto const kCols              = TElement::kNodes * Dims;
    Matrix<kRows, kCols> d2Psidx2 = Matrix<kRows, kCols>::Zero();
    for (auto kj = 0; kj < Dims; ++kj)
        for (auto ki = 0; ki < Dims; ++ki)
            for (auto j = 0; j < TElement::kNodes; ++j)
                for (auto i = 0; i < TElement::kNodes; ++i)
                    d2Psidx2.block<Dims, Dims>(i * Dims, j * Dims) +=
                        GP(i, ki) * GP(j, kj) * HF.block<Dims, Dims>(ki * Dims, kj * Dims);
    return d2Psidx2;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_DEFORMATION_GRADIENT_H