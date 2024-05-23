#ifndef PBA_CORE_FEM_DEFORMATION_GRADIENT_H
#define PBA_CORE_FEM_DEFORMATION_GRADIENT_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"

#include <Eigen/SVD>

namespace pba {
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

template <CElement TElement, class TDerivedXi, class TDerivedX>
Matrix<TElement::kNodes, TDerivedX::RowsAtCompileTime> BasisFunctionGradients(
    Eigen::MatrixBase<TDerivedXi> const& Xi,
    Eigen::MatrixBase<TDerivedX> const& X)
{
    // \phi(X) = N(J^{-1} X) = N(\Xi)
    // grad_X \phi(X) = d N(\Xi) / d\Xi d \Xi / dX
    //                = grad_\Xi N * J^{-1}
    // If we transpose that equation, we get
    // [ grad_X \phi(X) ]^T = J^{-T} * grad_\Xi N^T
    // Recall that the pseudoinverse of J is J^{-1} = U \Sigma^{-1} V^T
    // We pseudoinvert its transpose directly, J^{-T} = V \Sigma^{-1} U^T
    auto constexpr kInputDims                     = TElement::kDims;
    auto constexpr kOutputDims                    = TDerivedX::RowsAtCompileTime;
    using AffineElementType                       = typename TElement::AffineBaseType;
    Matrix<TElement::kNodes, kInputDims> const GN = TElement::GradN(Xi);
    Matrix<kInputDims, kOutputDims> const JT      = Jacobian<AffineElementType>(Xi, X).transpose();
    auto JinvT = JT.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix<TElement::kNodes, kOutputDims> const GP = JinvT.solve(GN.transpose()).transpose();
    return GP;
}

/**
 * @brief
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
 * @tparam Dims
 * @param GF
 * @param GP
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
 * @brief
 *
 *
 *
 * @tparam TDerivedF
 * @tparam TDerivedGP
 * @tparam TElement
 * @tparam Dims
 * @param HF
 * @param GP
 * @return
 */
template <CElement TElement, int Dims, class TDerivedF, class TDerivedGP>
Matrix<TElement::kNodes * Dims, TElement::kNodes * Dims>
HessianWrtDofs(Eigen::DenseBase<TDerivedF> const& HF, Eigen::DenseBase<TDerivedGP> const& GP)
{
    auto const kRows              = TElement::kNodes * Dims;
    auto const kCols              = TElement::kNodes * Dims;
    Matrix<kRows, kCols> d2Psidx2 = Matrix<kRows, kCols>::Zero();
    for (auto ki = 0; ki < Dims; ++ki)
        for (auto kj = 0; kj < Dims; ++kj)
            for (auto j = 0; j < TElement::kNodes; ++j)
                for (auto i = 0; i < TElement::kNodes; ++i)
                    d2Psidx2.block<Dims, Dims>(i * Dims, j * Dims) +=
                        GP(i, ki) * GP(i, kj) * HF.block<Dims, Dims>(ki * Dims, kj * Dims);
    return d2Psidx2;
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_DEFORMATION_GRADIENT_H