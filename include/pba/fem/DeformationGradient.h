#ifndef PBA_CORE_FEM_DEFORMATION_GRADIENT_H
#define PBA_CORE_FEM_DEFORMATION_GRADIENT_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"

#include <Eigen/QR>
#include <Eigen/SVD>

namespace pba {
namespace fem {

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

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_DEFORMATION_GRADIENT_H