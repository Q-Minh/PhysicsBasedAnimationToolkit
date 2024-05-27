#ifndef PBA_CORE_FEM_SHAPE_FUNCTION_GRADIENTS_H
#define PBA_CORE_FEM_SHAPE_FUNCTION_GRADIENTS_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"

#include <Eigen/SVD>

namespace pba {
namespace fem {

/**
 * @brief Computes gradients of FEM basis functions in reference element
 * @tparam TDerivedXi
 * @tparam TDerivedX
 * @tparam TElement
 * @param Xi Point in reference element at which to evaluate the gradients
 * @param X Element vertices, i.e. nodes of affine element
 * @return |#nodes|x|Dims| matrix of basis function gradients in rows
 */
template <CElement TElement, class TDerivedXi, class TDerivedX>
Matrix<TElement::kNodes, TDerivedX::RowsAtCompileTime> ShapeFunctionGradients(
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

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_SHAPE_FUNCTION_GRADIENTS_H
