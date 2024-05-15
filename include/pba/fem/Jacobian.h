#ifndef PBA_CORE_FEM_JACOBIAN_H
#define PBA_CORE_FEM_JACOBIAN_H

#include "Concepts.h"
#include "pba/aliases.h"

#include <Eigen/LU>
#include <Eigen/SVD>

namespace pba {
namespace fem {

template <CElement TElement, class TDerived>
[[maybe_unused]] Matrix<TDerived::RowsAtCompileTime, TElement::kDims>
Jacobian(Vector<TElement::kDims> const& X, Eigen::MatrixBase<TDerived> const& x)
{
    assert(x.cols() == TElement::kNodes);
    auto constexpr kDimsOut                   = TDerived::RowsAtCompileTime;
    Matrix<kDimsOut, TElement::kDims> const J = x * TElement::GradN(X);
    return J;
}

template <class TDerived>
[[maybe_unused]] Scalar DeterminantOfJacobian(Eigen::MatrixBase<TDerived> const& J)
{
    bool const bIsSquare = J.rows() == J.cols();
    Scalar const detJ    = bIsSquare ? J.determinant() : J.jacobiSvd().singularValues().prod();
    // TODO: Should define a numerical zero somewhere
    if (detJ <= 0.)
    {
        throw std::runtime_error("Inverted or singular jacobian");
    }
    return detJ;
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_JACOBIAN_H