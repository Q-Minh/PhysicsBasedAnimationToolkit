#ifndef PBA_CORE_FEM_JACOBIAN_H
#define PBA_CORE_FEM_JACOBIAN_H

#include "Concepts.h"
#include "pba/aliases.h"

#include <Eigen/SVD>
#include <cmath>

namespace pba {
namespace fem {

template <CElement TElement, class TDerived>
[[maybe_unused]] Matrix<TDerived::RowsAtCompileTime, TElement::kDims>
Jacobian(Vector<TElement::kDims> const& X, Eigen::DenseBase<TDerived> const& x)
{
    static_assert(TDerived::RowsAtCompileTime != Eigen::Dynamic);
    assert(x.cols() == TElement::kNodes);
    auto constexpr kDimsOut                   = TDerived::RowsAtCompileTime;
    Matrix<kDimsOut, TElement::kDims> const J = x * GradN(X);
    return J;
}

template <class TDerived>
[[maybe_unused]] Scalar DeterminantOfJacobian(Eigen::MatrixBase<TDerived> const& J)
{
    bool constexpr bIsSquare = J.rows() == J.cols();
    if constexpr (bIsSquare)
    {
        return J.determinant();
    }
    else
    {
        return std::abs(J.jacobiSvd().singularValues().prod());
    }
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_JACOBIAN_H