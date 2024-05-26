#ifndef PBA_CORE_FEM_JACOBIAN_H
#define PBA_CORE_FEM_JACOBIAN_H

#include "Concepts.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Profiling.h"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <tbb/parallel_for.h>

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
        throw std::invalid_argument("Inverted or singular jacobian");
    }
    return detJ;
}

template <int QuadratureOrder, CMesh TMesh>
MatrixX DeterminantOfJacobian(TMesh const& mesh)
{
    PBA_PROFILE_SCOPE;

    using ElementType        = typename TMesh::ElementType;
    using AffineElementType  = typename ElementType::AffineBaseType;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;

    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<ElementType::kDims>();

    auto const numberOfElements = mesh.E.cols();
    MatrixX detJe(QuadratureRuleType::kPoints, numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = TMesh::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        if constexpr (AffineElementType::bHasConstantJacobian)
        {
            Scalar const detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));
            detJe.col(e).setConstant(detJ);
        }
        else
        {
            auto const wg = common::ToEigen(QuadratureRuleType::weights);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                Scalar const detJ =
                    DeterminantOfJacobian(Jacobian<AffineElementType>(Xg.col(g), Ve));
                detJe(g, e) = detJ;
            }
        }
    });
    return detJe;
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_JACOBIAN_H