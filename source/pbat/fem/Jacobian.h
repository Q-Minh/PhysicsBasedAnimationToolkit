#ifndef PBAT_FEM_JACOBIAN_H
#define PBAT_FEM_JACOBIAN_H

#include "Concepts.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
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
    auto constexpr eps   = 1e-10;
    if (detJ <= eps)
    {
        throw std::invalid_argument("Inverted or singular jacobian");
    }
    return detJ;
}

template <int QuadratureOrder, CMesh TMesh>
MatrixX DeterminantOfJacobian(TMesh const& mesh)
{
    PBAT_PROFILE_NAMED_SCOPE("fem.DeterminantOfJacobian");

    using ElementType        = typename TMesh::ElementType;
    using AffineElementType  = typename ElementType::AffineBaseType;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;

    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .template bottomRows<ElementType::kDims>();

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

template <CElement TElement, class TDerivedX, class TDerivedx>
Vector<TElement::kDims> ReferencePosition(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedx> const& x,
    int const maxIterations = 5,
    Scalar const eps        = 1e-10)
{
    // We need to solve the inverse problem \argmin_\Xi f(\Xi) = 1/2 ||x(\Xi) - X||_2^2
    // to find the reference position \Xi that corresponds to the domain position X in the element
    // whose vertices are x.

    // We use Gauss-Newton iterations on \argmin f.
    // This gives the iteration dx^k = [H(\Xi^k)]^{-1} J_x(\Xi^k)^T (x(\Xi^k) - X),
    // where H(\Xi^k) = J_x(\Xi^k)^T J_x(\Xi^k).

    using ElementType = TElement;
    assert(x.cols() == ElementType::kNodes);
    auto constexpr kDims    = ElementType::kDims;
    auto constexpr kNodes   = ElementType::kNodes;
    auto constexpr kOutDims = Eigen::MatrixBase<TDerivedx>::RowsAtCompileTime;
    // Initial guess is element's barycenter.
    auto const coords = common::ToEigen(ElementType::Coordinates).reshaped(kDims, kNodes);
    auto const vertexLagrangePositions =
        (coords(Eigen::all, ElementType::Vertices).template cast<Scalar>() /
         static_cast<Scalar>(ElementType::kOrder));
    Vector<kDims> Xik =
        vertexLagrangePositions.rowwise().sum() / static_cast<Scalar>(ElementType::Vertices.size());

    Vector<kOutDims> rk = x * ElementType::N(Xik) - X;
    Matrix<kOutDims, kDims> J;
    Eigen::LDLT<Matrix<kDims, kDims>> LDLT;

    // If Jacobian is constant, let's precompute it
    if constexpr (ElementType::bHasConstantJacobian)
    {
        J = Jacobian<ElementType>(Xik, x);
        LDLT.compute(J.transpose() * J);
    }
    // Do up to maxIterations Gauss Newton iterations
    for (auto k = 0; k < maxIterations; ++k)
    {
        if (rk.template lpNorm<1>() <= eps)
            break;

        // Non-constant jacobians need to be updated
        if constexpr (not ElementType::bHasConstantJacobian)
        {
            J = Jacobian<ElementType>(Xik, x);
            LDLT.compute(J.transpose() * J);
        }

        Xik -= LDLT.solve(J.transpose() * rk);
        rk = x * ElementType::N(Xik) - X;
    }
    return Xik;
}

template <CMesh TMesh, class TDerivedE, class TDerivedX>
MatrixX ReferencePositions(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X,
    int maxIterations = 5,
    Scalar eps        = 1e-10)
{
    PBAT_PROFILE_NAMED_SCOPE("fem.ReferencePositions");
    using MeshType    = TMesh;
    using ElementType = typename MeshType::ElementType;
    MatrixX Xi(ElementType::kDims, E.size());
    tbb::parallel_for(Index{0}, Index{E.size()}, [&](Index k) {
        Index const e                     = E(k);
        auto const nodes                  = mesh.E.col(e);
        auto constexpr kOutDims           = MeshType::kDims;
        auto constexpr kNodes             = ElementType::kNodes;
        Matrix<kOutDims, kNodes> const Xe = mesh.X(Eigen::all, nodes);
        Vector<kOutDims> const Xk         = X.col(k);
        Xi.col(k) = ReferencePosition<ElementType>(Xk, Xe, maxIterations, eps);
    });
    return Xi;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_JACOBIAN_H
