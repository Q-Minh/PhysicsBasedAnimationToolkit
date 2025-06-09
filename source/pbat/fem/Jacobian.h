/**
 * @file Jacobian.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Functions to compute jacobians, their determinants, domain quadrature weights and mapping
 * domain to reference space.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_JACOBIAN_H
#define PBAT_FEM_JACOBIAN_H

#include "Concepts.h"
#include "pbat/common/Concepts.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat::fem {

/**
 * @brief Given a map \f$ x(\xi) = \sum_i x_i N_i(\xi) \f$, where \f$ \xi \in \Omega^{\text{ref}}
 * \f$ is in reference space coordinates, computes \f$ \nabla_\xi x \f$.
 *
 * @tparam TElement FEM element type
 * @tparam TDerived Eigen matrix expression for map coefficients
 * @param X Reference space coordinates \f$ \xi \f$
 * @param x Map coefficients \f$ \mathbf{x} \f$
 * @return \f$ \nabla_\xi x \f$
 */
template <
    CElement TElement,
    class TDerivedX,
    class TDerivedx,
    common::CFloatingPoint TScalar = typename TDerivedX::Scalar>
[[maybe_unused]] auto
Jacobian(Eigen::MatrixBase<TDerivedX> const& X, Eigen::MatrixBase<TDerivedx> const& x)
    -> Eigen::Matrix<TScalar, TDerivedx::RowsAtCompileTime, TElement::kDims>
{
    static_assert(
        TDerivedx::ColsAtCompileTime == TElement::kNodes,
        "x must have the same number of columns as the number of nodes in the element.");
    auto constexpr kDimsOut                                   = TDerivedx::RowsAtCompileTime;
    Eigen::Matrix<TScalar, kDimsOut, TElement::kDims> const J = x * TElement::GradN(X);
    return J;
}

/**
 * @brief Computes the determinant of a (potentially non-square) Jacobian matrix.
 *
 * If the Jacobian matrix is square, the determinant is computed directly, otherwise the singular
 * values \f$ \sigma_i \f$ are computed and their product \f$ \Pi_i \sigma_i \f$ is returned.
 *
 * @tparam TDerived Eigen matrix expression
 * @param J Jacobian matrix
 * @return Determinant of the Jacobian matrix
 */
template <class TDerived>
[[maybe_unused]] auto DeterminantOfJacobian(Eigen::MatrixBase<TDerived> const& J) ->
    typename TDerived::Scalar
{
    using ScalarType         = typename TDerived::Scalar;
    bool const bIsSquare     = J.rows() == J.cols();
    auto const detJ          = bIsSquare ? J.determinant() : J.jacobiSvd().singularValues().prod();
    ScalarType constexpr eps = ScalarType(1e-10);
    if (detJ <= eps)
    {
        throw std::invalid_argument("Inverted or singular jacobian");
    }
    return detJ;
}

/**
 * @brief Computes the determinant of the Jacobian matrix at element quadrature points.
 * @tparam TElement FEM element type
 * @tparam QuadratureOrder Quadrature order
 * @tparam TDerivedE Eigen matrix expression for element matrix
 * @tparam TDerivedX Eigen matrix expression for nodal positions
 * @param E `|# elem nodes| x |# elems|` element matrix
 * @param X `|# dims| x |# nodes|` mesh nodal position matrix
 * @return `|# elem quad.pts.| x |# elems|` matrix of jacobian determinants at element quadrature
 * points
 */
template <CElement TElement, int QuadratureOrder, class TDerivedE, class TDerivedX>
[[maybe_unused]] auto
DeterminantOfJacobian(Eigen::DenseBase<TDerivedE> const& E, Eigen::MatrixBase<TDerivedX> const& X)
    -> Eigen::Matrix<
        typename TDerivedX::Scalar,
        TElement::template QuadratureType<QuadratureOrder, typename TDerivedX::Scalar>::kPoints,
        Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.DeterminantOfJacobian");
    using ScalarType        = typename TDerivedX::Scalar;
    using ElementType       = TElement;
    using AffineElementType = typename ElementType::AffineBaseType;
    using QuadratureRuleType =
        typename ElementType::template QuadratureType<QuadratureOrder, ScalarType>;
    using MatrixType = Eigen::Matrix<ScalarType, QuadratureRuleType::kPoints, Eigen::Dynamic>;

    auto const Xg = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .template bottomRows<ElementType::kDims>();
    auto const numberOfElements = E.cols();
    MatrixType detJe(QuadratureRuleType::kPoints, numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                                   = E.col(e);
        auto const vertices                                = nodes(ElementType::Vertices);
        auto constexpr kRowsJ                              = TDerivedX::RowsAtCompileTime;
        auto constexpr kColsJ                              = AffineElementType::kNodes;
        Eigen::Matrix<ScalarType, kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        if constexpr (AffineElementType::bHasConstantJacobian)
        {
            ScalarType const detJ = DeterminantOfJacobian(
                Jacobian<AffineElementType>(Xg.col(0) /*arbitrary eval.pt.*/, Ve));
            detJe.col(e).setConstant(detJ);
        }
        else
        {
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                auto const detJ = DeterminantOfJacobian(Jacobian<AffineElementType>(Xg.col(g), Ve));
                detJe(g, e)     = detJ;
            }
        }
    });
    return detJe;
}

/**
 * @brief
 * @tparam QuadratureOrder
 * @tparam TMesh
 * @param mesh
 * @return \f$ \mathbf{D}^J \in \mathbb{R}^{|G^e| \times |E| } \f$ matrix of element jacobian
 * determinants at element quadrature points, where \f$ |G^e| \f$ is the number of quadrature points
 * per element and \f$ |E| \f$ is the number of elements.
 */
template <int QuadratureOrder, CMesh TMesh>
[[maybe_unused]] auto DeterminantOfJacobian(TMesh const& mesh) -> Eigen::Matrix<
    typename TMesh::ScalarType,
    TMesh::ElementType::template QuadratureType<QuadratureOrder, typename TMesh::ScalarType>::
        kPoints,
    Eigen::Dynamic>
{
    return DeterminantOfJacobian<typename TMesh::ElementType, QuadratureOrder>(mesh.E, mesh.X);
}

/**
 * @brief Computes the determinant of the Jacobian matrix at element quadrature points.
 *
 * @tparam TElement FEM element type
 * @tparam TDerivedEg Eigen matrix expression for element indices at evaluation points
 * @tparam TDerivedX Eigen matrix expression for mesh nodal positions
 * @tparam TDerivedXi Eigen matrix expression for evaluation points in reference space
 * @param Eg `|# elem nodes| x |# eval.pts.|` element indices at evaluation points
 * @param X `|# dims| x |# nodes|` mesh nodal position matrix
 * @param Xi `|# dims| x |# eval.pts.|` evaluation points in reference space
 * @return `|# eval.pts.| x 1` vector of jacobian determinants at evaluation points
 */
template <CElement TElement, class TDerivedEg, class TDerivedX, class TDerivedXi>
[[maybe_unused]] auto DeterminantOfJacobian(
    Eigen::DenseBase<TDerivedEg> const& Eg,
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::Vector<typename TDerivedX::Scalar, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.DeterminantOfJacobian");
    using ScalarType        = typename TDerivedX::Scalar;
    using ElementType       = TElement;
    using AffineElementType = typename ElementType::AffineBaseType;
    Eigen::Vector<ScalarType, Eigen::Dynamic> detJe(Eg.cols());
    tbb::parallel_for(Index{0}, Index{Eg.cols()}, [&](Index g) {
        auto const nodes                                   = Eg.col(g);
        auto const vertices                                = nodes(ElementType::Vertices);
        auto constexpr kRowsJ                              = TDerivedX::RowsAtCompileTime;
        auto constexpr kColsJ                              = AffineElementType::kNodes;
        Eigen::Matrix<ScalarType, kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        ScalarType const detJ = DeterminantOfJacobian(Jacobian<AffineElementType>(Xi.col(g), Ve));
        detJe(g)              = detJ;
    });
    return detJe;
}

/**
 * @brief Computes the determinant of the Jacobian matrix at element quadrature points.
 *
 * @tparam TElement FEM element type
 * @tparam TDerivedE Eigen matrix expression for element matrix
 * @tparam TDerivedX Eigen matrix expression for mesh nodal positions
 * @tparam TDerivedeg Eigen matrix expression for element indices at evaluation points
 * @tparam TDerivedXi Eigen matrix expression for evaluation points in reference space
 * @param E `|# elem nodes| x |# elems|` mesh element matrix
 * @param X `|# dims| x |# nodes|` mesh nodal position matrix
 * @param eg `|# eval.pts.|` element indices at evaluation points
 * @param Xi `|# dims| x |# eval.pts.|` evaluation points in reference space
 * @return `|# eval.pts.| x 1` vector of jacobian determinants at evaluation points
 */
template <CElement TElement, class TDerivedE, class TDerivedX, class TDerivedeg, class TDerivedXi>
[[maybe_unused]] auto DeterminantOfJacobian(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedXi> const& Xi)
    -> Eigen::Vector<typename TDerivedX::Scalar, Eigen::Dynamic>
{
    return DeterminantOfJacobian<TElement>(
        E(Eigen::placeholders::all, eg.derived()),
        X.derived(),
        Xi.derived());
}

/**
 * @brief Computes the reference position \f$ \xi \f$ such that \f$ x(\xi) = x \f$.
 * This inverse problem is solved using Gauss-Newton iterations.
 *
 * We need to solve the inverse problem
 * \f[
 * \min_\xi f(\xi) = 1/2 ||x(\xi) - X||_2^2
 * \f]
 * to find the reference position \f$ \xi \f$ that corresponds to the domain position \f$ X \f$ in
 * the element whose vertices are \f$ x \f$.
 *
 * We use Gauss-Newton iterations on \f$ \min f \f$.
 * This gives the iteration
 * \f[
 * dx^{k+1} = [H(\xi^k)]^{-1} J_x(\xi^k)^T (x(\xi^k) - X)
 * \f]
 * where \f$ H(\xi^k) = J_x(\xi^k)^T J_x(\xi^k) \f$.
 *
 * @tparam TElement FEM element type
 * @tparam TDerivedX Eigen matrix expression for reference positions
 * @tparam TDerivedx Eigen matrix expression for domain positions
 * @param X Reference positions \f$ \xi \f$
 * @param x Domain positions \f$ x \f$
 * @param maxIterations Maximum number of Gauss-Newton iterations
 * @param eps Convergence tolerance
 * @return Reference position \f$ \xi \f$
 */
template <
    CElement TElement,
    class TDerivedX,
    class TDerivedx,
    common::CFloatingPoint TScalar = typename TDerivedX::Scalar>
auto ReferencePosition(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedx> const& x,
    int const maxIterations = 5,
    TScalar const eps       = 1e-10) -> Eigen::Vector<TScalar, TElement::kDims>
{
    using ScalarType = TScalar;
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedX::Scalar>,
        "X and x must have given scalar type");
    static_assert(
        std::is_same_v<ScalarType, typename TDerivedx::Scalar>,
        "X and x must have given scalar type");
    using ElementType = TElement;
    assert(x.cols() == ElementType::kNodes);
    auto constexpr kDims    = ElementType::kDims;
    auto constexpr kNodes   = ElementType::kNodes;
    auto constexpr kOutDims = Eigen::MatrixBase<TDerivedx>::RowsAtCompileTime;
    // Initial guess is element's barycenter.
    auto const coords = common::ToEigen(ElementType::Coordinates).reshaped(kDims, kNodes);
    auto const vertexLagrangePositions =
        (coords(Eigen::placeholders::all, ElementType::Vertices).template cast<ScalarType>() /
         static_cast<ScalarType>(ElementType::kOrder));
    Eigen::Vector<ScalarType, kDims> Xik = vertexLagrangePositions.rowwise().sum() /
                                           static_cast<ScalarType>(ElementType::Vertices.size());

    Eigen::Vector<ScalarType, kOutDims> rk = x * ElementType::N(Xik) - X;
    Eigen::Matrix<ScalarType, kOutDims, kDims> J;
    Eigen::LDLT<Eigen::Matrix<ScalarType, kDims, kDims>> LDLT;

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

/**
 * @brief Computes reference positions \f$ \xi \f$ such that \f$ X(\xi) = X_n \f$ for every point in
 * \f$ \mathbf{X} \in \mathbb{R}^{d \times n} \f$.
 *
 * @tparam TElement FEM element type
 * @tparam TDerivedEg Eigen matrix expression for element indices at evaluation points
 * @tparam TDerivedX Eigen matrix expression for mesh node positions
 * @tparam TDerivedXg Eigen matrix expression for evaluation points
 * @param Eg `|# elem nodes| x |# eval. pts.|` matrix of element indices at evaluation points
 * @param X `|# dims| x |# nodes|` matrix of mesh node positions
 * @param Xg `|# dims| x |# eval. pts.|` matrix of evaluation points in domain space
 * @param maxIterations Maximum number of Gauss-Newton iterations
 * @param eps Convergence tolerance
 * @return `|# element dims| x n` matrix of reference positions associated with domain points X in
 * corresponding elements E
 */
template <
    CElement TElement,
    class TDerivedEg,
    class TDerivedX,
    class TDerivedXg,
    common::CFloatingPoint TScalar = typename TDerivedXg::Scalar>
auto ReferencePositions(
    Eigen::DenseBase<TDerivedEg> const& Eg,
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    int maxIterations = 5,
    TScalar eps       = 1e-10) -> Eigen::Matrix<TScalar, TElement::kDims, Eigen::Dynamic>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ReferencePositions");
    static_assert(
        std::is_same_v<TScalar, typename TDerivedX::Scalar>,
        "X must have given scalar type");
    static_assert(
        std::is_same_v<TScalar, typename TDerivedXg::Scalar>,
        "Xg must have given scalar type");
    using ElementType = TElement;
    using ScalarType  = TScalar;
    using MatrixType  = Eigen::Matrix<ScalarType, ElementType::kDims, Eigen::Dynamic>;
    MatrixType Xi(ElementType::kDims, Eg.cols());
    tbb::parallel_for(Index{0}, Index{Eg.cols()}, [&](Index g) {
        auto const nodes                                     = Eg.col(g);
        auto constexpr kOutDims                              = TDerivedX::RowsAtCompileTime;
        auto constexpr kNodes                                = ElementType::kNodes;
        Eigen::Matrix<ScalarType, kOutDims, kNodes> const Xe = X(Eigen::placeholders::all, nodes);
        Eigen::Vector<ScalarType, kOutDims> const Xk         = Xg.col(g);
        Xi.col(g) = ReferencePosition<ElementType>(Xk, Xe, maxIterations, eps);
    });
    return Xi;
}

/**
 * @brief
 * @tparam TElement FEM element type
 * @tparam TDerivedE Eigen matrix expression for elements
 * @tparam TDerivedX Eigen matrix expression for nodal positions
 * @tparam TDerivedEg Eigen matrix expression for indices of elements at evaluation points
 * @tparam TDerivedXg Eigen matrix expression for evaluation points in domain space
 * @param E `|# elem nodes| x |# elems|` element matrix
 * @param X `|# dims| x |# nodes|` nodal position matrix
 * @param eg `|# eval.pts.| x 1` Indices of elements at evaluation points
 * @param Xg `|# dims| x |# eval.pts.|` evaluation points in domain space
 * @param maxIterations Maximum number of Gauss-Newton iterations
 * @param eps Convergence tolerance
 * @return `|# element dims| x n` matrix of reference positions associated with domain points
 */
template <CElement TElement, class TDerivedE, class TDerivedX, class TDerivedEg, class TDerivedXg>
auto ReferencePositions(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    int maxIterations               = 5,
    typename TDerivedXg::Scalar eps = 1e-10)
    -> Eigen::Matrix<typename TDerivedXg::Scalar, TElement::kDims, Eigen::Dynamic>
{
    return ReferencePositions<TElement>(
        E(Eigen::placeholders::all, eg.derived()),
        X.derived(),
        Xg.derived(),
        maxIterations,
        eps);
}

/**
 * @brief Computes reference positions \f$ \xi \f$ such that \f$ X(\xi) = X_n \f$ for every point in
 * \f$ \mathbf{X} \in \mathbb{R}^{d \times n} \f$.
 *
 * @tparam TMesh FEM mesh type
 * @tparam TDerivedEg Eigen matrix expression for indices of elements
 * @tparam TDerivedXg Eigen matrix expression for evaluation points
 * @param mesh FEM mesh
 * @param eg `|# eval.pts.| x 1` indices of elements at evaluation points
 * @param Xg `|# dims| x |# eval.pts.|` evaluation points in domain space
 * @param maxIterations Maximum number of Gauss-Newton iterations
 * @param eps Convergence tolerance
 * @return `|# element dims| x |# eval.pts.|` matrix of reference positions associated with domain
 * points X in corresponding elements E
 * @pre `E.size() == X.cols()`
 */
template <CMesh TMesh, class TDerivedEg, class TDerivedXg>
auto ReferencePositions(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedXg> const& Xg,
    int maxIterations              = 5,
    typename TMesh::ScalarType eps = 1e-10)
    -> Eigen::Matrix<typename TMesh::ScalarType, TMesh::ElementType::kDims, Eigen::Dynamic>
{
    return ReferencePositions<typename TMesh::ElementType>(
        mesh.E(Eigen::placeholders::all, eg.derived()),
        mesh.X,
        Xg.derived(),
        maxIterations,
        eps);
}

} // namespace pbat::fem

#endif // PBAT_FEM_JACOBIAN_H
