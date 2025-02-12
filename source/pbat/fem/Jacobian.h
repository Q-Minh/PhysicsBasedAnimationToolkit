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

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

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
template <CElement TElement, class TDerived>
[[maybe_unused]] auto
Jacobian(Vector<TElement::kDims> const& X, Eigen::MatrixBase<TDerived> const& x)
    -> Matrix<TDerived::RowsAtCompileTime, TElement::kDims>
{
    assert(x.cols() == TElement::kNodes);
    auto constexpr kDimsOut                   = TDerived::RowsAtCompileTime;
    Matrix<kDimsOut, TElement::kDims> const J = x * TElement::GradN(X);
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
MatrixX DeterminantOfJacobian(TMesh const& mesh)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.DeterminantOfJacobian");

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
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::placeholders::all, vertices);
        if constexpr (AffineElementType::bHasConstantJacobian)
        {
            Scalar const detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));
            detJe.col(e).setConstant(detJ);
        }
        else
        {
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

/**
 * @brief Computes the inner product weights \f$ \mathbf{w}_{ge} \in \mathbb{R}^{|G^e| \times |E|}
 * \f$ such that \f$ \int_\Omega \cdot d\Omega = \sum_e \sum_g w_{ge} \cdot \f$.
 *
 * In other words, \f$ w_{ge} = w_g \det(J^e_g) \f$ where \f$ J^e_g \f$ is the Jacobian of the
 * element map at the \f$ g^\text{{th} \f$ quadrature point and \f$ w_g \f$ is the \f$ g^\text{th}
 * \f$ quadrature weight.
 *
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @param mesh FEM mesh
 * @return `|# quad.pts.|x|# elements|` matrix of quadrature weights multiplied by jacobian
 * determinants at element quadrature points
 */
template <int QuadratureOrder, CMesh TMesh>
MatrixX InnerProductWeights(TMesh const& mesh)
{
    MatrixX detJe            = DeterminantOfJacobian<QuadratureOrder>(mesh);
    using ElementType        = typename TMesh::ElementType;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto const wg            = common::ToEigen(QuadratureRuleType::weights);
    detJe.array().colwise() *= wg.array();
    return detJe;
}

/**
 * @brief Computes the inner product weights \f$ \mathbf{w}_{ge} \in \mathbb{R}^{|G^e| \times |E|}
 * \f$ such that \f$ \int_\Omega \cdot d\Omega = \sum_e \sum_g w_{ge} \cdot \f$.
 *
 * @tparam QuadratureOrder Quadrature order
 * @tparam TMesh Mesh type
 * @tparam TDerivedDetJe Eigen matrix expression for jacobian determinants at quadrature points
 * @param mesh FEM mesh
 * @param detJe Matrix of jacobian determinants at element quadrature points
 * @return `|# quad.pts.|x|# elements|` matrix of quadrature weights multiplied by jacobian
 * determinants at element quadrature points
 */
template <int QuadratureOrder, CMesh TMesh, class TDerivedDetJe>
MatrixX InnerProductWeights(TMesh const& mesh, Eigen::MatrixBase<TDerivedDetJe> const& detJe)
{
    using ElementType        = typename TMesh::ElementType;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto const wg            = common::ToEigen(QuadratureRuleType::weights);
    if (wg.size() != detJe.rows() or detJe.cols() != mesh.E.cols())
    {
        std::string const what = fmt::format(
            "detJe of invalid dimensions {}x{}, expected {}x{}",
            detJe.rows(),
            detJe.cols(),
            wg.size(),
            mesh.E.cols());
        throw std::invalid_argument(what);
    }
    MatrixX Ihat = detJe;
    Ihat.array().colwise() *= wg.array();
    return Ihat;
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
template <CElement TElement, class TDerivedX, class TDerivedx>
Vector<TElement::kDims> ReferencePosition(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedx> const& x,
    int const maxIterations = 5,
    Scalar const eps        = 1e-10)
{
    using ElementType = TElement;
    assert(x.cols() == ElementType::kNodes);
    auto constexpr kDims    = ElementType::kDims;
    auto constexpr kNodes   = ElementType::kNodes;
    auto constexpr kOutDims = Eigen::MatrixBase<TDerivedx>::RowsAtCompileTime;
    // Initial guess is element's barycenter.
    auto const coords = common::ToEigen(ElementType::Coordinates).reshaped(kDims, kNodes);
    auto const vertexLagrangePositions =
        (coords(Eigen::placeholders::all, ElementType::Vertices).template cast<Scalar>() /
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

/**
 * @brief Computes reference positions \f$ \xi \f$ such that \f$ X(\xi) = X_n \f$ for every point in
 * \f$ \mathbf{X} \in \mathbb{R}^{d \times n} \f$.
 *
 * @tparam TDerivedE Eigen matrix expression for indices of elements
 * @tparam TDerivedX Eigen matrix expression for domain positions
 * @tparam TMesh FEM mesh type
 * @param mesh FEM mesh
 * @param E Indices of elements
 * @param X Domain positions
 * @param maxIterations Maximum number of Gauss-Newton iterations
 * @param eps Convergence tolerance
 * @return `|# element dims| x n` matrix of reference positions associated with domain points
 * X in corresponding elements E
 * @pre `E.size() == X.cols()`
 */
template <CMesh TMesh, class TDerivedE, class TDerivedX>
MatrixX ReferencePositions(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::MatrixBase<TDerivedX> const& X,
    int maxIterations = 5,
    Scalar eps        = 1e-10)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.ReferencePositions");
    using MeshType    = TMesh;
    using ElementType = typename MeshType::ElementType;
    MatrixX Xi(ElementType::kDims, E.size());
    tbb::parallel_for(Index{0}, Index{E.size()}, [&](Index k) {
        Index const e                     = E(k);
        auto const nodes                  = mesh.E.col(e);
        auto constexpr kOutDims           = MeshType::kDims;
        auto constexpr kNodes             = ElementType::kNodes;
        Matrix<kOutDims, kNodes> const Xe = mesh.X(Eigen::placeholders::all, nodes);
        Vector<kOutDims> const Xk         = X.col(k);
        Xi.col(k) = ReferencePosition<ElementType>(Xk, Xe, maxIterations, eps);
    });
    return Xi;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_JACOBIAN_H
