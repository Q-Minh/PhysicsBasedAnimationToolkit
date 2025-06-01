/**
 * @file Gradient.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM gradient operator
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_GRADIENT_H
#define PBAT_FEM_GRADIENT_H

#include "Concepts.h"
#include "ShapeFunctions.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Represents an FEM gradient operator \f$ \mathbf{G} \in \mathbb{R}^{dg \times n} \f$
 *
 * @details
 * Given a function \f$ u(X) \f$ discretized at mesh nodes \f$ X_i \f$
 * \f[
 * \nabla u(X) = \sum_j u_j \nabla \phi_j(X) \; \forall \; e \in E
 * \f]
 * where \f$ \nabla \phi_j(X) \f$ is the gradient of the shape function \f$ \phi_j(X) \f$ at element
 * \f$ e \f$.
 *
 * The gradient operator \f$ \mathbf{G} \f$ is thus defined as the matrix
 * \f[
 * \mathbf{G}^k = \begin{bmatrix} \frac{\partial \phi_i}{\partial X_k} \end{bmatrix} \in
 * \mathbb{R}^{n \times n}
 * \f]
 *
 * where \f$ n \f$ is the number of nodes in the mesh and \f$ 1 \leq k \leq d \f$ is a spatial
 * dimension of the domain.
 * \f[
 * \mathbf{G} = \begin{bmatrix} \mathbf{G}^1 \\ \vdots \\ \mathbf{G}^d \end{bmatrix} \in
 * \mathbb{R}^{dg \times n}
 * \f]
 *
 * where \f$ g \f$ is the number of evaluation points.
 *
 * @note Link to my higher-level FEM crash course doc.
 *
 * @tparam TMesh
 */
template <CMesh TMesh>
struct Gradient
{
  public:
    using SelfType              = Gradient<TMesh>;             ///< Self type
    using MeshType              = TMesh;                       ///< Mesh type
    using ElementType           = typename TMesh::ElementType; ///< Element type
    static int constexpr kDims  = MeshType::kDims; ///< Number of domain spatial dimensions
    static int constexpr kOrder = (ElementType::kOrder > 1) ?
                                      (ElementType::kOrder - 1) :
                                      1; ///< Polynomial order of the gradient operator

    /**
     * @brief
     * @param mesh The finite element mesh
     * @param eg |# quad.pts.| x 1 array of element indices associated with quadrature points
     * @param GNeg |# element nodes|x|# dims * # quad.pts.| matrix of element shape function
     * gradients at quadrature points points
     */
    Gradient(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> eg,
        Eigen::Ref<MatrixX const> const& GNeg);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies the gradient matrix as a linear operator on x, adding result to y.
     *
     * @tparam TDerivedIn Input matrix type
     * @tparam TDerivedOut Output matrix type
     * @param x Input matrix
     * @param y Output matrix
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free Gradient operator into sparse compressed format.
     * @return Sparse compressed column matrix representation of the Gradient operator
     */
    CSCMatrix ToMatrix() const;

    /**
     * @brief Returns the number columns
     * @return Number of columns
     */
    Index InputDimensions() const { return mesh.X.cols(); }
    /**
     * @brief Returns the number of rows
     * @return Number of rows
     */
    Index OutputDimensions() const { return kDims * eg.size(); }

    /**
     * @brief Checks the validity of the gradient operator
     */
    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh

    Eigen::Ref<IndexVectorX const>
        eg; ///< `|# quad.pts.|x1` array of element indices corresponding to quadrature points
    Eigen::Ref<MatrixX const> GNeg; ///< `|# element nodes|x|# dims * # quad.pts. * # elements|`
                                    ///< matrix of element shape function gradients at quadrature
                                    ///< points. See ShapeFunctionGradients().
};

/**
 * @brief Assembled gradient matrix
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedEg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TScalar Scalar type of the gradient matrix
 * @tparam TIndex Index type of the gradient matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh element
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.|` vector of element indices at quadrature points
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @return Sparse matrix representation of the gradient operator
 */
template <
    class TDerivedE,
    class TDerivedEg,
    class TDerivedGNeg,
    common::CFloatingPoint TScalar = typename TDerivedGNeg::Scalar,
    common::CIndex TIndex          = typename TDerivedEg::Scalar>
auto GradientMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Index nNodes,
    Eigen::DenseBase<TDerivedEg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
    -> Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GradientMatrix");
    auto const kDims       = GNeg.cols() / eg.size();
    using SparseMatrixType = Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex>;
    SparseMatrixType G(kDims * nNodes, nNodes);
    using Triplet = Eigen::Triplet<TScalar, TIndex>;
    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(GNeg.size()));
    auto const numberOfQuadraturePoints = eg.size();
    auto const kNodesPerElement         = E.rows();
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const Geg   = GNeg.block(0, g * kDims, kNodesPerElement, kDims);
        for (auto d = 0; d < kDims; ++d)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                auto const ni = static_cast<TIndex>(d * numberOfQuadraturePoints + g);
                auto const nj = static_cast<TIndex>(nodes(j));
                triplets.emplace_back(ni, nj, Geg(j, d));
            }
        }
    }
    return G;
}

template <CMesh TMesh>
template <class TDerivedIn, class TDerivedOut>
inline void Gradient<TMesh>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.Gradient.Apply");
    // Check inputs
    bool const bDimensionsMatch = (x.cols() == y.cols()) and (x.rows() == InputDimensions()) and
                                  (y.rows() == OutputDimensions());
    if (not bDimensionsMatch)
    {
        std::string const what = fmt::format(
            "Expected input to have rows={} and output to have rows={}, and same number of "
            "columns, but got dimensions "
            "x,y=({},{}), ({},{})",
            InputDimensions(),
            OutputDimensions(),
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }
    // Compute gradient
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto const numberOfQuadraturePoints = eg.size();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto g = 0; g < numberOfQuadraturePoints; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.col(c)(nodes);
            auto const Geg   = GNeg.block<kNodesPerElement, kDims>(0, g * kDims);
            for (auto d = 0; d < kDims; ++d)
            {
                y(d * numberOfQuadraturePoints + g) +=
                    Geg(Eigen::placeholders::all, d).transpose() * xe;
            }
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_GRADIENT_H