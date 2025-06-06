/**
 * @file Gradient.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM gradient operator
 * @date 2025-02-11
 *
 * @details Given a function \f$ u(X) \f$ discretized at mesh nodes \f$ X_i \f$,
 * \f[
 * \nabla u(X) = \sum_j u_j \nabla \phi_j(X) \; \forall \; e \in E
 * \f]
 * where \f$ \nabla \phi_j(X) \f$ is the gradient of the shape function \f$ \phi_j(X) \f$ at element
 * \f$ e \f$. The gradient operator \f$ \mathbf{G} \in \mathbb{R}^{d|Q| \times n} \f$ maps the nodal
 * values to the gradient at quadrature points \f$ Q \f$ in dimensions \f$ d \f$.
 *
 * This file provides functions to compute gradient operator related quantities.
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_GRADIENT_H
#define PBAT_FEM_GRADIENT_H

#include "Concepts.h"
#include "ShapeFunctions.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <exception>
#include <fmt/core.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Concept for matrix-free gradient parameter types
 * @tparam T Type to check
 */
template <typename T>
concept CMatrixFreeGradient = requires(T G)
{
    typename T::ElementType;
    typename T::IndexType;
    typename T::ScalarType;
    {T::kDims}->std::convertible_to<int>;
    {G.E};
    {G.nNodes};
    {G.eg};
    {G.GNeg};
};

/**
 * @brief Parameters for gradient operator computations
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 */
template <CElement TElement, int Dims, class TDerivedE, class TDerivedeg, class TDerivedGNeg>
struct MatrixFreeGradient
{
    using ElementType = TElement;                      ///< Element type
    using IndexType   = typename TDerivedE::Scalar;    ///< Index type (usually Eigen::Index)
    using ScalarType  = typename TDerivedGNeg::Scalar; ///< Scalar type (usually double or float)
    static constexpr int kDims = Dims;                 ///< Number of spatial dimensions

    Eigen::DenseBase<TDerivedE> const& E;        ///< Element connectivity matrix
    IndexType nNodes;                            ///< Number of mesh nodes
    Eigen::DenseBase<TDerivedeg> const& eg;      ///< Element indices at quadrature points
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg; ///< Shape function gradients at quadrature points

    /**
     * @brief Construct gradient parameters
     * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
     * @param nNodes Number of mesh nodes
     * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
     * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
     * quadrature points
     */
    MatrixFreeGradient(
        Eigen::DenseBase<TDerivedE> const& E,
        IndexType nNodes,
        Eigen::DenseBase<TDerivedeg> const& eg,
        Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
        : E(E), nNodes(nNodes), eg(eg), GNeg(GNeg)
    {
    }
};

/**
 * @brief Helper function to create gradient parameters
 */
template <CElement TElement, int Dims, class TDerivedE, class TDerivedeg, class TDerivedGNeg>
auto MakeMatrixFreeGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
{
    return MatrixFreeGradient<TElement, Dims, TDerivedE, TDerivedeg, TDerivedGNeg>(
        E,
        nNodes,
        eg,
        GNeg);
}

/**
 * @brief Compute gradient matrix-vector multiply \f$ Y += \mathbf{G} X \f$
 *
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param X `|# nodes| x |# cols|` input matrix
 * @param Y `|# dims * # quad.pts.| x |# cols|` output matrix
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGNeg,
    class TDerivedIn,
    class TDerivedOut>
void GemmGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y);

/**
 * @brief Compute gradient matrix-vector multiply \f$ Y += \mathbf{G} X \f$ using mesh
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param X `|# nodes| x |# cols|` input matrix
 * @param Y `|# dims * # quad.pts.| x |# cols|` output matrix
 */
template <CMesh TMesh, class TDerivedeg, class TDerivedGNeg, class TDerivedIn, class TDerivedOut>
inline void GemmGradient(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    GemmGradient<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg,
        GNeg,
        X,
        Y);
}

/**
 * @brief Compute gradient matrix-vector multiply \f$ Y += \mathbf{G} X \f$ using MatrixFreeGradient
 * parameters
 * @tparam TGradient MatrixFreeGradient type (must satisfy CMatrixFreeGradient)
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param G MatrixFreeGradient parameter struct
 * @param X `|# nodes| x |# cols|` input matrix
 * @param Y `|# dims * # quad.pts.| x |# cols|` output matrix
 */
template <CMatrixFreeGradient TGradient, class TDerivedIn, class TDerivedOut>
inline void GemmGradient(
    TGradient const& G,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    GemmGradient<typename TGradient::ElementType, TGradient::kDims>(
        G.E,
        G.nNodes,
        G.eg,
        G.GNeg,
        X,
        Y);
}

/**
 * @brief Construct the gradient operator's sparse matrix representation
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @return Sparse matrix representation of the gradient operator
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGNeg>
auto GradientMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the gradient operator's sparse matrix representation using mesh
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @return Sparse matrix representation of the gradient operator
 */
template <Eigen::StorageOptions Options, CMesh TMesh, class TDerivedeg, class TDerivedGNeg>
auto GradientMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TMesh::IndexType>
{
    return GradientMatrix<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg,
        GNeg);
}

/**
 * @brief Construct the gradient operator's sparse matrix representation using MatrixFreeGradient
 * parameters
 * @tparam TGradient MatrixFreeGradient type (must satisfy CMatrixFreeGradient)
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @param G MatrixFreeGradient parameter struct
 * @return Sparse matrix representation of the gradient operator
 */
template <Eigen::StorageOptions Options, CMatrixFreeGradient TGradient>
auto GradientMatrix(TGradient const& G)
    -> Eigen::SparseMatrix<typename TGradient::ScalarType, Options, typename TGradient::IndexType>
{
    return GradientMatrix<typename TGradient::ElementType, TGradient::kDims, Options>(
        G.E,
        G.nNodes,
        G.eg,
        G.GNeg);
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGNeg,
    class TDerivedIn,
    class TDerivedOut>
inline void GemmGradient(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GemmGradient");
    // Check inputs
    auto const rows = Dims * eg.size();
    auto const cols = nNodes;
    bool const bDimensionsMatch =
        (X.cols() == Y.cols()) and (X.rows() == cols) and (Y.rows() == rows);
    if (not bDimensionsMatch)
    {
        std::string const what = fmt::format(
            "Expected input to have rows={} and output to have rows={}, and same number of "
            "columns, but got dimensions "
            "X,Y=({} x {}), ({} x {})",
            cols,
            rows,
            X.rows(),
            X.cols(),
            Y.rows(),
            Y.cols());
        throw std::invalid_argument(what);
    }
    // Compute gradient
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDims            = Dims;
    auto const nQuadPts             = eg.size();
    for (auto c = 0; c < X.cols(); ++c)
    {
        for (auto g = 0; g < nQuadPts; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = E.col(e);
            auto const Xe    = X.col(c)(nodes);
            auto const Geg   = GNeg.template block<kNodesPerElement, kDims>(0, g * kDims);
            for (auto d = 0; d < kDims; ++d)
            {
                Y(d * nQuadPts + g) += Geg(Eigen::placeholders::all, d).transpose() * Xe;
            }
        }
    }
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedGNeg>
auto GradientMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GradientMatrix");
    using ScalarType       = typename TDerivedGNeg::Scalar;
    using IndexType        = typename TDerivedE::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<ScalarType, Options, IndexType>;
    using Triplet          = Eigen::Triplet<ScalarType, IndexType>;

    // Compile-time constants
    static_assert(
        TElement::kNodes != Eigen::Dynamic,
        "Element nodes must be known at compile time");
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDims            = Dims;

    auto const nQuadPts = eg.size();
    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(kNodesPerElement * kDims * nQuadPts));
    SparseMatrixType G(kDims * nQuadPts, nNodes);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const Geg   = GNeg.template block<kNodesPerElement, kDims>(0, g * kDims);
        for (auto d = 0; d < kDims; ++d)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                auto const ni = static_cast<IndexType>(d * nQuadPts + g);
                auto const nj = static_cast<IndexType>(nodes(j));
                triplets.emplace_back(ni, nj, Geg(j, d));
            }
        }
    }
    G.setFromTriplets(triplets.begin(), triplets.end());
    return G;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_GRADIENT_H