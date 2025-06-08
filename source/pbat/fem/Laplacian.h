/**
 * @file LaplacianMatrix.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief FEM Laplacian matrix operator
 * @date 2025-02-11
 *
 * @details The symmetric part of the Laplacian \f$\Delta u\f$ of a finite element discretized
 * function \f$ u(X) \f$ under Galerkin projection.
 *
 * The precise definition of the Laplacian matrix's symmetric part is given by
 * \f$ \mathbf{L}_{ij} = \int_\Omega -\nabla \phi_i \cdot \nabla \phi_j d\Omega \f$.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_FEM_LAPLACIAN_H
#define PBAT_FEM_LAPLACIAN_H

#include "Concepts.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <exception>
#include <fmt/core.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Compute Laplacian matrix-vector multiply \f$ Y += \mathbf{L} X \f$
 *
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @param X `|# nodes * dims| x |# cols|` input matrix
 * @param Y `|# nodes * dims| x |# cols|` output matrix
 */
template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedIn,
    class TDerivedOut>
void GemmLaplacian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y);

/**
 * @brief Compute Laplacian matrix-vector multiply \f$ Y += \mathbf{L} X \f$ using mesh
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @tparam TDerivedIn Type of the input matrix
 * @tparam TDerivedOut Type of the output matrix
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @param X `|# nodes * dims| x |# cols|` input matrix
 * @param Y `|# nodes * dims| x |# cols|` output matrix
 */
template <
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedIn,
    class TDerivedOut>
inline void GemmLaplacian(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    GemmLaplacian<typename TMesh::ElementType, TMesh::kDims>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        dims,
        X,
        Y);
}

/**
 * @brief Construct the Laplacian operator's sparse matrix representation
 * @tparam TElement Element type
 * @tparam Dims Number of spatial dimensions
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TDerivedE Type of the element matrix
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @param E `|# nodes per element| x |# elements|` matrix of mesh elements
 * @param nNodes Number of mesh nodes
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the Laplacian operator
 */
template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg>
auto LaplacianMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TDerivedE::Scalar>;

/**
 * @brief Construct the Laplacian operator's sparse matrix representation using mesh
 * @tparam Options Storage options for the matrix (default: Eigen::ColMajor)
 * @tparam TMesh Type of the mesh
 * @tparam TDerivedeg Type of the element indices at quadrature points
 * @tparam TDerivedwg Type of the quadrature weights
 * @tparam TDerivedGNeg Type of the shape function gradients at quadrature points
 * @param mesh The finite element mesh
 * @param eg `|# quad.pts.| x 1` vector of element indices at quadrature points
 * @param wg `|# quad.pts.| x 1` vector of quadrature weights
 * @param GNeg `|# nodes per element| x |# dims * # quad.pts.|` shape function gradients at
 * quadrature points
 * @param dims Dimensionality of the image of the FEM function space
 * @return Sparse matrix representation of the Laplacian operator
 */
template <
    Eigen::StorageOptions Options,
    CMesh TMesh,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg>
auto LaplacianMatrix(
    TMesh const& mesh,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims = 1)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TMesh::IndexType>
{
    return LaplacianMatrix<typename TMesh::ElementType, TMesh::kDims, Options>(
        mesh.E,
        static_cast<typename TMesh::IndexType>(mesh.X.cols()),
        eg.derived(),
        wg.derived(),
        GNeg.derived(),
        dims);
}

template <
    CElement TElement,
    int Dims,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg,
    class TDerivedIn,
    class TDerivedOut>
inline void GemmLaplacian(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims,
    Eigen::MatrixBase<TDerivedIn> const& X,
    Eigen::DenseBase<TDerivedOut>& Y)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.GemmLaplacian");

    // Check inputs
    auto const numberOfDofs = dims * nNodes;
    bool const bDimensionsMatch =
        (X.cols() == Y.cols()) and (X.rows() == numberOfDofs) and (Y.rows() == numberOfDofs);
    if (not bDimensionsMatch)
    {
        std::string const what = fmt::format(
            "Expected input and output to have {} rows and same number of columns, but got "
            "dimensions X,Y=({} x {}), ({} x {})",
            numberOfDofs,
            X.rows(),
            X.cols(),
            Y.rows(),
            Y.cols());
        throw std::invalid_argument(what);
    }

    if (dims < 1)
    {
        std::string const what = fmt::format("Expected dims >= 1, got {} instead", dims);
        throw std::invalid_argument(what);
    }

    // Compute element Laplacians and apply
    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDims            = Dims;
    auto const nQuadPts             = eg.size();
    for (auto c = 0; c < X.cols(); ++c)
    {
        for (auto g = 0; g < nQuadPts; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = E.col(e);
            auto const w     = wg(g);
            // Get shape function gradients at this quadrature point
            auto const GP = GNeg.template block<kNodesPerElement, kDims>(0, g * kDims);
            // Apply to each dimension
            auto ye       = Y.col(c).reshaped(dims, nNodes)(Eigen::placeholders::all, nodes);
            auto const xe = X.col(c).reshaped(dims, nNodes)(Eigen::placeholders::all, nodes);
            ye -= ((w * xe) * GP) * GP.transpose(); // Laplacian is -w GP GP^T
        }
    }
}

template <
    CElement TElement,
    int Dims,
    Eigen::StorageOptions Options,
    class TDerivedE,
    class TDerivedeg,
    class TDerivedwg,
    class TDerivedGNeg>
auto LaplacianMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    typename TDerivedE::Scalar nNodes,
    Eigen::DenseBase<TDerivedeg> const& eg,
    Eigen::DenseBase<TDerivedwg> const& wg,
    Eigen::MatrixBase<TDerivedGNeg> const& GNeg,
    int dims)
    -> Eigen::SparseMatrix<typename TDerivedGNeg::Scalar, Options, typename TDerivedE::Scalar>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.LaplacianMatrix");
    using ScalarType       = typename TDerivedGNeg::Scalar;
    using IndexType        = typename TDerivedE::Scalar;
    using SparseMatrixType = Eigen::SparseMatrix<ScalarType, Options, IndexType>;
    using Triplet          = Eigen::Triplet<ScalarType, IndexType>;

    auto constexpr kNodesPerElement = TElement::kNodes;
    auto constexpr kDims            = Dims;

    auto const nQuadPts = eg.size();
    std::vector<Triplet> triplets{};
    triplets.reserve(
        static_cast<std::size_t>(kNodesPerElement * kNodesPerElement * nQuadPts * dims));

    auto const numberOfDofs = dims * nNodes;
    SparseMatrixType L(numberOfDofs, numberOfDofs);

    for (auto g = 0; g < nQuadPts; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = E.col(e);
        auto const w     = wg(g);

        // Get shape function gradients at this quadrature point
        auto const GP = GNeg.template block<kNodesPerElement, kDims>(0, g * kDims);

        // Compute element Laplacian: -w * GP * GP^T
        auto const Leg = -w * GP * GP.transpose();

        // Add contributions for each dimension
        for (auto i = 0; i < kNodesPerElement; ++i)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                for (auto d = 0; d < dims; ++d)
                {
                    auto const ni = static_cast<IndexType>(dims * nodes(i) + d);
                    auto const nj = static_cast<IndexType>(dims * nodes(j) + d);
                    triplets.emplace_back(ni, nj, Leg(i, j));
                }
            }
        }
    }

    L.setFromTriplets(triplets.begin(), triplets.end());
    return L;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_LAPLACIAN_H