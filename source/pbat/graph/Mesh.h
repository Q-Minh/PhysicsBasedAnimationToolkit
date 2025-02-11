/**
 * @file Mesh.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Mesh graph utilities
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_GRAPH_MESH_H
#define PBAT_GRAPH_MESH_H

#include "pbat/Aliases.h"

#include <concepts>

namespace pbat {
namespace graph {

/**
 * @brief Construct adjacency matrix from mesh
 *
 * @tparam TDerivedE Eigen dense expression for mesh elements
 * @tparam TDerivedW Eigen dense expression for element-vertex weights
 * @tparam TIndex Type of indices used in element array
 * @tparam TScalar Type of weights
 * @param E |#nodes per element|x|#elements| array of element indices
 * @param w |#nodes per element|x|#elements| array of element-vertex weights
 * @param nNodes Number of nodes in the mesh. If nNodes < 1, the number of nodes is inferred from E.
 * @param bVertexToElement If true, the adjacency matrix maps vertices to elements, rather than
 * elements to vertices
 * @param bHasDuplicates If true, duplicate entries in the input mesh will be handled
 * @return Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex> Adjacency matrix of requested mesh
 * connectivity
 */
template <
    class TDerivedE,
    class TDerivedW,
    std::integral TIndex = typename TDerivedE::Scalar,
    class TScalar        = typename TDerivedW::Scalar>
Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex> MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedW> const& w,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false,
    bool bHasDuplicates   = false)
{
    if (nNodes < 0)
        nNodes = E.maxCoeff() + TIndex(1);

    using AdjacencyMatrix = Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex>;
    AdjacencyMatrix G(nNodes, E.cols());
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    if (not bHasDuplicates)
    {
        G.reserve(IndexVectorType::Constant(E.cols(), static_cast<TIndex>(E.rows())));
        for (auto e = 0; e < E.cols(); ++e)
            for (auto i = 0; i < E.rows(); ++i)
                G.insert(E(i, e), e) = w(i, e);
    }
    else
    {
        using Triplet = Eigen::Triplet<TScalar, TIndex>;
        std::vector<Triplet> triplets{};
        triplets.reserve(static_cast<std::size_t>(E.rows() * E.cols()));
        for (auto e = 0; e < E.cols(); ++e)
            for (auto i = 0; i < E.rows(); ++i)
                triplets.emplace_back(E(i, e), e, w(i, e));
        G.setFromTriplets(triplets.begin(), triplets.end());
    }
    if (bVertexToElement)
        G = G.transpose();
    return G;
}

/**
 * @brief Construct adjacency matrix from mesh
 *
 * @tparam TDerivedE Eigen dense expression for mesh elements
 * @tparam TIndex Type of indices used in element array
 * @param E |#nodes per element|x|#elements| array of element indices
 * @param nNodes Number of nodes in the mesh. If nNodes < 1, the number of nodes is inferred from E.
 * @param bVertexToElement If true, the adjacency matrix maps vertices to elements, rather than
 * @param bHasDuplicates If true, duplicate entries in the input mesh will be handled
 * @return Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> Adjacency matrix of requested mesh
 * connectivity
 */
template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false,
    bool bHasDuplicates   = false)
{
    using WeightMatrixType = Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>;
    return MeshAdjacencyMatrix(
        E,
        WeightMatrixType::Ones(E.rows(), E.cols()),
        nNodes,
        bVertexToElement,
        bHasDuplicates);
}

/**
 * @brief Construct primal graph of input mesh, i.e. the graph of adjacent vertices
 *
 * @tparam TDerivedE Eigen dense expression for mesh elements
 * @tparam TIndex Type of indices used in element array
 * @param E |#nodes per element|x|#elements| array of element indices
 * @param nNodes Number of nodes in the mesh. If nNodes < 1, the number of nodes is inferred from E.
 * @return Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> Primal graph of the input mesh

 */
template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
MeshPrimalGraph(Eigen::DenseBase<TDerivedE> const& E, TIndex nNodes = TIndex(-1))
{
    auto const G = MeshAdjacencyMatrix(E, nNodes);
    return G * G.transpose();
}

/**
 * @brief Types of dual graph adjacencies
 */
enum class EMeshDualGraphOptions : std::int32_t {
    VertexAdjacent = 0b001,
    EdgeAdjacent   = 0b010,
    FaceAdjacent   = 0b100,
    All            = 0b111
};

/**
 * @brief Construct dual graph of input mesh, i.e. the graph of adjacent elements
 *
 * @tparam TDerivedE Eigen dense expression for mesh elements
 * @tparam TIndex Type of indices used in element array
 * @param E |#nodes per element|x|#elements| array of element indices
 * @param nNodes Number of nodes in the mesh. If nNodes < 1, the number of nodes is inferred from E.
 * @param opts Adjacency types to keep in the dual graph
 * @return Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> Dual graph of the input mesh
 */
template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> MeshDualGraph(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes              = TIndex(-1),
    EMeshDualGraphOptions opts = EMeshDualGraphOptions::All)
{
    auto const G           = MeshAdjacencyMatrix(E, nNodes);
    using SparseMatrixType = Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>;
    SparseMatrixType GTG   = G.transpose() * G;
    if (opts == EMeshDualGraphOptions::All)
        return GTG;
    auto flags = static_cast<std::int32_t>(opts);
    bool const bKeepFaceAdjacencies =
        flags & static_cast<std::int32_t>(EMeshDualGraphOptions::FaceAdjacent);
    bool const bKeepEdgeAdjacencies =
        flags & static_cast<std::int32_t>(EMeshDualGraphOptions::EdgeAdjacent);
    bool const bKeepVertexAdjacencies =
        flags & static_cast<std::int32_t>(EMeshDualGraphOptions::VertexAdjacent);
    auto const fKeepAdjacency = [=](auto row, auto col, auto degree) {
        bool const bKeep = (degree == 3 and bKeepFaceAdjacencies) or
                           (degree == 2 and bKeepEdgeAdjacencies) or
                           (degree == 1 and bKeepVertexAdjacencies);
        return bKeep;
    };
    GTG.prune(fKeepAdjacency);
    return GTG;
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_MESH_H