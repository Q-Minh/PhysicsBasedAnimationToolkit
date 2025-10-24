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

#include "BreadthFirstSearch.h"
#include "ConnectedComponents.h"
#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Permute.h"
#include "pbat/profiling/Profiling.h"

namespace pbat {
namespace graph {

/**
 * @brief Construct adjacency matrix from mesh
 *
 * @tparam TDerivedE Eigen dense expression for mesh elements
 * @tparam TDerivedW Eigen dense expression for element-vertex weights
 * @tparam TIndex Type of indices used in element array
 * @tparam TScalar Type of weights
 * @param E `|# nodes per element|x|# elements|` array of element indices
 * @param w `|# nodes per element|x|# elements|` array of element-vertex weights
 * @param nNodes Number of nodes in the mesh. If `nNodes < 1`, the number of nodes is inferred from
 * E.
 * @param bVertexToElement If true, the adjacency matrix maps vertices to elements, rather than
 * elements to vertices
 * @param bHasDuplicates If true, duplicate entries in the input mesh will be handled
 * @return Adjacency matrix of requested mesh connectivity
 * @pre `E.rows() == w.rows()` and `E.cols() == w.cols()`
 */
template <
    class TDerivedE,
    class TDerivedW,
    common::CIndex TIndex       = typename TDerivedE::Scalar,
    common::CArithmetic TScalar = typename TDerivedW::Scalar>
auto MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedW> const& w,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false,
    bool bHasDuplicates   = false) -> Eigen::SparseMatrix<TScalar, Eigen::ColMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.MeshAdjacencyMatrix");
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
 * @param E `|# nodes per element|x|# elements|` array of element indices
 * @param nNodes Number of nodes in the mesh. If `nNodes < 1`, the number of nodes is inferred from
 * E.
 * @param bVertexToElement If true, the adjacency matrix maps vertices to elements, rather than
 * @param bHasDuplicates If true, duplicate entries in the input mesh will be handled
 * @return Adjacency matrix of requested mesh connectivity
 */
template <class TDerivedE, common::CIndex TIndex = typename TDerivedE::Scalar>
auto MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false,
    bool bHasDuplicates   = false) -> Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
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
 * @param E `|# nodes per element|x|# elements|` array of element indices
 * @param nNodes Number of nodes in the mesh. If `nNodes < 1`, the number of nodes is inferred from
 * E.
 * @return Primal graph of the input mesh
 */
template <class TDerivedE, common::CIndex TIndex = typename TDerivedE::Scalar>
auto MeshPrimalGraph(Eigen::DenseBase<TDerivedE> const& E, TIndex nNodes = TIndex(-1))
    -> Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.MeshPrimalGraph");
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
 * @param E `|# nodes per element|x|# elements|` array of element indices
 * @param nNodes Number of nodes in the mesh. If `nNodes < 1`, the number of nodes is inferred from
 * E.
 * @param opts Adjacency types to keep in the dual graph
 * @return Dual graph of the input mesh
 */
template <class TDerivedE, common::CIndex TIndex = typename TDerivedE::Scalar>
auto MeshDualGraph(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes              = TIndex(-1),
    EMeshDualGraphOptions opts = EMeshDualGraphOptions::All)
    -> Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.MeshDualGraph");
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
    auto const fKeepAdjacency =
        [=]([[maybe_unused]] auto row, [[maybe_unused]] auto col, auto degree) {
            bool const bKeep = (degree > 3) or (degree == 3 and bKeepFaceAdjacencies) or
                               (degree == 2 and bKeepEdgeAdjacencies) or
                               (degree == 1 and bKeepVertexAdjacencies);
            return bKeep;
        };
    GTG.prune(fKeepAdjacency);
    return GTG;
}

/**
 * @brief Obtain ordering of mesh vertices and elements by sorted connected components
 *
 * @tparam TDerivedX Type of node position matrix
 * @tparam TDerivedE Type of element index matrix
 * @tparam TDerivedXCC Type of node connected component index vector
 * @tparam TDerivedECC Type of element connected component index vector
 * @tparam TDerivedXordering Type of node re-indexing vector
 * @tparam TDerivedEordering Type of element re-indexing vector
 * @tparam TIndex Type of indices used in element array
 * @param X `|# dims| x |# nodes|` node position matrix
 * @param E `|# elem. nodes| x |# elements|` element index matrix
 * @param XCC `|# nodes| x 1` node connected component index vector
 * @param ECC `|# elements| x 1` element connected component index vector
 * @param Xordering `|# nodes| x 1` node re-indexing vector
 * @param Eordering `|# elements| x 1` element re-indexing vector
 * @return Number of connected components in the mesh
 * @post `XCC[i]` gives the connected component index of node `i` in the input mesh
 * @post `ECC[e]` gives the connected component index of element `e` in the input mesh
 * @post `Xordering[i]` gives the new index of node `i` in the re-indexed mesh
 * @post `Eordering[e]` gives the new index of element `e` in the re-indexed mesh
 * @post The ordering is such that all nodes and elements belonging to the same connected component
 * are grouped together, and such connected component groups are sorted, i.e. all `i1` and `e1` with
 * the same component index `c` appear before all `i2` and `e2` with connected component index `d >
 * c`.
 */
template <
    class TDerivedX,
    class TDerivedE,
    class TDerivedXCC,
    class TDerivedECC,
    class TDerivedXordering,
    class TDerivedEordering,
    common::CIndex TIndex = typename TDerivedE::Scalar>
Eigen::Index SortedConnectedComponentOrdering(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedXCC>& XCC,
    Eigen::DenseBase<TDerivedECC>& ECC,
    Eigen::DenseBase<TDerivedXordering>& Xordering,
    Eigen::DenseBase<TDerivedEordering>& Eordering)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.SortedConnectedComponentOrdering");
    using IndexType           = TIndex;
    using XccIndexType        = typename TDerivedXCC::Scalar;
    using EccIndexType        = typename TDerivedECC::Scalar;
    IndexType const nNodes    = static_cast<IndexType>(X.cols());
    IndexType const nElements = static_cast<IndexType>(E.cols());
    // 1. Compute the mesh's dual graph over elements
    Eigen::SparseMatrix<IndexType, Eigen::ColMajor, IndexType> const EG =
        graph::MeshDualGraph(E, nNodes, graph::EMeshDualGraphOptions::All);
    // 2. Compute the connected components of the mesh
    graph::BreadthFirstSearch<EccIndexType> bfs(nElements);
    ECC.resize(nElements);
    ECC.setConstant(EccIndexType(-1));
    IndexType const nComponents = graph::ConnectedComponents<EccIndexType>(
        Eigen::Map<Eigen::Vector<IndexType, Eigen::Dynamic> const>(
            EG.outerIndexPtr(),
            EG.outerSize() + 1),
        Eigen::Map<Eigen::Vector<IndexType, Eigen::Dynamic> const>(
            EG.innerIndexPtr(),
            EG.nonZeros()),
        ECC,
        bfs);
    // 3. Transfer the element connected components to the mesh vertex connected component map
    XCC.resize(nNodes);
    XCC.setConstant(XccIndexType(-1));
    auto verticesToElements =
        Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(nElements, 0, nElements - 1)
            .replicate(1, E.rows())
            .transpose()
            .reshaped(); // `|# elem. nodes| x |# elements|` matrix `[[0,0,0,0], [1,1,1,1], ...,
                         // [nElements-1,nElements-1,nElements-1,nElements-1]]`
    XCC(E.reshaped()) = ECC(verticesToElements).template cast<XccIndexType>();
    // 4. Sort the elements by connected component
    Eordering = common::ArgSort<IndexType>(nElements, [&](IndexType ei, IndexType ej) {
        return ECC[ei] < ECC[ej];
    });
    // 5. Sort vertices by connected component
    Xordering = common::ArgSort<IndexType>(nNodes, [&](IndexType i, IndexType j) {
        return XCC[i] < XCC[j];
    });
    return nComponents;
}

/**
 * @brief Re-index mesh vertices and elements by connected components
 *
 * @tparam TDerivedX Type of node position matrix
 * @tparam TDerivedE Type of element index matrix
 * @tparam TDerivedXCC Type of node connected component index vector
 * @tparam TDerivedECC Type of element connected component index vector
 * @tparam TDerivedXordering Type of node re-indexing vector
 * @tparam TDerivedEordering Type of element re-indexing vector
 * @tparam TIndex Type of indices used in element array
 * @param X `|# dims| x |# nodes|` node position matrix
 * @param E `|# elem. nodes| x |# elements|` element index matrix
 * @param XCC `|# nodes| x 1` node connected component index vector
 * @param ECC `|# elements| x 1` element connected component index vector
 * @param Xordering `|# nodes| x 1` node re-indexing vector
 * @param Eordering `|# elements| x 1` element re-indexing vector
 * @pre `XCC[i]` gives the connected component index of node `i` in the input mesh
 * @pre `ECC[e]` gives the connected component index of element `e` in the input mesh
 * @pre `Xordering[i]` gives the new index of node `i` in the re-indexed mesh
 * @pre `Eordering[e]` gives the new index of element `e` in the re-indexed mesh
 * @post The input mesh X and E are re-indexed in-place such that nodes and elements belonging to
 * the same connected component are grouped together, and such connected component groups are
 * sorted, i.e. all `i1` and `e1` with the same component index `c` appear before all `i2` and `e2`
 * with connected component index `d > c`.
 */
template <
    class TDerivedX,
    class TDerivedE,
    class TDerivedXCC,
    class TDerivedECC,
    class TDerivedXordering,
    class TDerivedEordering,
    common::CIndex TIndex = typename TDerivedE::Scalar>
void ReindexMeshByConnectedComponents(
    Eigen::DenseBase<TDerivedX>& X,
    Eigen::DenseBase<TDerivedE>& E,
    Eigen::DenseBase<TDerivedXCC>& XCC,
    Eigen::DenseBase<TDerivedECC>& ECC,
    Eigen::DenseBase<TDerivedXordering>& Xordering,
    Eigen::DenseBase<TDerivedEordering>& Eordering)
{
    // Re-index elements and element connected components
    for (auto r = 0; r < E.rows(); ++r)
        common::Permute(E.row(r).begin(), E.row(r).end(), Eordering.begin());
    common::Permute(ECC.begin(), ECC.end(), Eordering.begin());
    // Re-index nodes and node connected components
    for (auto d = 0; d < X.rows(); ++d)
        common::Permute(X.row(d).begin(), X.row(d).end(), Xordering.begin());
    common::Permute(XCC.begin(), XCC.end(), Xordering.begin());
    // 6. Re-index mesh node indices to match the sorted order
    // If Xordering[i] = j, then all nodes i in E must become j
    auto nNodes = X.cols();
    Eigen::Vector<TIndex, Eigen::Dynamic> XorderingInverse(nNodes);
    XorderingInverse(Xordering.derived()) =
        Eigen::Vector<TIndex, Eigen::Dynamic>::LinSpaced(nNodes, 0, nNodes - 1);
    E.reshaped() = XorderingInverse(E.reshaped());
}

/**
 * @brief Re-index mesh vertices and elements by connected components
 *
 * @tparam TDerivedX Type of node position matrix
 * @tparam TDerivedE Type of element index matrix
 * @tparam TIndex Type of indices used in element array
 * @param X `|# dims| x |# nodes|` node position matrix
 * @param E `|# elem. nodes| x |# elements|` element index matrix
 * @return Number of connected components in the mesh
 * @post The input mesh X and E are re-indexed in-place such that nodes and elements
 * belonging to the same connected component are grouped together, and such connected component
 * groups are sorted, i.e. all `i1` and `e1` with the same component index `c` appear before all
 * `i2` and `e2` with connected component index `d > c`.
 */
template <class TDerivedX, class TDerivedE, common::CIndex TIndex = typename TDerivedE::Scalar>
Eigen::Index
ReindexMeshByConnectedComponents(Eigen::DenseBase<TDerivedX>& X, Eigen::DenseBase<TDerivedE>& E)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.ReindexMeshByConnectedComponents");
    using IndexType           = TIndex;
    IndexType const nNodes    = static_cast<IndexType>(X.cols());
    IndexType const nElements = static_cast<IndexType>(E.cols());
    Eigen::Vector<IndexType, Eigen::Dynamic> XCC(nNodes);
    Eigen::Vector<IndexType, Eigen::Dynamic> ECC(nElements);
    Eigen::Vector<IndexType, Eigen::Dynamic> Xordering(nNodes);
    Eigen::Vector<IndexType, Eigen::Dynamic> Eordering(nElements);
    Eigen::Index nComponents =
        SortedConnectedComponentOrdering(X, E, XCC, ECC, Xordering, Eordering);
    ReindexMeshByConnectedComponents(X, E, XCC, ECC, Xordering, Eordering);
    return nComponents;
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_MESH_H
