/**
 * @file Adjacency.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Adjacency matrix utilities
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_GRAPH_ADJACENCY_H
#define PBAT_GRAPH_ADJACENCY_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"

#include <concepts>
#include <iterator>
#include <tuple>
#include <vector>

namespace pbat {
namespace graph {

/**
 * @brief Weighted edge (wrapper around Eigen triplet type)
 *
 * @tparam TWeight Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 */
template <class TWeight = Scalar, class TIndex = Index>
using WeightedEdge = Eigen::Triplet<TWeight, TIndex>;

template <class TWeightedEdge>
struct WeightedEdgeTraits
{
    using ScalarType = std::remove_cvref_t<decltype(std::declval<TWeightedEdge>().value())>;
    using IndexType  = std::remove_cvref_t<decltype(std::declval<TWeightedEdge>().row())>;
};

/**
 * @brief Construct adjacency matrix from edge/triplet list
 *
 * @tparam TWeightedEdgeIterator Iterator type of the edge list
 * @tparam TWeightedEdge Type of the edge
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param begin Iterator to the beginning of the edge list
 * @param end Iterator to the end of the edge list
 * @param m Number of rows of the adjacency matrix. If not provided (i.e. m < 0), it is inferred
 * from the edge list.
 * @param n Number of columns of the adjacency matrix. If not provided (i.e. n < 0), it is inferred
 * from the edge list.
 * @return Adjacency matrix
 */
template <
    class TWeightedEdgeIterator,
    class TWeightedEdge = typename std::iterator_traits<TWeightedEdgeIterator>::value_type,
    class TScalar       = typename WeightedEdgeTraits<TWeightedEdge>::ScalarType,
    class TIndex        = typename WeightedEdgeTraits<TWeightedEdge>::IndexType>
auto AdjacencyMatrixFromEdges(
    TWeightedEdgeIterator begin,
    TWeightedEdgeIterator end,
    TIndex m = TIndex(-1),
    TIndex n = TIndex(-1)) -> Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex>
{
    if (m < 0)
    {
        m = std::max_element(
                begin,
                end,
                [](TWeightedEdge const& lhs, TWeightedEdge const& rhs) {
                    return lhs.row() < rhs.row();
                })
                ->row() +
            TIndex(1);
    }
    if (n < 0)
    {
        n = std::max_element(
                begin,
                end,
                [](TWeightedEdge const& lhs, TWeightedEdge const& rhs) {
                    return lhs.col() < rhs.col();
                })
                ->col() +
            TIndex(1);
    }
    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> G(m, n);
    G.setFromTriplets(begin, end);
    return G;
}

/**
 * @brief Non-owning wrapper around the offset pointers of a compressed sparse matrix
 *
 * @tparam TDerivedA Type of input adjacency matrix
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param A Input adjacency matrix
 * @return Non-owning wrapper around the offset pointers of the adjacency matrix
 */
template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto AdjacencyMatrixPrefix(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    using IndexVectorType = Eigen::Map<Eigen::Vector<TIndex, Eigen::Dynamic>>;
    return Eigen::Map<IndexVectorType const>(A.outerIndexPtr(), A.outerSize() + 1);
}

/**
 * @brief Non-owning wrapper around the indices of a compressed sparse matrix
 *
 * @tparam TDerivedA Type of input adjacency matrix
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param A Input adjacency matrix
 * @return Non-owning wrapper around the indices of the adjacency matrix
 */
template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto AdjacencyMatrixIndices(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    using IndexVectorType = Eigen::Map<Eigen::Vector<TIndex, Eigen::Dynamic>>;
    return Eigen::Map<IndexVectorType const>(A.innerIndexPtr(), A.nonZeros());
}

/**
 * @brief Non-owning wrapper around the weights of a compressed sparse matrix
 *
 * @tparam TDerivedA Type of input adjacency matrix
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param A Input adjacency matrix
 * @return Non-owning wrapper around the weights of the adjacency matrix
 */
template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto AdjacencyMatrixWeights(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    using WeightVectorType = Eigen::Map<Eigen::Vector<TScalar, Eigen::Dynamic>>;
    return Eigen::Map<WeightVectorType const>(A.valuePtr(), A.nonZeros());
}

/**
 * @brief Construct adjacency list in compressed sparse format from a map p s.t. p(i) is the index
 * of the vertex adjacent to i
 *
 * @tparam TDerivedP Type of input map
 * @tparam TScalar Scalar type of the graph edge weights
 * @param p Input map
 * @param n Number of vertices in the graph. If not provided (i.e. n < 0), it is inferred from the
 * map.
 * @return Tuple of the offset pointers and indices of the adjacency list
 */
template <class TDerivedP, std::integral TIndex = typename TDerivedP::Scalar>
auto MapToAdjacency(Eigen::DenseBase<TDerivedP> const& p, TIndex n = TIndex(-1))
    -> std::tuple<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    if (n < 0)
        n = p.maxCoeff() + TIndex(1);
    auto s   = common::Counts(p.begin(), p.end(), n);
    auto ptr = common::CumSum(s);
    auto adj = common::ArgSort(p.size(), [&](auto i, auto j) { return p(i) < p(j); });
    return std::make_tuple(ptr, adj);
}

/**
 * @brief Obtain the offset and indices arrays of an input adjacency matrix
 *
 * @tparam TDerivedA Type of input adjacency matrix
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param A Input adjacency matrix
 * @return Tuple of the offset pointers and indices of the adjacency matrix
 */
template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto MatrixToAdjacency(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    return std::make_tuple(
        AdjacencyMatrixPrefix<TDerivedA, TScalar, TIndex>(A.derived()),
        AdjacencyMatrixIndices<TDerivedA, TScalar, TIndex>(A.derived()));
}

/**
 * @brief Construct adjacency list in compressed sparse format from an input adjacency matrix
 *
 * @tparam TDerivedA Type of input adjacency matrix
 * @tparam TScalar Scalar type of the graph edge weights
 * @tparam TIndex Index type of the graph vertices
 * @param A Input adjacency matrix
 * @return Tuple of the offset pointers, indices, and weights of the adjacency matrix
 */
template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto MatrixToWeightedAdjacency(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    return std::make_tuple(
        AdjacencyMatrixPrefix<TDerivedA, TScalar, TIndex>(A.derived()),
        AdjacencyMatrixIndices<TDerivedA, TScalar, TIndex>(A.derived()),
        AdjacencyMatrixWeights<TDerivedA, TScalar, TIndex>(A.derived()));
}

/**
 * @brief Construct adjacency list in compressed sparse format from an input adjacency list in list
 * of lists format
 *
 * @tparam TIndex Index type of the graph vertices
 * @param lil Input adjacency list in list of lists format
 * @return Tuple of the offset pointers and indices of the adjacency list
 */
template <class TIndex = Index>
auto ListOfListsToAdjacency(std::vector<std::vector<TIndex>> const& lil)
{
    auto n = static_cast<TIndex>(lil.size());
    TIndex nEdges{0};
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType ptr(n + TIndex(1));
    ptr(0) = TIndex(0);
    for (auto l = 0; l < n; ++l)
    {
        auto lStl                 = static_cast<std::size_t>(l);
        auto nVerticesInPartition = static_cast<TIndex>(lil[lStl].size());
        ptr(l + 1)                = ptr(l) + nVerticesInPartition;
        nEdges += nVerticesInPartition;
    }
    IndexVectorType adj(nEdges);
    for (auto l = 0; l < n; ++l)
    {
        auto start                           = ptr(l);
        auto end                             = ptr(l + 1);
        auto lStl                            = static_cast<std::size_t>(l);
        adj(Eigen::seqN(start, end - start)) = common::ToEigen(lil[lStl]);
    }
    return std::make_tuple(ptr, adj);
}

/**
 * @brief Edge iteration over the adjacency list in compressed sparse format
 *
 * @tparam TDerivedPtr Eigen dense expression of the offset pointers of the adjacency list
 * @tparam TDerivedAdj Eigen dense expression of the indices of the adjacency list
 * @tparam TIndex Index type of the graph vertices
 * @tparam Func Callable type of the edge iteration function
 * @param ptr Offset pointers of the adjacency list
 * @param adj Indices of the adjacency list
 * @param f Callable function to be applied to each edge with signature `void(TIndex i, TIndex j,
 * TIndex k)`
 */
template <
    class TDerivedPtr,
    class TDerivedAdj,
    class TIndex = typename TDerivedPtr::Scalar,
    class Func>
void ForEachEdge(Eigen::DenseBase<TDerivedPtr>& ptr, Eigen::DenseBase<TDerivedAdj>& adj, Func&& f)
{
    auto nVertices = ptr.size() - 1;
    for (TIndex u = 0; u < nVertices; ++u)
    {
        auto uBegin = ptr(u);
        auto uEnd   = ptr(u + 1);
        for (auto k = uBegin; k < uEnd; ++k)
        {
            auto v = adj(k);
            f(u, v, k);
        }
    }
}

/**
 * @brief In-place removal of edges from the adjacency list in compressed sparse format
 *
 * @tparam TDerivedPtr Eigen dense expression of the offset pointers of the adjacency list
 * @tparam TDerivedAdj Eigen dense expression of the indices of the adjacency list
 * @tparam TIndex Index type of the graph vertices
 * @tparam Func Callable type of the edge removal function with signature `bool(TIndex i, TIndex j)`
 * @param ptr Offset pointers of the adjacency list
 * @param adj Indices of the adjacency list
 * @param fShouldDeleteEdge Callable function to determine if an edge should be removed with
 * signature `bool(TIndex i, TIndex j)`
 */
template <
    class TDerivedPtr,
    class TDerivedAdj,
    class TIndex = typename TDerivedPtr::Scalar,
    class Func>
void RemoveEdges(
    Eigen::DenseBase<TDerivedPtr>& ptr,
    Eigen::DenseBase<TDerivedAdj>& adj,
    Func fShouldDeleteEdge)
{
    auto nPartitions      = ptr.size() - 1;
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType sizes(nPartitions);
    sizes.setZero();
    TIndex nEdges(0);
    // Compress edges by discarding edges to remove
    IndexVectorType adjTmp(adj.size());
    IndexVectorType ptrTmp = ptr;
    for (auto p = 0, kk = 0; p < nPartitions; ++p)
    {
        auto kBegin = ptr(p);
        auto kEnd   = ptr(p + TIndex(1));
        sizes(p)    = kEnd - kBegin;
        for (auto k = kBegin; k < kEnd; ++k)
        {
            if (fShouldDeleteEdge(p, adj(k)))
            {
                --sizes(p);
            }
            else
            {
                adjTmp(kk) = adj(k);
                ++kk;
            }
        }
        nEdges += sizes(p);
        ptrTmp(p + TIndex(1)) = ptrTmp(p) + sizes(p);
    }
    ptr = ptrTmp;
    adj = adjTmp.head(nEdges);
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCY_H