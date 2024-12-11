#ifndef PBAT_GRAPH_ADJACENCY_H
#define PBAT_GRAPH_ADJACENCY_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"

#include <concepts>
#include <tuple>
#include <vector>

namespace pbat {
namespace graph {

template <class TDerivedP, std::integral TIndex = typename TDerivedP::Scalar>
std::tuple<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
MapToAdjacency(Eigen::DenseBase<TDerivedP> const& p, TIndex n = TIndex(-1))
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    if (n < 0)
        n = p.maxCoeff() + TIndex(1);
    auto s   = common::Counts(p.begin(), p.end(), n);
    auto ptr = common::CumSum(s);
    auto adj = common::ArgSort(p.size(), [&](auto i, auto j) { return p(i) < p(j); });
    return std::make_tuple(ptr, adj);
}

template <
    class TDerivedA,
    class TScalar        = typename TDerivedA::Scalar,
    std::integral TIndex = typename TDerivedA::StorageIndex>
auto MatrixToAdjacency(Eigen::SparseCompressedBase<TDerivedA> const& A)
{
    using IndexVectorType  = Eigen::Map<Eigen::Vector<TIndex, Eigen::Dynamic>>;
    using WeightVectorType = Eigen::Map<Eigen::Vector<TScalar, Eigen::Dynamic>>;
    return std::make_tuple(
        Eigen::Map<IndexVectorType const>(A.outerIndexPtr(), A.outerSize() + 1),
        Eigen::Map<IndexVectorType const>(A.innerIndexPtr(), A.nonZeros()),
        Eigen::Map<WeightVectorType const>(A.valuePtr(), A.nonZeros()));
}

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
        auto nVerticesInPartition = static_cast<TIndex>(lil[l].size());
        ptr(l + 1)                = ptr(l) + nVerticesInPartition;
        nEdges += nVerticesInPartition;
    }
    IndexVectorType adj(nEdges);
    for (auto l = 0; l < n; ++l)
    {
        auto start                           = ptr(l);
        auto end                             = ptr(l + 1);
        adj(Eigen::seqN(start, end - start)) = common::ToEigen(lil[l]);
    }
    return std::make_tuple(ptr, adj);
}

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