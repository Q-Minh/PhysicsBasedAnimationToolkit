#ifndef PBAT_GRAPH_ADJACENCY_H
#define PBAT_GRAPH_ADJACENCY_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"

#include <concepts>
#include <tuple>

namespace pbat {
namespace graph {

template <class TDerivedP, std::integral TIndex = typename TDerivedP::Scalar>
std::tuple<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
MapToAdjacency(Eigen::DenseBase<TDerivedP> const& p, TIndex n = TIndex(-1))
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    if (n < 0)
        n = p.maxCoeff() + TIndex(1);
    auto s   = common::Counts<TIndex>(p.begin(), p.end(), n);
    auto ptr = common::CumSum(s);
    auto adj = common::ArgSort<TIndex>(p.size(), [&](auto i, auto j) { return p(i) < p(j); });
    return std::make_tuple(common::ToEigen(ptr), common::ToEigen(adj));
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

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCY_H