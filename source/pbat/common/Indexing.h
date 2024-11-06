#ifndef PBAT_COMMON_INDEXING_H
#define PBAT_COMMON_INDEXING_H

#include "Concepts.h"

#include <concepts>
#include <numeric>
#include <pbat/Aliases.h>
#include <ranges>
#include <vector>

namespace pbat {
namespace common {

template <CContiguousIndexRange R>
std::vector<Index> CumSum(R&& sizes)
{
    namespace rng = std::ranges;
    std::vector<Index> cs{};
    cs.reserve(rng::size(sizes) + 1);
    auto bi    = std::back_inserter(cs);
    *bi++      = Index{0};
    auto begin = rng::begin(sizes);
    auto end   = rng::end(sizes);
    std::partial_sum(begin, end, bi);
    return cs;
}

template <std::integral TIndex>
std::vector<TIndex> Counts(auto begin, auto end, auto ncounts)
{
    std::vector<TIndex> counts(ncounts, TIndex(0));
    for (auto it = begin; it != end; ++it)
        ++counts[*it];
    return counts;
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_INDEXING_H