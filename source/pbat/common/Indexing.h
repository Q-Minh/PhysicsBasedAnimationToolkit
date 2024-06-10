#ifndef PBAT_COMMON_INDEXING_H
#define PBAT_COMMON_INDEXING_H

#include "Concepts.h"

#include <numeric>
#include <pbat/aliases.h>
#include <ranges>
#include <vector>

namespace pbat {
namespace common {

template <CContiguousIndexRange R>
std::vector<Index> cumsum(R&& sizes)
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

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_INDEXING_H