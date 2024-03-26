#ifndef PBA_CORE_COMMON_INDEXING_H
#define PBA_CORE_COMMON_INDEXING_H

#include "Concepts.h"
#include "pba/aliases.h"

#include <numeric>
#include <ranges>
#include <vector>

namespace pba {
namespace common {

template <ContiguousIndexRange R>
std::vector<Index> cumsum(R&& sizes);

template <ContiguousIndexRange R>
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
} // namespace pba

#endif // PBA_CORE_COMMON_INDEXING_H