#ifndef PBAT_COMMON_ARG_SORT_H
#define PBAT_COMMON_ARG_SORT_H

#include <algorithm>
#include <concepts>
#include <iterator>
#include <numeric>
#include <vector>

namespace pbat {
namespace common {

template <std::integral TIndex, class FLess>
std::vector<TIndex> ArgSort(auto n, FLess less)
{
    std::vector<TIndex> inds(static_cast<std::size_t>(n));
    std::iota(inds.begin(), inds.end(), TIndex(0));
    std::stable_sort(inds.begin(), inds.end(), less);
    return inds;
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_ARG_SORT_H