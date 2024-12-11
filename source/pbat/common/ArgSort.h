#ifndef PBAT_COMMON_ARG_SORT_H
#define PBAT_COMMON_ARG_SORT_H

#include "pbat/Aliases.h"

#include <algorithm>
#include <concepts>
#include <numeric>

namespace pbat {
namespace common {

template <std::integral TIndex, class FLess>
Eigen::Vector<TIndex, Eigen::Dynamic> ArgSort(TIndex n, FLess less)
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType inds(n);
    std::iota(inds.begin(), inds.end(), TIndex(0));
    std::stable_sort(inds.begin(), inds.end(), less);
    return inds;
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_ARG_SORT_H