/**
 * @file ArgSort.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Non-intrusive sorting
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup common
 */

#ifndef PBAT_COMMON_ARG_SORT_H
#define PBAT_COMMON_ARG_SORT_H

#include "pbat/Aliases.h"

#include <algorithm>
#include <concepts>
#include <numeric>

namespace pbat {
namespace common {

/**
 * @brief Computes the indices that would sort an array
 *
 * @tparam TIndex Coefficient type of returned vector
 * @tparam FLess Callable with signature `bool(TIndex, TIndex)`
 * @param n Number of elements
 * @param less Less-than comparison function object
 * @return Eigen::Vector<TIndex, Eigen::Dynamic>
 * @ingroup common
 */
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