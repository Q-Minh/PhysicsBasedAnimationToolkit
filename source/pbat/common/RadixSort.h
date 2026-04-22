/**
 * @file RadixSort.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Radix sort
 * @version 0.1
 * @date 2026-02-24
 * @copyright Copyright (c) 2026
 */

#ifndef PBAT_COMMON_RADIXSORT_H
#define PBAT_COMMON_RADIXSORT_H

#include "CountingSort.h"

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>
#include <type_traits>

namespace pbat::common {

template <
    std::ranges::random_access_range TInRng,
    std::ranges::random_access_range TOutRng,
    class FProject = std::identity,
    std::integral TKey =
        std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TInRng>>>>
void RadixSort(TInRng&& in, TOutRng&& out, FProject fProject = {})
{
}

} // namespace pbat::common

#endif // PBAT_COMMON_RADIXSORT_H