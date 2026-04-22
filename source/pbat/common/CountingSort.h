/**
 * @file CountingSort.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Counting sort
 * @date 2025-03-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_COMMON_COUNTINGSORT_H
#define PBAT_COMMON_COUNTINGSORT_H

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <type_traits>

namespace pbat::common {

/**
 * @brief In-place counting sort for integer keys in a random access range, with specified key
 * range.
 * @note The range is modified in-place, and the order of equal keys is not guaranteed to be stable.
 * @tparam TRng Integer random access range.
 * @tparam TWork Integer random access range.
 * @tparam FProject Callable that maps elements of `rng` to integer keys (default: identity).
 * @tparam TKey Integer type of the keys (deduced from FProject if not specified).
 * @param rng Range of integer keys to sort
 * @param work Temporary buffer for counting occurrences.
 * @param min Minimum key value
 * @param max Maximum key value
 * @param fProject Projection function to extract keys from elements of `rng` (default: identity)
 * @pre `std::ranges::size(work) > (max(rng) - min(rng))` to ensure the count array can accommodate
 * all keys.
 */
template <
    std::ranges::random_access_range TRng,
    std::ranges::random_access_range TWork,
    class FProject = std::identity,
    std::integral TKey =
        std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TRng>>>>
    requires std::integral<TKey> and std::integral<std::ranges::range_value_t<TWork>>
void CountingSort(TRng&& rng, TWork&& work, TKey min, TKey max, FProject fProject = {})
{
    using SizeType = std::ranges::range_size_t<TRng>;
    SizeType n     = std::ranges::size(rng);
    if (n == 0)
        return;
    auto wb = std::ranges::begin(work);
    auto we = wb + (max - min + 1);
    std::fill(wb, we, TKey(0));
    for (SizeType i = 0; i < n; ++i)
        ++work[fProject(rng[i]) - min];
    std::inclusive_scan(wb, we, wb);
    for (SizeType i = 0; i < n; ++i)
    {
        auto k = fProject(rng[i]) - min;
        while (i < work[k] - 1)
        {
            auto j = --work[k];
            using std::swap;
            swap(rng[i], rng[j]);
            k = fProject(rng[i]) - min;
        }
    }
}

/**
 * @brief In-place counting sort for integer keys in a random access range.
 * @note The range is modified in-place, and the order of equal keys is not guaranteed to be stable.
 * @tparam TWork Integer random access range.
 * @tparam TRng Integer random access range.
 * @tparam FProject Callable that maps elements of `rng` to integer keys (default: identity).
 * @param rng Range of integer keys to sort
 * @param work Temporary buffer for counting occurrences.
 * @param fProject Projection function to extract keys from elements of `rng` (default: identity)
 * @pre `std::ranges::size(work) > (max(rng) - min(rng))` to ensure the count array can accommodate
 * all keys.
 */
template <
    std::ranges::random_access_range TRng,
    std::ranges::random_access_range TWork,
    class FProject = std::identity,
    std::integral TKey =
        std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TRng>>>>
    requires std::integral<std::ranges::range_value_t<TWork>>
void CountingSort(TRng&& rng, TWork&& work, FProject fProject = {})
{
    if (std::ranges::empty(rng))
        return;
    auto const begin      = std::ranges::begin(rng);
    auto const end        = std::ranges::end(rng);
    auto const [min, max] = std::ranges::minmax_element(rng, {}, fProject);
    CountingSort(rng, work, fProject(*min), fProject(*max), std::move(fProject));
}

} // namespace pbat::common

#endif // PBAT_COMMON_COUNTINGSORT_H
