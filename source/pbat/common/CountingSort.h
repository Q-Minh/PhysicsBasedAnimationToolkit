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
    auto wb         = std::ranges::begin(work);
    auto we         = wb + (max - min + 1);
    using CountType = std::ranges::range_value_t<TWork>;
    std::fill(wb, we, CountType(0));
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
 * @brief Stable (out-of-place) counting sort for integer keys in a random access range, with
 * specified key range.
 * @tparam TRng Input range type
 * @tparam TWork Working range type
 * @tparam FProject Projection function type with signature `(T const& ) -> TKey` where `T` is the
 * range value type.
 * @tparam TKey Key type
 * @param rng Input range
 * @param cpy Copy range
 * @param work Working range
 * @param min Minimum key value
 * @param max Maximum key value
 * @param fProject Projection function to extract keys from elements of `rng` (default: identity)
 * @post The unsorted elements in `rng` are in `cpy`.
 */
template <
    std::ranges::random_access_range TRng,
    std::ranges::random_access_range TWork,
    class FProject = std::identity,
    std::integral TKey =
        std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TRng>>>>
    requires std::integral<TKey> and std::integral<std::ranges::range_value_t<TWork>>
void StableCountingSort(
    TRng&& rng,
    TRng&& cpy,
    TWork&& work,
    TKey min,
    TKey max,
    FProject fProject = {})
{
    using SizeType = std::ranges::range_size_t<TRng>;
    SizeType n     = std::ranges::size(rng);
    if (n == 0)
        return;
    assert(std::ranges::size(cpy) >= n);
    auto wb         = std::ranges::begin(work);
    auto we         = wb + (max - min + 1);
    using CountType = std::ranges::range_value_t<TWork>;
    std::fill(wb, we, CountType(0));
    for (SizeType i = 0; i < n; ++i)
        ++work[fProject(rng[i]) - min];
    std::exclusive_scan(wb, we, wb, CountType(0));
    for (SizeType i = 0; i < n; ++i)
        cpy[work[fProject(rng[i]) - min]++] = rng[i];
    using std::swap;
    swap(cpy, rng);
}

} // namespace pbat::common

#endif // PBAT_COMMON_COUNTINGSORT_H
