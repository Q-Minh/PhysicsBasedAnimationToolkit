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
#include <array>
#include <concepts>
#include <new>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

namespace pbat::common {

namespace detail {

} // namespace detail

/**
 * @brief Histogram for radix sort
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam nBits Number of bits to use for each bucket
 */
template <std::integral TCount = std::size_t, int nBits = 8>
struct alignas(std::hardware_destructive_interference_size) RadixSortCountArray
{
    static int constexpr Radix = 1 << nBits; ///< Radix
    static int constexpr kBits = nBits;      ///< Number of bits in a digit
    std::array<TCount, Radix> counts;        ///< Histogram bucket counts
    /**
     * @brief Access a histogram bucket count
     * @param i Index of the bucket
     * @return Reference to the bucket count
     */
    auto& operator[](auto i) { return counts[i]; }
    /**
     * @brief Access a histogram bucket count
     * @param i Index of the bucket
     * @return Reference to the bucket count
     */
    auto const& operator[](auto i) const { return counts[i]; }
    /**
     * @brief Access the beginning of the histogram bucket counts
     * @return Begin iterator
     */
    auto begin() { return counts.begin(); }
    /**
     * @brief Access the end of the histogram bucket counts
     * @return End iterator
     */
    auto end() { return counts.end(); }
    /**
     * @brief Reset the histogram bucket counts
     */
    void SetZero() { counts.fill(TCount(0)); }
};

/**
 * @brief Radix sort (single threaded)
 * @tparam TInOutRng Range type to sort
 * @tparam TCpyRng Range type for the copy buffer
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam FProject Projection function type
 * @tparam TKey Type of key to sort on
 * @param inout Input range to sort
 * @param cpy Copy buffer
 * @param work Working memory
 * @param fProject Projection function
 */
template <
    std::ranges::random_access_range TInOutRng,
    std::ranges::random_access_range TCpyRng,
    std::integral TCount = std::size_t,
    class FProject       = std::identity,
    std::integral TKey =
        std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TInOutRng>>>>
void RadixSort(
    TInOutRng&& inout,
    TCpyRng&& cpy,
    RadixSortCountArray<TCount>& work,
    FProject fProject = {})
{
    using SizeType = std::ranges::range_size_t<TInOutRng>;
    SizeType n     = std::ranges::size(inout);
    if (n == 0)
        return;
    assert(std::ranges::size(cpy) >= n);
    auto constexpr nKeyBits = sizeof(TKey) * 8;
    auto constexpr nPasses  = (nKeyBits + work.kBits - 1) / work.kBits;
    auto constexpr mask     = (work.Radix - 1);
    using std::swap;
    for (auto d = 0; d < nPasses; ++d)
    {
        int const shift = d * work.kBits;
        work.SetZero();
        for (SizeType i = 0; i < n; ++i)
            ++work[(fProject(inout[i]) >> shift) & mask];
        std::inclusive_scan(work.begin(), work.end(), work.begin());
        for (SizeType i = n; i-- > 0;)
        {
            TKey const k   = (fProject(inout[i]) >> shift) & mask;
            cpy[--work[k]] = inout[i];
        }
        swap(inout, cpy);
    }
}



} // namespace pbat::common

#endif // PBAT_COMMON_RADIXSORT_H