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

#include "Concepts.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <limits>
#include <new>
#include <numeric>
#include <ranges>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <type_traits>
#include <vector>

namespace pbat::common {

/**
 * @brief Histogram for radix sort
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam nBits Number of bits to use for each bucket
 */
template <std::integral TCount = std::size_t, int nBits = 8>
struct alignas(std::hardware_destructive_interference_size) RadixSortWorkspace
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
 * @param max Upper bound for the keys
 * @pre `0 <= fProject(inout[i]) <= max` for all `0 <= i < inout.size()`
 */
template <
    std::ranges::random_access_range TInOutRng,
    std::ranges::random_access_range TCpyRng,
    std::integral TCount = std::size_t,
    class FProject       = std::identity,
    std::integral TKey   = std::make_unsigned_t<
          std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TInOutRng>>>>>
void RadixSort(
    TInOutRng&& inout,
    TCpyRng&& cpy,
    RadixSortWorkspace<TCount>& work,
    FProject fProject = {},
    TKey max          = std::numeric_limits<TKey>::max())
{
    using SizeType = std::ranges::range_size_t<TInOutRng>;
    SizeType n     = std::ranges::size(inout);
    if (n == 0)
        return;
    assert(std::ranges::size(cpy) >= n);
    auto nKeyBits =
        sizeof(TKey) * 8 - std::countl_zero(static_cast<std::make_unsigned_t<TKey>>(max));
    auto nPasses        = (nKeyBits + work.kBits - 1) / work.kBits;
    auto constexpr mask = (work.Radix - 1);
    using std::swap;
    for (auto d = 0; d < nPasses; ++d)
    {
        int const shift = d * work.kBits;
        work.SetZero();
        for (SizeType i = 0; i < n; ++i)
            ++work[(fProject(inout[i]) >> shift) & mask];
        std::exclusive_scan(work.begin(), work.end(), work.begin(), TCount(0));
        for (SizeType i = 0; i < n; ++i)
        {
            TKey const k   = (fProject(inout[i]) >> shift) & mask;
            cpy[work[k]++] = inout[i];
        }
        swap(inout, cpy);
    }
}

/**
 * @brief Radix sort (multi-threaded)
 * @tparam TInOutRng Range type to sort
 * @tparam TCpyRng Range type for the copy buffer
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam FProject Projection function type
 * @tparam TKey Type of key to sort on
 * @param inout Input range to sort
 * @param cpy Copy buffer
 * @param work Working memory per-thread
 * @param fProject Projection function
 * @param max Upper bound for the keys
 * @pre `0 <= fProject(inout[i]) <= max` for all `0 <= i < inout.size()`
 */
template <
    std::ranges::random_access_range TInOutRng,
    std::ranges::random_access_range TCpyRng,
    std::integral TCount = std::size_t,
    class FProject       = std::identity,
    std::integral TKey   = std::make_unsigned_t<
          std::decay_t<std::invoke_result_t<FProject, std::ranges::range_value_t<TInOutRng>>>>>
void RadixSort(
    TInOutRng&& inout,
    TCpyRng&& cpy,
    std::vector<RadixSortWorkspace<TCount>>& lwork,
    FProject fProject = {},
    TKey max          = std::numeric_limits<TKey>::max())
{
    using SizeType = std::ranges::range_size_t<TInOutRng>;
    SizeType n     = std::ranges::size(inout);
    if (n == 0)
        return;
    assert(std::ranges::size(cpy) >= n);
    auto nRequestedThreads = lwork.size();
    RadixSortWorkspace<TCount> gwork;
    if (nRequestedThreads <= 1)
        RadixSort(inout, cpy, gwork, fProject, max);
    auto nKeyBits =
        sizeof(TKey) * 8 - std::countl_zero(static_cast<std::make_unsigned_t<TKey>>(max));
    auto nPasses        = (nKeyBits + gwork.kBits - 1) / gwork.kBits;
    auto constexpr mask = (gwork.Radix - 1);
    using std::swap;
    SizeType const nThreads = std::min(
        static_cast<SizeType>(nRequestedThreads),
        static_cast<SizeType>(std::thread::hardware_concurrency()));
    auto const nThreadWorkload = n / nThreads;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nThreads);

    for (auto d = 0; d < nPasses; ++d)
    {
        int const shift = d * gwork.kBits;
        // 1. Parallel (local) histogram/count
        tbb::parallel_for(
            tbb::blocked_range<SizeType>(SizeType(0), nThreads, 1),
            [&](tbb::blocked_range<SizeType> const& r) {
                for (SizeType t = r.begin(); t != r.end(); ++t)
                {
                    lwork[t].SetZero();
                    auto start = t * nThreadWorkload;
                    auto end   = std::min((t + 1) * nThreadWorkload, n);
                    for (SizeType i = start; i < end; ++i)
                        ++lwork[t][(fProject(inout[i]) >> shift) & mask];
                }
            },
            tbb::simple_partitioner{});
        // 2. Compute (global) histogram/counts
        gwork.SetZero();
        tbb::parallel_for(0, gwork.Radix, [&](int r) {
            for (SizeType t = 0; t < nThreads; ++t)
                gwork[r] += lwork[t][r];
        });
        // 3. Compute (exclusive) prefix sum over buckets
        std::exclusive_scan(gwork.begin(), gwork.end(), gwork.begin(), TCount(0));
        // 4. Parallel (local) offset computation
        tbb::parallel_for(0, gwork.Radix, [&](int r) {
            for (SizeType t = 0; t < nThreads; ++t)
            {
                auto lcount = lwork[t][r];
                lwork[t][r] = gwork[r];
                gwork[r] += lcount;
            }
        });
        // 5. Parallel (local) scatter
        tbb::parallel_for(
            tbb::blocked_range<SizeType>(SizeType(0), nThreads, 1),
            [&](tbb::blocked_range<SizeType> const& r) {
                for (SizeType t = r.begin(); t != r.end(); ++t)
                {
                    auto start = t * nThreadWorkload;
                    auto end   = std::min((t + 1) * nThreadWorkload, n);
                    for (SizeType i = start; i < end; ++i)
                    {
                        auto digit             = (fProject(inout[i]) >> shift) & mask;
                        cpy[lwork[t][digit]++] = inout[i];
                    }
                }
            },
            tbb::simple_partitioner{});
        // 6. Swap input and output buffers
        swap(cpy, inout);
    }
}

/**
 * @brief Radix sort (single threaded) specialization for ranges of tuples
 * @tparam TInOutRng Input/output range type
 * @tparam TCpyRng Copy range type
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam FProjects Projection function types
 * @tparam TKeys Key types
 * @param inout Input/output range
 * @param cpy Copy range
 * @param work Working memory
 * @param fProjects Projection functions
 * @param maxes Upper bounds for the keys
 */
template <
    std::ranges::random_access_range TInOutRng,
    std::ranges::random_access_range TCpyRng,
    std::integral TCount,
    CTupleLike FProjects,
    CTupleLike TKeys>
void RadixSort(
    TInOutRng&& inout,
    TCpyRng&& cpy,
    RadixSortWorkspace<TCount>& work,
    FProjects fProjects,
    TKeys maxes)
{
    using ValueType = std::ranges::range_value_t<TInOutRng>;
    static_assert(
        std::tuple_size_v<FProjects> == std::tuple_size_v<TKeys>,
        "Mismatched tuple sizes");
    static_assert(
        std::tuple_size_v<FProjects> >= std::tuple_size_v<ValueType>,
        "Mismatched tuple sizes");
    auto const fReverseForEach = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        std::size_t constexpr N = sizeof...(Is);
        (RadixSort(
             inout,
             cpy,
             work,
             [&](auto&& tup) { return std::get<N - 1 - Is>(fProjects)(std::get<N - 1 - Is>(tup)); },
             std::get<N - 1 - Is>(maxes)),
         ...);
    };
    fReverseForEach(std::make_index_sequence<std::tuple_size_v<FProjects>>{});
}

/**
 * @brief Radix sort (multi-threaded) specialized for ranges of tuple-like elements
 * @tparam TInOutRng Range type to sort
 * @tparam TCpyRng Range type for the copy buffer
 * @tparam TCount Type of integer to use for the histogram bucket counts
 * @tparam FProject Projection function type
 * @tparam TKey Type of key to sort on
 * @param inout Input range to sort
 * @param cpy Copy buffer
 * @param work Working memory per-thread
 * @param fProject Projection functions
 * @param maxes Upper bounds for the keys
 * @pre `0 <= fProject(inout[i]) <= max` for all `0 <= i < inout.size()`
 */
template <
    std::ranges::random_access_range TInOutRng,
    std::ranges::random_access_range TCpyRng,
    std::integral TCount,
    CTupleLike FProjects,
    CTupleLike TKeys>
void RadixSort(
    TInOutRng&& inout,
    TCpyRng&& cpy,
    std::vector<RadixSortWorkspace<TCount>>& lwork,
    FProjects fProjects,
    TKeys maxes)
{
    using ValueType = std::ranges::range_value_t<TInOutRng>;
    static_assert(
        std::tuple_size_v<FProjects> == std::tuple_size_v<TKeys>,
        "Mismatched tuple sizes");
    static_assert(
        std::tuple_size_v<FProjects> >= std::tuple_size_v<ValueType>,
        "Mismatched tuple sizes");
    auto const fReverseForEach = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        std::size_t constexpr N = sizeof...(Is);
        (RadixSort(
             inout,
             cpy,
             lwork,
             [&](auto&& tup) { return std::get<N - 1 - Is>(fProjects)(std::get<N - 1 - Is>(tup)); },
             std::get<N - 1 - Is>(maxes)),
         ...);
    };
    fReverseForEach(std::make_index_sequence<std::tuple_size_v<TKeys>>{});
}

} // namespace pbat::common

#endif // PBAT_COMMON_RADIXSORT_H