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
#include <concepts>
#include <iterator>
#include <limits>
#include <numeric>
#include <type_traits>

namespace pbat::common {

/**
 * @brief Counting sort
 * @tparam TWorkBegin Iterator type to the beginning of the work array
 * @tparam TWorkEnd Iterator type to the end of the work array
 * @tparam TValuesBegin Iterator type to the beginning of the values
 * @tparam TValuesEnd Iterator type to the end of the values
 * @tparam FKey Key accessor callable with signature `TKey(T)`
 * @tparam T Value type
 * @tparam TKey Key type
 * @param wb Iterator to the beginning of the work array
 * @param we Iterator to the end of the work array
 * @param vb Iterator to the beginning of values
 * @param ve Iterator to the end of values
 * @param keyMin Minimum key value
 * @param fKey Key accessor callable
 * @pre `*(wb+i) == 0` for all `i in [0, std::distance(wb,we))`
 */
template <
    std::random_access_iterator TWorkBegin,
    std::random_access_iterator TWorkEnd,
    std::random_access_iterator TValuesBegin,
    std::random_access_iterator TValuesEnd,
    class FKey,
    class T    = typename std::iterator_traits<TValuesBegin>::value_type,
    class TKey = typename std::invoke_result_t<FKey, T>>
void CountingSort(
    TWorkBegin wb,
    TWorkEnd we,
    TValuesBegin vb,
    TValuesEnd ve,
    TKey keyMin = std::numeric_limits<TKey>::max(),
    FKey fKey   = [](T const& key) { return key; })
{
    using IndexType = std::iterator_traits<TWorkBegin>::value_type;
    static_assert(
        std::is_integral_v<IndexType> and not std::is_same_v<IndexType, bool>,
        "Work index must be range over integers");
    using KeyType = std::invoke_result_t<FKey, T>;
    static_assert(std::is_integral_v<KeyType>, "Key type must be integral");
    auto const n = std::distance(vb, ve);
    if (n == 0)
        return;
    // Find key offset
    if (keyMin == std::numeric_limits<KeyType>::max())
        for (auto it = vb; it != ve; ++it)
            keyMin = std::min(keyMin, fKey(*it));
    // Count occurrences
    for (auto it = vb; it != ve; ++it)
    {
        KeyType key = fKey(*it);
        KeyType j   = key - keyMin;
        ++(*(wb + j));
    }
    // Compute prefix sum
    std::exclusive_scan(wb, we, wb, IndexType(0));
    // Sort in place.
    // NOTE: Taken from
    // [SO](https://stackoverflow.com/questions/15682100/sorting-in-linear-time-and-in-place)
    for (auto i = n - 1; i >= 0; --i)
    {
        T val       = *(vb + i);
        KeyType key = fKey(val) - keyMin;
        IndexType j = *(wb + key); // counts[key]
        if (j < i)
        {
            do
            {
                ++(*(wb + key));           // ++counts[key]
                std::swap(val, *(vb + j)); // swap(val, a[j])
                key = fKey(val) - keyMin;
                j   = *(wb + key); // j <- counts[key]
            } while (j < i);
            // Move final value into place.
            *(vb + i) = val;
        }
    }
}

/**
 * @brief
 * @tparam FKey
 * @tparam TValuesBegin
 * @tparam TValuesEnd
 * @tparam TWorkBegin
 * @param vb
 * @param ve
 * @param wb
 * @param fKey
 */
template <
    std::random_access_iterator TValuesBegin,
    std::random_access_iterator TValuesEnd,
    std::random_access_iterator TWorkBegin,
    class FKey>
void PrefixSumFromSortedKeys(TValuesBegin vb, TValuesEnd ve, TWorkBegin wb, FKey fKey)
{
    using KeyType =
        std::invoke_result_t<FKey, typename std::iterator_traits<TValuesBegin>::value_type>;
    static_assert(std::is_integral_v<KeyType>, "Key type must be integral");
    using IndexType = std::iterator_traits<TWorkBegin>::value_type;
    static_assert(
        std::is_integral_v<IndexType> and not std::is_same_v<IndexType, bool>,
        "Work index must be range over integers");
    if (vb == ve)
        return;
    KeyType key = fKey(*vb);
    *wb         = IndexType(0);
    auto wit    = wb;
    for (auto it = vb; it != ve; ++it)
    {
        auto keyNext = fKey(*it);
        if (key != keyNext)
        {
            key        = keyNext;
            *(wit + 1) = *wit;
            ++wit;
        }
        ++(*wit);
    }
}

} // namespace pbat::common

#endif // PBAT_COMMON_COUNTINGSORT_H
