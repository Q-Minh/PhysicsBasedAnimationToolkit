/**
 * @file DenseAdjacencySet.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Dynamic adjacency set with per-edge associated data
 * @date 2026-02-10
 *
 * @copyright Copyright (c) 2026
 */

#ifndef PBAT_GRAPH_ADJACENCYSET_H
#define PBAT_GRAPH_ADJACENCYSET_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat {
namespace graph {

template <common::CIndex TIndex, class... T>
class DenseAdjacencySet
{
  public:
    using SelfType  = DenseAdjacencySet<TIndex, T...>;
    using IndexType = TIndex;
    /**
     * @brief Reserve space for the adjacency set.
     * @param nAdjacencies Expected number of adjacencies
     * @param nSourceVertices Expected number of source vertices
     */
    void Reserve(std::size_t nAdjacencies, std::size_t nSourceVertices = 0);
    /**
     * @brief Computes C <- B and (A or B) = B, where A is *this pre-assignment, and C is *this
     * post-assignment.
     *
     * This set's state is updated as follows:
     * - (A and B) -> unchanged
     * - (A and not B) -> removed
     * - (not A and B) -> added
     *
     * @tparam TIncomingAdjacencies Random access range over integer pair-like values
     * @param B Right operand
     * @param bTryRestoreB Whether to attempt restoring B's elements after modification. Only
     * applies if `std::is_signed_v<TIndex>`
     * @return Size of (A and B)
     * @pre `B` is sorted, contains no duplicates and has all values non-negative
     * @post `B`'s elements that are already in A have their first tuple-element modified if `B` has
     * elements of unsigned integer type or `bTryRestoreB` is false.
     * @post If `k >= |A and B|` then `Data<TData>(k)` is a new (default-constructed) data entry
     */
    template <std::ranges::random_access_range TIncomingAdjacencies>
        requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
    TIndex Assign(TIncomingAdjacencies&& B, bool bTryRestoreB = true);
    /**
     * @brief Finalize the adjacency set by computing its prefix sum over edge source vertices.
     * @param nSourceVertices Number of source vertices
     */
    void Finalize(std::size_t nSourceVertices = std::size_t(0));
    /**
     * @brief Compact the indirection table ids for memory efficiency
     */
    void CompactIds();
    /**
     * @brief Get the number of edges in the adjacency set.
     * @return std::size_t
     */
    std::size_t Size() const { return mAdjacencies.size(); }
    /**
     * @brief Get the number of source vertices of the graph from which *this is a sub-graph.
     * @return std::size_t
     * @pre `Finalize()` has been called
     */
    std::size_t NumSourceVertices() const { return mPrefix.size() > 0 ? mPrefix.size() - 1 : 0; }
    /**
     * @brief Check if an edge exists in the adjacency set, and if so, return its associated data
     * index.
     * @param u Source vertex
     * @param v Target vertex
     * @param bUseBinarySearch Whether to use binary search for lookup
     * @return If (u,v) is in the set, return `0 <= k < Size()`. Otherwise, return `-1` if
     * std::is_signed_v<TIndex> else std::numeric_limits<TIndex>::max().
     * @pre `Finalize()` has been called
     */
    TIndex Has(TIndex u, TIndex v, bool bUseBinarySearch = true) const;
    /**
     * @brief Visit each adjacency of a source vertex.
     * @tparam FOnAdjacency Callable with signature `void(TIndex u, TIndex v)` or `void(TIndex u,
     * TIndex v, TIndex k)` where `k` indexes into Data<TData>(), depending on if data is requested
     * or not.
     * @param u Source vertex
     * @param f Callback
     * @pre `Finalize()` has been called
     */
    template <class FOnAdjacency>
    void ForEach(TIndex u, FOnAdjacency&& f) const;
    /**
     * @brief Visit adjacency in the set.
     * @tparam FOnAdjacency Callable with signature `void(TIndex u, TIndex v)` or `void(TIndex u,
     * TIndex v, TIndex k)` where `k` indexes into Data<TData>(), depending on if data is requested
     * or not.
     * @param f Callback
     */
    template <class FOnAdjacency>
    void ForEach(FOnAdjacency&& f) const;
    /**
     * @brief Get a const reference to the edge data of type TData.
     * @tparam TData Type of the edge data
     * @return std::vector<TData> const&
     */
    template <class TData>
    std::vector<TData> const& Data() const
    {
        return std::get<std::vector<TData>>(mData);
    }
    /**
     * @brief Get a mutable reference to the edge data of type TData.
     * @tparam TData Type of the edge data
     * @return std::vector<TData>&
     */
    template <class TData>
    std::vector<TData>& Data()
    {
        return std::get<std::vector<TData>>(mData);
    }
    /**
     * @brief Get a const reference to the edge data of type TData.
     * @tparam TData Type of the edge data
     * @param k Index of the edge
     * @return std::vector<TData> const&
     */
    template <class TData>
    TData const& Data(TIndex k) const
    {
        return Data<TData>()[k];
    }
    /**
     * @brief Get a mutable reference to the edge data of type TData.
     * @tparam TData Type of the edge data
     * @param k Index of the edge
     * @return std::vector<TData>&
     */
    template <class TData>
    TData& Data(TIndex k)
    {
        return Data<TData>()[k];
    }

  private:
    std::vector<std::tuple<TIndex, TIndex, TIndex>> mAdjacencies; ///< (source, target, id) tuples
    std::vector<std::tuple<TIndex, TIndex, TIndex>>
        mCpy;                            ///< Copy of the adjacency list for std::merge
    std::vector<TIndex> mPrefix;         ///< Prefix sums for efficient range queries
    std::vector<TIndex> mIdToData;       ///< id -> index into mData
    std::vector<TIndex> mDataToId;       ///< index into mData -> id
    std::tuple<std::vector<T>...> mData; ///< Per-edge associated data
};

template <common::CIndex TIndex, class... T>
inline void
DenseAdjacencySet<TIndex, T...>::Reserve(std::size_t nAdjacencies, std::size_t nSourceVertices)
{
    mAdjacencies.reserve(nAdjacencies);
    mCpy.reserve(nAdjacencies);
    mPrefix.reserve(nSourceVertices + 1);
    mIdToData.reserve(nAdjacencies);
    mDataToId.reserve(nAdjacencies);
    std::apply([&](auto&&... data) { (data.reserve(nAdjacencies), ...); }, mData);
}

template <common::CIndex TIndex, class... T>
template <std::ranges::random_access_range TIncomingAdjacencies>
    requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
inline TIndex DenseAdjacencySet<TIndex, T...>::Assign(TIncomingAdjacencies&& B_, bool bTryRestoreB)
{
    // Let A = *this
    bool constexpr bAreKeysSigned = std::is_signed_v<TIndex>;
    auto const fSetNone           = [](auto&& tup) {
        if constexpr (bAreKeysSigned)
            std::get<0>(tup) = -std::get<0>(tup) - 1;
        else
            std::get<0>(tup) = std::numeric_limits<TIndex>::max();
    };
    auto const fIsNone = [](auto&& tup) {
        if constexpr (bAreKeysSigned)
            return std::get<0>(tup) < 0;
        else
            return std::get<0>(tup) == std::numeric_limits<TIndex>::max();
    };
    auto const fProj = [](auto&& tup) {
        return std::tie(std::get<0>(tup), std::get<1>(tup));
    };
    auto A = std::views::transform(mAdjacencies, fProj);
    auto B = std::views::transform(B_, fProj);
    assert(std::ranges::is_sorted(B));
    assert(std::ranges::adjacent_find(B) == std::ranges::end(B));
    // 1. Mark elements in A to remove and elements in B that are already in A
    auto ait  = std::ranges::begin(A);
    auto bit  = std::ranges::begin(B);
    auto aend = std::ranges::end(A);
    auto bend = std::ranges::end(B);
    while (ait != aend and bit != bend)
    {
        auto cmp = *ait <=> *bit;
        if (cmp == 0)
        {
            fSetNone(*bit);
            ++ait;
            ++bit;
        }
        else if (cmp < 0)
        {
            fSetNone(*ait);
            ++ait;
        }
        else // (cmp > 0)
        {
            ++bit;
        }
    }
    for (; ait != aend; ++ait)
        fSetNone(*ait);
    // 2. Remove elements from A while preserving sorted order
    std::erase_if(mAdjacencies, [&](auto&& tup) {
        bool const bRemove = fIsNone(tup);
        if (bRemove)
        {
            // Remove data entries associated with adjacency i via swap/remove
            auto const id = std::get<2>(tup);
            auto c        = mIdToData[id];
            auto last     = mDataToId.size() - 1;
            using std::swap;
            std::apply([&](auto&&... data) { (swap(data[c], data[last]), ...); }, mData);
            // Update indirection for the element that was at 'last'
            auto movedId       = mDataToId[last];
            mIdToData[movedId] = c;
            mDataToId[c]       = movedId;
            // Shrink the data array
            mDataToId.pop_back();
            std::apply([](auto&&... data) { (data.pop_back(), ...); }, mData);
        }
        return bRemove;
    });
    auto mid = mAdjacencies.size(); // keep note of end of (A and B)
    // 3. Stack (B \ A) contiguously after (A and B)
    std::ranges::for_each(B, [&](auto&& tup) {
        if (not fIsNone(tup))
        {
            // Append a new data entry and create its indirection
            auto const& [u, v] = tup;
            auto id            = mIdToData.size();
            auto c             = mDataToId.size();
            mIdToData.push_back(static_cast<TIndex>(c));
            mDataToId.push_back(static_cast<TIndex>(id));
            mAdjacencies.push_back(std::make_tuple(u, v, id));
            std::apply([](auto&&... data) { (data.emplace_back(), ...); }, mData);
        }
    });
    // 4. Merge (A and B) with (B \ A)
    std::ranges::merge(
        std::ranges::subrange(mAdjacencies.begin(), mAdjacencies.begin() + mid),
        std::ranges::subrange(mAdjacencies.begin() + mid, mAdjacencies.end()),
        std::back_inserter(mCpy),
        std::ranges::less{},
        fProj,
        fProj);
    using std::swap;
    swap(mAdjacencies, mCpy);
    mCpy.clear();
    // 5. If keys are signed, then restore B_ via the relation u' = -u - 1 <=> u = -u' - 1
    if (bAreKeysSigned and bTryRestoreB)
        std::ranges::for_each(B_, [&](auto&& tup) {
            if (fIsNone(tup))
                std::get<0>(tup) = -std::get<0>(tup) - 1;
        });
    return mid;
}

template <common::CIndex TIndex, class... T>
inline void DenseAdjacencySet<TIndex, T...>::Finalize(std::size_t nSourceVertices)
{
    if (nSourceVertices == 0)
    {
        if (Size() == 0)
            return;
        auto sources = mAdjacencies |
                       std::views::transform([](auto&& tup) -> TIndex { return std::get<0>(tup); });
        auto max        = *std::ranges::max_element(sources);
        nSourceVertices = static_cast<std::size_t>(max) + 1;
    }
    mPrefix.resize(nSourceVertices + 1, TIndex(0));
    for (auto const& tup : mAdjacencies)
        ++mPrefix[std::get<0>(tup)];
    std::exclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin(), TIndex(0));
}

template <common::CIndex TIndex, class... T>
inline void DenseAdjacencySet<TIndex, T...>::CompactIds()
{
    std::size_t const n = mAdjacencies.size();
    // Remap each triplet's id to its current data index
    for (auto& [tu, tv, tid] : mAdjacencies)
        tid = mIdToData[tid];
    // Rebuild indirection as identity: id == data index
    mIdToData.resize(n);
    mDataToId.resize(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        mIdToData[i] = static_cast<TIndex>(i);
        mDataToId[i] = static_cast<TIndex>(i);
    }
}

template <common::CIndex TIndex, class... T>
inline TIndex DenseAdjacencySet<TIndex, T...>::Has(TIndex u, TIndex v, bool bUseBinarySearch) const
{
    TIndex k = std::is_signed_v<TIndex> ? TIndex(-1) : std::numeric_limits<TIndex>::max();
    if (u >= NumSourceVertices())
        return k;
    auto adjacenciesFromU = std::ranges::subrange(
        mAdjacencies.begin() + mPrefix[u],
        mAdjacencies.begin() + mPrefix[u + 1]);
    auto const fProj = [](auto&& tup) {
        return std::get<1>(tup);
    };
    auto it = bUseBinarySearch ?
                  std::ranges::lower_bound(adjacenciesFromU, v, std::ranges::less{}, fProj) :
                  std::ranges::find(adjacenciesFromU, v, fProj);
    if (it != std::ranges::end(adjacenciesFromU))
    {
        auto id = std::get<2>(*it);
        k       = mIdToData[id];
    }
    return k;
}

template <common::CIndex TIndex, class... T>
template <class FOnAdjacency>
inline void DenseAdjacencySet<TIndex, T...>::ForEach(TIndex u, FOnAdjacency&& f) const
{
    if (u >= NumSourceVertices())
        return;
    auto adjacenciesFromU = std::ranges::subrange(
        mAdjacencies.begin() + mPrefix[u],
        mAdjacencies.begin() + mPrefix[u + 1]);
    std::ranges::for_each(adjacenciesFromU, [&](auto&& tup) {
        auto const& [u, v, id] = tup;
        if constexpr (
            std::is_invocable_v<FOnAdjacency, decltype(u), decltype(v), decltype(mIdToData[id])>)
        {
            f(u, v, mIdToData[id]);
        }
        else
        {
            f(u, v);
        }
    });
}

template <common::CIndex TIndex, class... T>
template <class FOnAdjacency>
inline void DenseAdjacencySet<TIndex, T...>::ForEach(FOnAdjacency&& f) const
{
    for (auto const& [u, v, id] : mAdjacencies)
    {
        if constexpr (
            std::is_invocable_v<FOnAdjacency, decltype(u), decltype(v), decltype(mIdToData[id])>)
        {
            f(u, v, mIdToData[id]);
        }
        else
        {
            f(u, v);
        }
    }
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCYSET_H
