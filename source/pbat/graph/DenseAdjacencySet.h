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
     * @return Size of (A and B)
     * @pre `B` is sorted, contains no duplicates and has all values non-negative
     * @post `B`'s elements that are already in A are swapped to the back of B.
     * @post If `k >= |A and B|` then `Data<TData>(k)` is a new (default-constructed) data entry
     */
    template <std::ranges::random_access_range TIncomingAdjacencies>
        requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
    TIndex Assign(TIncomingAdjacencies&& B);
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
     * @brief Clear the adjacency set.
     */
    void Clear();
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
    /**
     * @brief Get the i^{th} adjacency (u,v)
     * @param i Index of adjacency
     * @return Tuple-like (u,v)
     */
    auto Adjacency(TIndex i) const
    {
        return std::tie(std::get<0>(mAdjacencies[i]), std::get<1>(mAdjacencies[i]));
    }
    /**
     * @brief Get the i^{th} weighted adjacency (u,v,k)
     * @param i Index of weighted adjacency
     * @return Tuple-like (u,v,k)
     */
    auto WeightedAdjacency(TIndex i) const
    {
        return std::tie(
            std::get<0>(mAdjacencies[i]),
            std::get<1>(mAdjacencies[i]),
            mIdToData[std::get<2>(mAdjacencies[i])]);
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
inline TIndex DenseAdjacencySet<TIndex, T...>::Assign(TIncomingAdjacencies&& B_)
{
    auto const fProj = [](auto&& tup) {
        return std::tie(std::get<0>(tup), std::get<1>(tup));
    };
    assert(std::ranges::is_sorted(B_, std::ranges::less{}, fProj));
    assert(std::ranges::adjacent_find(B_, std::ranges::equal_to{}, fProj) == std::ranges::end(B_));
    // 1. Swap elements of (A and B) to the front of A, and elements of (B \ A) to the front of B.
    auto abegin = std::ranges::begin(mAdjacencies);
    auto bbegin = std::ranges::begin(B_);
    auto aend   = std::ranges::end(mAdjacencies);
    auto bend   = std::ranges::end(B_);
    auto ahead = abegin, atail = abegin;
    auto bhead = bbegin, btail = bbegin;
    while (ahead != aend and bhead != bend)
    {
        auto cmp = fProj(*ahead) <=> fProj(*bhead);
        if (cmp == 0)
        {
            ++bhead;
            std::iter_swap(atail++, ahead++);
        }
        else if (cmp < 0)
        {
            ++ahead;
        }
        else // (cmp > 0)
        {
            std::iter_swap(btail++, bhead++);
        }
    }
    while (bhead != bend)
        std::iter_swap(btail++, bhead++);
    // 2. Remove elements from (A \ B)
    std::for_each(atail, aend, [&](auto&& tup) {
        // Remove data entries associated with adjacency via swap/remove
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
    });
    mAdjacencies.erase(atail, aend);
    std::size_t mid = mAdjacencies.size(); // keep note of end of (A and B)
    // 3. Add elements from (B \ A)
    std::for_each(bbegin, btail, [&](auto&& tup) {
        // Append a new data entry and create its indirection
        auto const& [u, v] = fProj(tup);
        auto id            = mIdToData.size();
        auto c             = mDataToId.size();
        mIdToData.push_back(static_cast<TIndex>(c));
        mDataToId.push_back(static_cast<TIndex>(id));
        mAdjacencies.push_back(std::make_tuple(u, v, id));
        std::apply([](auto&&... data) { (data.emplace_back(), ...); }, mData);
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
inline void DenseAdjacencySet<TIndex, T...>::Clear()
{
    mAdjacencies.clear();
    mCpy.clear();
    mIdToData.clear();
    mDataToId.clear();
    mPrefix.clear();
    std::apply([](auto&&... data) { (data.clear(), ...); }, mData);
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
