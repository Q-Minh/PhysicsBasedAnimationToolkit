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

#include "pbat/common/Concepts.h"

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat {
namespace graph {

/**
 * @brief Stateful adjacency set with per-edge associated data
 *
 * - This set is stateful in that any edge present in the set before and after a set operation
 * retains its associated data.
 * - It preserves unique edges only.
 * - Edges are stored in sorted order.
 * - Edge data is stored contiguously in memory (in order of insertion). After many edge removals
 * and insertions, no total order is guaranteed.
 *
 * @tparam TIndex Index type
 * @tparam T Data type to store per-edge
 */
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
     * @brief Computes C <- (A or B), where A is *this pre-assignment, and C is *this
     * post-assignment.
     * @tparam TIncomingAdjacencies Random access range over integer pair-like values
     * @param B Right operand
     * @return Size of (B \ A)
     * @pre `B` is sorted, contains no duplicates and has all values non-negative
     * @post `B`'s elements that are already in A are swapped to the back of B.
     * @post If `k >= |A and B|` then `Data<TData>(k)` is a new (default-constructed) data entry
     */
    template <std::ranges::random_access_range TIncomingAdjacencies>
        requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
    TIndex Union(TIncomingAdjacencies&& B);
    /**
     * @brief Computes C <- (A \ B), where A is *this pre-assignment, and C is *this
     * post-assignment.
     * @tparam TIncomingAdjacencies Random access range over integer pair-like values
     * @param B Right operand
     * @return Size of (A and B)
     * @pre `B` is sorted, contains no duplicates and has all values non-negative
     */
    template <std::ranges::random_access_range TIncomingAdjacencies>
        requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
    TIndex Subtract(TIncomingAdjacencies&& B);
    /**
     * @brief Remove edges satisfying a predicate.
     * @tparam FPredicate Callable with signature `bool(TIndex u, TIndex v, TIndex k)` where `k`
     * indexes into Data<TData>(). Return `true` to remove the edge.
     * @param f Predicate
     * @return Number of edges removed
     */
    template <class FPredicate>
    TIndex RemoveIf(FPredicate&& f);
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
     * @brief Get a const reference to the D^{th} edge data.
     * @tparam D Index of edge data
     * @return D^{th} data vector
     */
    template <std::size_t D>
    auto const& Data() const
    {
        return std::get<D>(mData);
    }
    /**
     * @brief Get a mutable reference to the edge data of type TData.
     * @tparam D Index of edge data
     * @return D^{th} data vector
     */
    template <std::size_t D>
    auto& Data()
    {
        return std::get<D>(mData);
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
     * @brief Get a const reference to the edge data of type TData.
     * @tparam D Index of edge data
     * @param k Index of the edge
     * @return D^{th} data vector
     */
    template <std::size_t D>
    auto const& Data(TIndex k) const
    {
        return Data<D>()[k];
    }
    /**
     * @brief Get a mutable reference to the edge data of type TData.
     * @tparam D Index of edge data
     * @param k Index of the edge
     * @return D^{th} data vector
     */
    template <std::size_t D>
    auto& Data(TIndex k)
    {
        return Data<D>()[k];
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
    /**
     * @brief Get read-only access to the adjacency list.
     * @return const reference to the internal (source, target, id) tuple vector
     */
    auto const& Adjacencies() const { return mAdjacencies; }
    /**
     * @brief Get read-only access to the prefix sums.
     * @return const reference to the prefix sums vector
     * @pre `Finalize()` has been called
     */
    auto const& Prefix() const { return mPrefix; }
    /**
     * @brief Construct the adjacency set from pre-built compact state.
     *
     * Assumes identity indirection (i.e. the id stored in each adjacency tuple equals the data
     * index). This is the state produced by `CompactIds()`.
     *
     * @param adjacencies Sorted (source, target, id) tuples where id == data index
     * @param prefix Prefix sums over source vertices
     * @param data Per-edge associated data, each vector of size `adjacencies.size()`
     */
    void Construct(
        std::vector<std::tuple<TIndex, TIndex, TIndex>> adjacencies,
        std::vector<TIndex> prefix,
        std::tuple<std::vector<T>...> data);

  protected:
    /**
     * @brief Add a new edge (u,v) with default-constructed data.
     * @param u Source vertex
     * @param v Target vertex
     */
    void AddEdge(TIndex u, TIndex v);
    /**
     * @brief Remove an edge by its internal id, using swap-and-pop on the data arrays.
     * @param id Internal id stored in the adjacency triplet
     */
    void RemoveEdgeData(TIndex id);

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
inline void DenseAdjacencySet<TIndex, T...>::Construct(
    std::vector<std::tuple<TIndex, TIndex, TIndex>> adjacencies,
    std::vector<TIndex> prefix,
    std::tuple<std::vector<T>...> data)
{
    mAdjacencies = std::move(adjacencies);
    mPrefix      = std::move(prefix);
    mData        = std::move(data);
    mCpy.clear();
    auto const n = mAdjacencies.size();
    mIdToData.resize(n);
    mDataToId.resize(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        mIdToData[i] = static_cast<TIndex>(i);
        mDataToId[i] = static_cast<TIndex>(i);
    }
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
    std::for_each(atail, aend, [&](auto&& tup) { RemoveEdgeData(std::get<2>(tup)); });
    mAdjacencies.erase(atail, aend);
    std::size_t mid = mAdjacencies.size(); // keep note of end of (A and B)
    // 3. Add elements from (B \ A)
    std::for_each(bbegin, btail, [&](auto&& tup) {
        auto const& [u, v] = fProj(tup);
        AddEdge(u, v);
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
template <std::ranges::random_access_range TIncomingAdjacencies>
    requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
inline TIndex DenseAdjacencySet<TIndex, T...>::Union(TIncomingAdjacencies&& B_)
{
    auto const fProj = [](auto&& tup) {
        return std::tie(std::get<0>(tup), std::get<1>(tup));
    };
    assert(std::ranges::is_sorted(B_, std::ranges::less{}, fProj));
    assert(std::ranges::adjacent_find(B_, std::ranges::equal_to{}, fProj) == std::ranges::end(B_));
    // 1. Swap the elements from (B \ A) to the front of B
    auto abegin = std::ranges::begin(mAdjacencies);
    auto aend   = std::ranges::end(mAdjacencies);
    auto bbegin = std::ranges::begin(B_);
    auto bend   = std::ranges::end(B_);
    auto ahead  = abegin;
    auto bhead = bbegin, btail = bbegin;
    while (ahead != aend and bhead != bend)
    {
        auto cmp = fProj(*ahead) <=> fProj(*bhead);
        if (cmp == 0)
        {
            ++bhead;
            ++ahead;
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
    // 2. Add elements from (B \ A)
    std::for_each(bbegin, btail, [&](auto&& tup) {
        auto const& [u, v] = fProj(tup);
        AddEdge(u, v);
    });
    auto nBnotA = std::distance(bbegin, btail);
    auto mid    = mAdjacencies.size() - nBnotA;
    // 3. Merge (A and B) with (B \ A)
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
    return static_cast<TIndex>(nBnotA);
}

template <common::CIndex TIndex, class... T>
template <std::ranges::random_access_range TIncomingAdjacencies>
    requires common::CTupleLike<std::ranges::range_value_t<TIncomingAdjacencies>>
inline TIndex DenseAdjacencySet<TIndex, T...>::Subtract(TIncomingAdjacencies&& B_)
{
    auto const fProj = [](auto&& tup) {
        return std::tie(std::get<0>(tup), std::get<1>(tup));
    };
    assert(std::ranges::is_sorted(B_, std::ranges::less{}, fProj));
    assert(std::ranges::adjacent_find(B_, std::ranges::equal_to{}, fProj) == std::ranges::end(B_));
    // 1. Swap elements of (A and not B) to the front of A.
    auto abegin = std::ranges::begin(mAdjacencies);
    auto bbegin = std::ranges::begin(B_);
    auto aend   = std::ranges::end(mAdjacencies);
    auto bend   = std::ranges::end(B_);
    auto ahead = abegin, atail = abegin;
    auto bhead = bbegin;
    while (ahead != aend and bhead != bend)
    {
        auto cmp = fProj(*ahead) <=> fProj(*bhead);
        if (cmp == 0)
        {
            ++bhead;
            ++ahead;
        }
        else if (cmp < 0)
        {
            std::iter_swap(atail++, ahead++);
        }
        else // (cmp > 0)
        {
            ++bhead;
        }
    }
    while (ahead != aend)
        std::iter_swap(atail++, ahead++);
    // 2. Remove elements from (A and B)
    auto nAandB = std::distance(atail, aend);
    std::for_each(atail, aend, [&](auto&& tup) { RemoveEdgeData(std::get<2>(tup)); });
    mAdjacencies.erase(atail, aend);
    return nAandB;
}

template <common::CIndex TIndex, class... T>
template <class FPredicate>
inline TIndex DenseAdjacencySet<TIndex, T...>::RemoveIf(FPredicate&& f)
{
    auto write = mAdjacencies.begin();
    auto read  = mAdjacencies.begin();
    auto end   = mAdjacencies.end();
    while (read != end)
    {
        auto const& [u, v, id] = *read;
        if (f(u, v, mIdToData[id]))
        {
            RemoveEdgeData(id);
        }
        else
        {
            if (write != read)
                *write = *read;
            ++write;
        }
        ++read;
    }
    auto nRemoved = static_cast<TIndex>(std::distance(write, end));
    mAdjacencies.erase(write, end);
    return nRemoved;
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

template <common::CIndex TIndex, class... T>
inline void DenseAdjacencySet<TIndex, T...>::AddEdge(TIndex u, TIndex v)
{
    auto id = static_cast<TIndex>(mIdToData.size());
    auto c  = static_cast<TIndex>(mDataToId.size());
    mIdToData.push_back(c);
    mDataToId.push_back(id);
    mAdjacencies.push_back(std::make_tuple(u, v, id));
    std::apply([](auto&&... data) { (data.emplace_back(), ...); }, mData);
}

template <common::CIndex TIndex, class... T>
inline void DenseAdjacencySet<TIndex, T...>::RemoveEdgeData(TIndex id)
{
    auto c    = mIdToData[id];
    auto last = static_cast<TIndex>(mDataToId.size() - 1);
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

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCYSET_H
