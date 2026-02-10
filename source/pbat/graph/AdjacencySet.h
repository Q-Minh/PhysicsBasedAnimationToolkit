/**
 * @file AdjacencySet.h
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

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <tbb/parallel_sort.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat {
namespace graph {

/**
 * @brief A unique adjacency triplet (u, v, id) where id indexes into an indirection table.
 *
 * @tparam TVertexIndex Integer type for vertex indices u and v.
 * @tparam TIdIndex     Integer type for the indirection id.
 *
 * The default (uint32_t, uint32_t) yields a 12-byte (96-bit) triplet. The index types can be
 * widened (e.g. int64_t) for very large graphs, or narrowed (e.g. int16_t) to improve cache
 * density when the vertex/id count is known to be small.
 */
template <common::CIndex TVertexIndex = std::uint32_t, common::CIndex TIdIndex = std::uint32_t>
struct AdjacencyTriplet
{
    using SelfType = AdjacencyTriplet<TVertexIndex, TIdIndex>; ///< Type of the triplet itself
    using VertexIndexType = TVertexIndex; ///< Index type of vertex endpoints u, v
    using IdIndexType     = TIdIndex;     ///< Index type of the indirection id

    TVertexIndex u; ///< First endpoint  (u < v)
    TVertexIndex v; ///< Second endpoint
    TIdIndex id;    ///< Index into the indirection table mIdToData

    /**
     * @brief Lexicographic ordering over (u, v) only; id is excluded.
     */
    friend auto operator<=>(SelfType const& lhs, SelfType const& rhs) noexcept
    {
        return std::tie(lhs.u, lhs.v) <=> std::tie(rhs.u, rhs.v);
    }

    /**
     * @brief Equality comparison (compares only u, v); id is excluded.
     */
    friend bool operator==(SelfType const& lhs, SelfType const& rhs) noexcept
    {
        return lhs.u == rhs.u and lhs.v == rhs.v;
    }
};

/**
 * @brief Options for AdjacencySet::Update()
 */
struct AdjacencySetUpdateOptions
{
    /**
     * @brief Policy controlling how Update() reconciles incoming adjacencies with the existing set.
     */
    enum class EUpdatePolicy {
        Overwrite,  ///< Replace the existing set: adjacencies not re-added are removed.
        AppendOnly, ///< Grow the set: incoming adjacencies are merged, existing ones are never
                    ///< removed.
    } eUpdatePolicy{EUpdatePolicy::Overwrite}; ///< Behavior of Update() (default: Overwrite)

    bool bAssumeUniqueIncoming{
        false}; ///< If true, the incoming adjacencies accumulated via Add() are assumed to be
                ///< already unique (i.e. no duplicates). This can be set to true to skip the
                ///< deduplication step in Update() when the user can guarantee that Add() is only
                ///< called with unique pairs.
    bool bAssumeSortedIncoming{
        false}; ///< If true, the incoming adjacencies accumulated via Add() are assumed to be
                ///< already sorted in canonical form (u < v). This can be set to true to skip the
                ///< sorting step in Update() when the user can guarantee that Add() is called with
                ///< pairs in canonical form and in sorted order.
    bool bBuildInverse{
        false}; ///< If true, build the inverse adjacency array keyed by v, enabling
                ///< InverseAdjacenciesOf(v). Costs one extra O(n) counting-sort pass.
};

/**
 * @brief Dynamic adjacency set that maintains a sorted collection of unique undirected edges (u,v),
 * each associated with user-defined data of type @p TData.
 *
 * The data structure supports incremental updates: the user first calls Add(u,v) for every
 * adjacency that should exist in the next state, then calls Update() to commit the changes.
 * Two update policies are available (see @ref EUpdatePolicy):
 * - **Overwrite**: computes the symmetric difference between the old and new adjacency sets.
 *   Callbacks are invoked for newly added and removed edges.
 * - **AppendOnly**: merges incoming adjacencies into the existing set without removing anything.
 *   Only the @p fOnAdded callback is invoked.
 *
 * @tparam TData        POD-like type stored for every live adjacency.
 * @tparam TVertexIndex Integer type for vertex indices u, v (default: uint32_t).
 * @tparam TIdIndex     Integer type for indirection ids     (default: uint32_t).
 *
 * clang-format off
 * ### Memory layout
 *
 * | Array                | Element type       | Purpose                                           |
 * |----------------------|--------------------|---------------------------------------------------|
 * | mAdjacencies         | TripletType        | Sorted unique (u,v,id) of the current set         |
 * | mExistingAdjacencies | TripletType        | Previous mAdjacencies (swapped in during Update)  |
 * | mData                | TData              | Per-adjacency payload                             |
 * | mIdToData            | TIdIndex           | id -> data index c (mData[c])                     |
 * | mDataToId            | TIdIndex           | data index c -> id                                |
 * | mIncomingAdjacencies | TripletType        | Incoming (possibly duplicated) user Add() calls   |
 * | mAdjacenciesToRemove | TripletType        | Old \ New (set difference)                        |
 * | mAdjacenciesToAdd    | TripletType        | New \ Old (set difference)                        |
 * | mPrefix              | TVertexIndex       | Prefix sum over u for fast AdjacenciesOf()        |
 * | mInverseAdjacencies  | TripletType        | Adjacencies sorted by v (opt-in via bBuildInverse)|
 * | mInversePrefix       | TVertexIndex       | Prefix sum over v for InverseAdjacenciesOf()      |
 * | mInverseCursors      | TVertexIndex       | Cursors for iterating over inverse adjacencies    |
 * clang-format on
 */
template <
    class TData,
    common::CIndex TVertexIndex = std::uint32_t,
    common::CIndex TIdIndex     = std::uint32_t>
class AdjacencySet
{
  public:
    using DataType        = TData;
    using VertexIndexType = TVertexIndex;
    using IdIndexType     = TIdIndex;
    using TripletType     = AdjacencyTriplet<TVertexIndex, TIdIndex>;

    AdjacencySet() = default;

    /**
     * @brief Initialise the set for @p n possible source vertices u ∈ [0, n).
     * @param n Number of possible source vertices
     */
    void Construct(TVertexIndex n);

    /**
     * @brief Preallocate memory for the expected number of adjacencies and incoming Add() calls,
     * minimising reallocations during Add() and Update().
     * @param nExpectedAdjacencies Expected number of unique adjacencies (edges) after Update()
     * @param nExpectedIncoming     Expected number of Add() calls before the next Update()
     */
    void Reserve(std::size_t nExpectedAdjacencies, std::size_t nExpectedIncoming);

    /**
     * @brief Register an incoming adjacency (u,v). The pair is canonicalised so that min(u,v)
     * comes first. Duplicates are allowed; they will be deduplicated during Update().
     * @param u First vertex index
     * @param v Second vertex index
     */
    void Add(TVertexIndex u, TVertexIndex v);

    /**
     * @brief Reconcile the incoming adjacencies accumulated via Add() with the existing set.
     *
     * When @p policy is @ref EUpdatePolicy::Overwrite (the default), this computes the symmetric
     * difference: adjacencies not re-Added since the last Update() are removed (invoking
     * @p fOnRemoved), and new adjacencies are added (invoking @p fOnAdded).
     *
     * When @p policy is @ref EUpdatePolicy::AppendOnly, existing adjacencies are always kept;
     * only genuinely new adjacencies are added (invoking @p fOnAdded). @p fOnRemoved is never
     * called.
     *
     * @tparam FOnAdded   Callable with signature `TData(TVertexIndex u, TVertexIndex v)`
     * @tparam FOnRemoved Callable with signature `void(TVertexIndex u, TVertexIndex v, TData& w)`
     * @param fOnAdded   Callback invoked for each newly added adjacency
     * @param fOnRemoved Callback invoked for each removed adjacency (Overwrite only)
     * @param policy     Update policy (default: Overwrite)
     */
    template <class FOnAdded, class FOnRemoved>
    void
    Update(FOnAdded&& fOnAdded, FOnRemoved&& fOnRemoved, AdjacencySetUpdateOptions options = {});

    /**
     * @brief Iterate over all adjacencies (u, v, w) where u is fixed.
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v, TData& w)`
     * @param u       Source vertex
     * @param fOnAdj  Callback invoked for each neighbour of u
     */
    template <class FOnAdjacency>
    void AdjacenciesOf(TVertexIndex u, FOnAdjacency&& fOnAdj);

    /**
     * @brief Const overload of AdjacenciesOf
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v,
     * TData const& w)`
     * @param u       Source vertex
     * @param fOnAdj  Callback invoked for each neighbour of u
     */
    template <class FOnAdjacency>
    void AdjacenciesOf(TVertexIndex u, FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Iterate over all adjacencies in the set, calling @p fOnAdj for each one.
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v,
     * TData const& w)`
     * @param fOnAdj Callback invoked for each adjacency
     */
    template <class FOnAdjacency>
    void ForAll(FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Iterate over all inverse adjacencies (v, u, w) where v is fixed.
     * Requires that the last Update() was called with bBuildInverse = true.
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex v, TVertexIndex u, TData& w)`
     * @param v       Target vertex
     * @param fOnAdj  Callback invoked for each source neighbour of v
     */
    template <class FOnAdjacency>
    void InverseAdjacenciesOf(TVertexIndex v, FOnAdjacency&& fOnAdj);

    /**
     * @brief Const overload of InverseAdjacenciesOf
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex v, TVertexIndex u,
     * TData const& w)`
     * @param v       Target vertex
     * @param fOnAdj  Callback invoked for each source neighbour of v
     */
    template <class FOnAdjacency>
    void InverseAdjacenciesOf(TVertexIndex v, FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Preallocate memory for the inverse adjacency array and its prefix sum.
     * Only useful when bBuildInverse will be set to true in Update().
     * @param nExpectedAdjacencies Expected number of unique adjacencies
     */
    void ReserveInverse(std::size_t nExpectedAdjacencies);

    /**
     * @brief Compact the indirection tables so that all live ids are dense in [0, Size()).
     *
     * After many add/remove cycles, mIdToData grows monotonically (one entry per id ever
     * allocated) while mData stays dense at Size() entries. The gap wastes memory.
     * This function remaps every triplet's id to its current data index, then shrinks
     * mIdToData and mDataToId to exactly Size() entries with identity values.
     *
     * Complexity: O(Size()).
     *
     * @note Call this periodically when memory pressure is a concern, not on every Update().
     */
    void CompactIds();

    /**
     * @brief Number of live adjacencies in the set
     * @return Number of adjacencies
     */
    std::size_t Size() const noexcept { return mAdjacencies.size(); }

    /**
     * @brief Number of source vertices
     * @return n passed to Construct()
     */
    TVertexIndex NumVertices() const noexcept
    {
        return std::max(static_cast<TVertexIndex>(mPrefix.size() - 1u), TVertexIndex{0});
    }

    /**
     * @brief Read-only access to the data array
     * @return Const reference to the data vector
     */
    std::vector<TData> const& Data() const noexcept { return mData; }

    /**
     * @brief Read-write access to the data array
     * @return Reference to the data vector
     */
    std::vector<TData>& Data() noexcept { return mData; }

  private:
    /**
     * @brief Allocate a fresh data slot and return its id.
     * @return The allocated id
     */
    TIdIndex AllocateId();

    /**
     * @brief Release the given id, compacting the data array via swap-and-pop.
     * @param id The id to release
     */
    void ReleaseId(TIdIndex id);

    /**
     * @brief Compute the prefix-sum array over u from the sorted mAdjacencies.
     */
    void ComputePrefix();

    /**
     * @brief Build the inverse adjacency array via counting sort by v.
     */
    void ComputeInverse();

    // -- Committed state (read by AdjacenciesOf / ForAll) --
    std::vector<TripletType> mAdjacencies; ///< Current sorted unique (u,v,id)
    std::vector<TVertexIndex> mPrefix;     ///< Prefix sum over u for fast AdjacenciesOf()

    // -- Inverse adjacencies (opt-in, read by InverseAdjacenciesOf) --
    std::vector<TripletType> mInverseAdjacencies; ///< Adjacencies sorted by v
    std::vector<TVertexIndex> mInversePrefix;     ///< Prefix sum over v
    std::vector<TVertexIndex> mInverseCursors;    ///< Write cursors for counting sort scatter

    // -- Per-adjacency data and indirection --
    std::vector<TData> mData;        ///< Per-adjacency payload (dense, exactly Size() entries)
    std::vector<TIdIndex> mIdToData; ///< id -> index into mData
    std::vector<TIdIndex> mDataToId; ///< index into mData -> id

    // -- Staging buffers (used only during Add / Update) --
    std::vector<TripletType> mIncomingAdjacencies; ///< Incoming Add() calls
    std::vector<TripletType> mExistingAdjacencies; ///< Previous adjacencies (scratch during Update)
    std::vector<TripletType> mAdjacenciesToAdd;    ///< New \ Old (set difference)
    std::vector<TripletType> mAdjacenciesToRemove; ///< Old \ New (set difference)
};

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::Construct(TVertexIndex n)
{
    // Committed state
    mAdjacencies.clear();
    mPrefix.resize(static_cast<std::size_t>(n) + 1u);
    // Inverse adjacencies
    mInverseAdjacencies.clear();
    mInversePrefix.clear();
    mInverseCursors.clear();
    // Per-adjacency data and indirection
    mData.clear();
    mIdToData.clear();
    mDataToId.clear();
    // Staging buffers
    mIncomingAdjacencies.clear();
    mExistingAdjacencies.clear();
    mAdjacenciesToAdd.clear();
    mAdjacenciesToRemove.clear();
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::Reserve(
    std::size_t nExpectedAdjacencies,
    std::size_t nExpectedIncoming)
{
    // Committed state
    mAdjacencies.reserve(nExpectedAdjacencies);
    // Per-adjacency data and indirection
    mData.reserve(nExpectedAdjacencies);
    mIdToData.reserve(nExpectedAdjacencies);
    mDataToId.reserve(nExpectedAdjacencies);
    // Staging buffers
    mIncomingAdjacencies.reserve(nExpectedIncoming);
    mExistingAdjacencies.reserve(nExpectedAdjacencies);
    mAdjacenciesToAdd.reserve(nExpectedIncoming);
    mAdjacenciesToRemove.reserve(nExpectedAdjacencies);
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::Add(TVertexIndex u, TVertexIndex v)
{
    // Canonicalise: ensure u < v to guarantee uniqueness of undirected edges
    mIncomingAdjacencies.push_back({std::min(u, v), std::max(u, v), TIdIndex{0}});
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdded, class FOnRemoved>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::Update(
    FOnAdded&& fOnAdded,
    FOnRemoved&& fOnRemoved,
    AdjacencySetUpdateOptions options)
{
    // 1. Move the current adjacency set into mExistingAdjacencies
    assert(mExistingAdjacencies.empty() and "mExistingAdjacencies must be empty before Update");
    std::swap(mAdjacencies, mExistingAdjacencies);

    // 2. Sort and deduplicate incoming adjacencies. mExistingAdjacencies is already sorted (it was
    // mAdjacencies which we maintain sorted).
    if (not options.bAssumeSortedIncoming)
        tbb::parallel_sort(mIncomingAdjacencies);
    if (not options.bAssumeUniqueIncoming)
        mIncomingAdjacencies.erase(
            std::unique(mIncomingAdjacencies.begin(), mIncomingAdjacencies.end()),
            mIncomingAdjacencies.end());

    // 3. Compute the adjacencies to add: incoming \ existing
    assert(mAdjacenciesToAdd.empty() and "mAdjacenciesToAdd must be empty before Update");
    std::ranges::set_difference(
        mIncomingAdjacencies,
        mExistingAdjacencies,
        std::back_inserter(mAdjacenciesToAdd));

    switch (options.eUpdatePolicy)
    {
        case AdjacencySetUpdateOptions::EUpdatePolicy::Overwrite: {
            // 4a. Set intersection: existing ∩ incoming  →  mAdjacencies (kept adjacencies)
            //     Copies existing triplets (preserving their ids) whose (u,v) also appear in
            //     incoming.
            std::ranges::set_intersection(
                mExistingAdjacencies,
                mIncomingAdjacencies,
                std::back_inserter(mAdjacencies));

            // 5a. Set difference: existing \ incoming  →  mAdjacenciesToRemove
            //     The output triplets carry the ids from mExistingAdjacencies.
            assert(
                mAdjacenciesToRemove.empty() and
                "mAdjacenciesToRemove must be empty before Update");
            std::ranges::set_difference(
                mExistingAdjacencies,
                mIncomingAdjacencies,
                std::back_inserter(mAdjacenciesToRemove));

            // 6a. Process removals: recycle ids and invoke fOnRemoved.
            for (auto const& [ru, rv, rid] : mAdjacenciesToRemove)
            {
                TIdIndex c = mIdToData[rid];
                fOnRemoved(ru, rv, mData[c]);
                ReleaseId(rid);
            }

            std::swap(mAdjacencies, mExistingAdjacencies);
            mAdjacencies.clear();
            break;
        }
        default: break;
    }

    // 7. Process additions: allocate ids and create data entries via fOnAdded.
    //    mAdjacenciesToAdd is already sorted (output of set_difference on sorted inputs),
    //    so we update each triplet's id in-place, preserving the sorted order.
    for (auto& [au, av, aid] : mAdjacenciesToAdd)
    {
        aid        = AllocateId();
        TIdIndex c = mIdToData[aid];
        mData[c]   = fOnAdded(au, av);
    }

    // 8. Merge the kept adjacencies (mAdjacencies) with the sorted additions
    //    (mAdjacenciesToAdd) into a single sorted array — O(n), no re-sort needed.
    //    We swap mAdjacencies into mExistingAdjacencies (already consumed) as scratch,
    //    then merge back into mAdjacencies.
    if (not mAdjacenciesToAdd.empty())
    {
        assert(mAdjacencies.empty() and "mAdjacencies must be empty before merging additions");
        mAdjacencies.reserve(mExistingAdjacencies.size() + mAdjacenciesToAdd.size());
        std::ranges::merge(
            mExistingAdjacencies,
            mAdjacenciesToAdd,
            std::back_inserter(mAdjacencies));
    }
    else
    {
        std::swap(mAdjacencies, mExistingAdjacencies);
    }

    // 9. Clear temporary buffers (but keep capacity for reuse)
    mExistingAdjacencies.clear();
    mIncomingAdjacencies.clear();
    mAdjacenciesToRemove.clear();
    mAdjacenciesToAdd.clear();

    // 10. Recompute prefix sum
    ComputePrefix();

    // 11. Optionally build inverse adjacency array
    if (options.bBuildInverse)
        ComputeInverse();
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::AdjacenciesOf(
    TVertexIndex u,
    FOnAdjacency&& fOnAdj)
{
    assert(static_cast<std::size_t>(u) + 1u < mPrefix.size() and "u out of range");
    TVertexIndex const begin = mPrefix[u];
    TVertexIndex const end   = mPrefix[static_cast<std::size_t>(u) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        auto const& [tu, tv, tid] = mAdjacencies[k];
        TIdIndex c                = mIdToData[tid];
        fOnAdj(tu, tv, mData[c]);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::AdjacenciesOf(
    TVertexIndex u,
    FOnAdjacency&& fOnAdj) const
{
    assert(static_cast<std::size_t>(u) + 1u < mPrefix.size() and "u out of range");
    TVertexIndex const begin = mPrefix[u];
    TVertexIndex const end   = mPrefix[static_cast<std::size_t>(u) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        auto const& [tu, tv, tid] = mAdjacencies[k];
        TIdIndex c                = mIdToData[tid];
        fOnAdj(tu, tv, mData[c]);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::ForAll(FOnAdjacency&& fOnAdj) const
{
    for (auto const& [tu, tv, tid] : mAdjacencies)
    {
        TIdIndex c = mIdToData[tid];
        fOnAdj(tu, tv, mData[c]);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::InverseAdjacenciesOf(
    TVertexIndex v,
    FOnAdjacency&& fOnAdj)
{
    assert(not mInversePrefix.empty() and "InverseAdjacenciesOf requires bBuildInverse = true");
    assert(static_cast<std::size_t>(v) + 1u < mInversePrefix.size() and "v out of range");
    TVertexIndex const begin = mInversePrefix[v];
    TVertexIndex const end   = mInversePrefix[static_cast<std::size_t>(v) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        auto const& [tv, tu, tid] = mInverseAdjacencies[k];
        TIdIndex c                = mIdToData[tid];
        fOnAdj(tv, tu, mData[c]);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::InverseAdjacenciesOf(
    TVertexIndex v,
    FOnAdjacency&& fOnAdj) const
{
    assert(not mInversePrefix.empty() and "InverseAdjacenciesOf requires bBuildInverse = true");
    assert(static_cast<std::size_t>(v) + 1u < mInversePrefix.size() and "v out of range");
    TVertexIndex const begin = mInversePrefix[v];
    TVertexIndex const end   = mInversePrefix[static_cast<std::size_t>(v) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        auto const& [tv, tu, tid] = mInverseAdjacencies[k];
        TIdIndex c                = mIdToData[tid];
        fOnAdj(tv, tu, mData[c]);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::ReserveInverse(std::size_t nExpectedAdjacencies)
{
    mInverseAdjacencies.reserve(nExpectedAdjacencies);
    mInversePrefix.reserve(static_cast<std::size_t>(NumVertices()) + 1u);
    mInverseCursors.reserve(static_cast<std::size_t>(NumVertices()) + 1u);
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::CompactIds()
{
    std::size_t const n = mAdjacencies.size();
    // Remap each triplet's id to its current data index
    for (auto& [tu, tv, tid] : mAdjacencies)
        tid = static_cast<TIdIndex>(mIdToData[tid]);
    // Also remap inverse adjacencies if they are populated
    for (auto& [tv, tu, tid] : mInverseAdjacencies)
        tid = static_cast<TIdIndex>(mIdToData[tid]);
    // Rebuild indirection as identity: id == data index
    mIdToData.resize(n);
    mDataToId.resize(n);
    for (std::size_t i = 0u; i < n; ++i)
    {
        mIdToData[i] = static_cast<TIdIndex>(i);
        mDataToId[i] = static_cast<TIdIndex>(i);
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
TIdIndex AdjacencySet<TData, TVertexIndex, TIdIndex>::AllocateId()
{
    TIdIndex id = static_cast<TIdIndex>(mIdToData.size());
    TIdIndex c  = static_cast<TIdIndex>(mData.size());
    mIdToData.push_back(c);
    mDataToId.push_back(id);
    mData.emplace_back(); // default-constructed; caller will overwrite
    return id;
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::ReleaseId(TIdIndex id)
{
    // Compact the data array by swapping the released slot with the last element.
    TIdIndex c    = mIdToData[id];
    TIdIndex last = static_cast<TIdIndex>(mData.size()) - TIdIndex{1};
    // Swap data
    std::swap(mData[c], mData[last]);
    // Update indirection for the element that was at 'last'
    TIdIndex movedId   = mDataToId[last];
    mIdToData[movedId] = c;
    mDataToId[c]       = movedId;
    // Shrink the data array
    mData.pop_back();
    mDataToId.pop_back();
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::ComputePrefix()
{
    TVertexIndex const n = NumVertices();
    // Reset all counts to zero
    std::fill(mPrefix.begin(), mPrefix.end(), TVertexIndex{0});
    // Count adjacencies per source vertex u
    for (auto const& [tu, tv, tid] : mAdjacencies)
    {
        assert(tu < n and "Adjacency source vertex out of range");
        ++mPrefix[static_cast<std::size_t>(tu)];
    }
    // Exclusive prefix sum: mPrefix[i] = sum of counts for vertices [0, i)
    std::exclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin(), TVertexIndex{0});
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void AdjacencySet<TData, TVertexIndex, TIdIndex>::ComputeInverse()
{
    TVertexIndex const n     = NumVertices();
    std::size_t const nAdj   = mAdjacencies.size();
    std::size_t const nSlots = static_cast<std::size_t>(n) + 1u;

    // 1. Resize inverse prefix to n+1 and zero it
    mInversePrefix.resize(nSlots);
    std::fill(mInversePrefix.begin(), mInversePrefix.end(), TVertexIndex{0});

    // 2. Count adjacencies per target vertex v
    for (auto const& [tu, tv, tid] : mAdjacencies)
        ++mInversePrefix[static_cast<std::size_t>(tv)];

    // 3. Exclusive prefix sum
    std::exclusive_scan(
        mInversePrefix.begin(),
        mInversePrefix.end(),
        mInversePrefix.begin(),
        TVertexIndex{0});

    // 4. Scatter: write (v, u, id) into the correct position via running offsets.
    //    We copy the prefix array into mInverseCursors as write cursors.
    mInverseCursors = mInversePrefix;
    mInverseAdjacencies.resize(nAdj);
    for (auto const& [tu, tv, tid] : mAdjacencies)
    {
        auto& pos                = mInverseCursors[static_cast<std::size_t>(tv)];
        mInverseAdjacencies[pos] = {tv, tu, tid};
        ++pos;
    }
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCYSET_H
