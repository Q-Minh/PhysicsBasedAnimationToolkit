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
#include <iterator>
#include <numeric>
#include <ranges>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat {
namespace graph {

namespace detail {

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

    TVertexIndex u; ///< First endpoint (source)
    TVertexIndex v; ///< Second endpoint (target)
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
 * @brief A unique adjacency pair (u, v) without associated data, used when TData is void.
 *
 * @tparam TVertexIndex Integer type for vertex indices u and v.
 */
template <common::CIndex TVertexIndex = std::uint32_t>
struct AdjacencyPair
{
    using SelfType        = AdjacencyPair<TVertexIndex>; ///< Type of the pair itself
    using VertexIndexType = TVertexIndex;                ///< Index type of vertex endpoints u, v

    TVertexIndex u; ///< First endpoint (source)
    TVertexIndex v; ///< Second endpoint (target)

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
 * @brief A minimal vector-like type that ignores all operations. Used as a placeholder when TData
 * is void.
 */
struct EmptyVector
{
    [[maybe_unused]]
    void reserve([[maybe_unused]] std::size_t)
    {
    }
    [[maybe_unused]]
    void clear() noexcept
    {
    }
    [[maybe_unused]]
    std::size_t size() const noexcept
    {
        return 0;
    }
    [[maybe_unused]]
    void emplace_back(auto&&...)
    {
    }
    [[maybe_unused]]
    void pop_back()
    {
    }
    [[maybe_unused]]
    void resize([[maybe_unused]] std::size_t)
    {
    }
    [[maybe_unused]]
    void push_back([[maybe_unused]] auto&&)
    {
    }
};

/**
 * @brief Sort adjacencies lexicographically by (u, v) using a two-pass counting sort (radix sort).
 *
 * This achieves O(n + nU + nV) time complexity, where n is the number of adjacencies,
 * nU is the number of source vertices, and nV is the number of target vertices.
 * This is faster than comparison-based O(n log n) sorting when n >> max(nU, nV).
 *
 * The algorithm performs two stable counting sort passes:
 * 1. First pass: sort by v (least significant key) → adjacencies grouped by v
 * 2. Second pass: sort by u (most significant key) → adjacencies sorted by (u, v)
 *
 * @tparam TAdjacency Adjacency entry type (AdjacencyPair or AdjacencyTriplet)
 * @param adjacencies Vector of adjacencies to sort in-place
 * @param temp Temporary buffer (will be resized to adjacencies.size())
 * @param counts Temporary count array (will be resized to max(nU, nV))
 * @param nU Number of source vertices (u ∈ [0, nU))
 * @param nV Number of target vertices (v ∈ [0, nV))
 */
template <class TAdjacency, common::CIndex TVertexIndex>
void CountingSortAdjacencies(
    std::vector<TAdjacency>& adjacencies,
    std::vector<TAdjacency>& temp,
    std::vector<TVertexIndex>& counts,
    TVertexIndex nU,
    TVertexIndex nV)
{
    std::size_t const n = adjacencies.size();
    if (n == 0u)
        return;

    temp.resize(n);
    counts.resize(static_cast<std::size_t>(std::max(nU, nV)));

    // Pass 1: Sort by v (least significant key)
    // Count occurrences of each v
    counts.assign(static_cast<std::size_t>(nV), TVertexIndex{0});
    for (TAdjacency const& adj : adjacencies)
        ++counts[adj.v];
    // Compute prefix sum (exclusive scan)
    std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), TVertexIndex{0});
    // Scatter into temp (stable: iterate forward)
    for (TAdjacency const& adj : adjacencies)
    {
        std::size_t const pos = counts[adj.v]++;
        temp[pos]             = adj;
    }

    // Pass 2: Sort by u (most significant key)
    // Count occurrences of each u
    counts.assign(static_cast<std::size_t>(nU), TVertexIndex{0});
    for (TAdjacency const& adj : temp)
        ++counts[adj.u];
    // Compute prefix sum (exclusive scan)
    std::exclusive_scan(counts.begin(), counts.end(), counts.begin(), TVertexIndex{0});
    // Scatter back into adjacencies (stable: iterate forward)
    for (TAdjacency const& adj : temp)
    {
        std::size_t const pos = counts[adj.u]++;
        adjacencies[pos]      = adj;
    }
}

} // namespace detail

/**
 * @brief Options for DenseAdjacencySet::Update()
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
                ///< already sorted lexicographically by (u, v). This can be set to true to skip
                ///< the sorting step in Update() when the user can guarantee that Add() is called
                ///< in sorted order.
    bool bUseParallelSort{
        false}; ///< If true, the sorting step in Update() (if needed) will be parallelized.
    std::int64_t nSourceVertices{
        -1}; ///< Number of source vertices (u values). When both nSourceVertices >= 0 and
             ///< nTargetVertices >= 0, Update() uses a linear-time two-pass counting sort
             ///< (radix sort: first by v, then by u) instead of comparison-based sorting.
             ///< This is faster when the number of adjacencies n >> max(nSourceVertices,
             ///< nTargetVertices). When < 0 (default), comparison-based sorting is used.
             ///< @note Counting sort requires additional memory: O(n) for a temporary buffer
             ///< (mCountingSortBuffer) and O(max(nSourceVertices, nTargetVertices)) for the
             ///< count array (mCountingSortCounts). These buffers are reused across Update() calls.
    std::int64_t nTargetVertices{
        -1}; ///< Number of target vertices (v values). See nSourceVertices for details.
};

/**
 * @brief Dynamic adjacency set that maintains a sorted collection of unique directed edges (u,v),
 * each associated with user-defined data of type @p TData.
 *
 * Edges are stored as-is (no canonicalization). To model undirected adjacencies, the user should
 * call both Add(u,v) and Add(v,u).
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
 */
template <
    class TData,
    common::CIndex TVertexIndex = std::uint32_t,
    common::CIndex TIdIndex     = std::uint32_t>
class DenseAdjacencySet
{
  public:
    using SelfType           = DenseAdjacencySet<TData, TVertexIndex, TIdIndex>;
    using DataType           = TData;
    using VertexIndexType    = TVertexIndex;
    using IdIndexType        = TIdIndex;
    using AdjacencyEntryType = std::conditional_t<
        std::is_void_v<DataType>,
        detail::AdjacencyPair<TVertexIndex>,
        detail::AdjacencyTriplet<TVertexIndex, TIdIndex>>;
    using DataContainerType =
        std::conditional_t<std::is_void_v<DataType>, detail::EmptyVector, std::vector<DataType>>;
    using IndirectionContainerType =
        std::conditional_t<std::is_void_v<DataType>, detail::EmptyVector, std::vector<IdIndexType>>;

    /**
     * @brief Preallocate memory for the expected number of adjacencies and incoming Add() calls,
     * minimising reallocations during Add() and Update().
     * @param nExpectedAdjacencies Expected number of unique adjacencies (edges) after Update()
     * @param nExpectedIncoming     Expected number of Add() calls before the next Update()
     */
    void Reserve(std::size_t nExpectedAdjacencies, std::size_t nExpectedIncoming);

    /**
     * @brief Register an incoming adjacency (u,v). Duplicates are allowed; they will be
     * deduplicated during Update().
     * @param u Source vertex index
     * @param v Target vertex index
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
     * @tparam FOnAdded   Callable with signature `TData(TVertexIndex u, TVertexIndex v)` or
     * `void(TVertexIndex u, TVertexIndex v)` if TData is void. The return value is used to
     * initialise the data for newly added adjacencies.
     * @tparam FOnRemoved Callable with signature `void(TVertexIndex u, TVertexIndex v, TData& w)`
     * or `void(TVertexIndex u, TVertexIndex v)` if TData is void.
     * @param fOnAdded   Callback invoked for each newly added adjacency
     * @param fOnRemoved Callback invoked for each removed adjacency (Overwrite only)
     * @param policy     Update policy (default: Overwrite)
     */
    template <class FOnAdded, class FOnRemoved>
    void
    Update(FOnAdded&& fOnAdded, FOnRemoved&& fOnRemoved, AdjacencySetUpdateOptions options = {});

    /**
     * @brief Reconcile the incoming adjacencies accumulated via Add() with the existing set, using
     * default callbacks that do nothing (for added adjacencies) or simply discard removed
     * adjacencies.
     *
     * @param options Update options (default: Overwrite policy, no assumptions about incoming
     * adjacencies)
     */
    void Update(AdjacencySetUpdateOptions options = {});

    /**
     * @brief Merge all adjacencies from @p other into this set, consuming @p other.
     *
     * Adjacencies present in @p other but not in this are added, moving their associated data
     * from @p other if applicable. Adjacencies already present in this are left unchanged (this's
     * data wins).
     *
     * When @p bAssumeDisjoint is true, the merge assumes that this and @p other share no
     * common adjacencies. This skips the O(n+m) set-difference check and directly merges all
     * triplets.
     *
     * Complexity: O(this->Size() + other.Size()).
     *
     * @post @p other is left empty (Size() == 0) but retains its allocated capacity.
     *
     * @tparam TOtherData    Data type of the other set, not necessarily the same as this's TData.
     * @param other          The adjacency set whose entries are merged into this (consumed)
     * @param bAssumeDisjoint If true, skip duplicate detection (caller guarantees no overlap)
     */
    template <class TOtherData>
    void Merge(
        DenseAdjacencySet<TOtherData, VertexIndexType, IdIndexType>& other,
        bool bAssumeDisjoint = false);

    /**
     * @brief Reduce a collection of AdjacencySets into a single set via parallel tree reduction.
     *
     * Each level of the tree merges pairs of sets in parallel, yielding O(A log T) total work
     * and O(A) span, where A is the total number of adjacencies and T is the number of sets.
     * The input sets are consumed (moved from). The iterator's value_type must be
     * DenseAdjacencySet.
     *
     * @post All input sets in [begin, end) are left in an empty state.
     *
     * @tparam TRandomIt     Random-access iterator over DenseAdjacencySet elements, not necessarily
     * of same data type.
     * @param begin          Iterator to the first DenseAdjacencySet
     * @param end            Iterator past the last DenseAdjacencySet
     * @param bAssumeInputDisjoint If true, skip duplicate detection in each input-input merge
     * @param bAssumeOutputDisjoint If true, skip duplicate detection of all inputs against the
     * output set in final merge
     * @return The merged DenseAdjacencySet
     */
    template <std::random_access_iterator TRandomIt>
    void Reduce(
        TRandomIt begin,
        TRandomIt end,
        bool bAssumeInputDisjoint  = false,
        bool bAssumeOutputDisjoint = false);

    /**
     * @brief Finalize the adjacency set by recomputing the prefix-sum array.
     *
     * Must be called once after any sequence of Update() and/or Merge() calls, before
     * using AdjacenciesOf(). Grows the prefix array if adjacencies reference source
     * vertices beyond the range established by Construct().
     *
     * @params n Number of source vertices (optional). If `n < 0`, it is inferred from the maximum
     * vertex index in mAdjacencies. If `n >= 0`, it is used to size the prefix array, and all
     * adjacencies with source vertex `u >= n` are considered invalid and ignored in
     * AdjacenciesOf().
     */
    void Finalize(std::make_signed_t<TVertexIndex> n = -1);

    /**
     * @brief Check if the adjacency (u, v) exists in the set, optionally retrieving its associated
     * data id.
     *
     * @param u Source vertex
     * @param v Target vertex
     * @param c Optional pointer to store the associated data id (if TData is not void)
     * @param bUseBinarySearch If true, use binary search over the sorted mAdjacencies. Otherwise,
     * perform a linear search. Prefer linear search when expected degree of u is small.
     * @return true if the adjacency exists, false otherwise
     * @pre Finalize() has been called to establish the prefix array for AdjacenciesOf() and enable
     * binary search.
     */
    bool
    Has(TVertexIndex u, TVertexIndex v, TIdIndex* c = nullptr, bool bUseBinarySearch = true) const;

    /**
     * @brief Return the degree of vertex u (number of outgoing adjacencies from u).
     * @param u Source vertex
     * @return Number of outgoing adjacencies from u
     * @pre Finalize() has been called to establish the prefix array
     */
    TVertexIndex Degree(TVertexIndex u) const;

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
     * TData& w)`
     * @param fOnAdj Callback invoked for each adjacency
     */
    template <class FOnAdjacency>
    void ForAll(FOnAdjacency&& fOnAdj);

    /**
     * @brief Const overload of ForAll
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v,
     * TData const& w)`
     * @param fOnAdj Callback invoked for each adjacency
     */
    template <class FOnAdjacency>
    void ForAll(FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Clear all vectors, resetting the set to an empty state while preserving
     * allocated capacity for reuse.
     */
    void Clear();

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
     * @brief Check if the set is empty (contains no adjacencies)
     * @return true if empty, false otherwise
     */
    bool Empty() const noexcept { return mAdjacencies.empty(); }

    /**
     * @brief Number of source vertices inferred from the prefix array
     * @return Number of source vertices, or 0 if Finalize() has not been called
     */
    TVertexIndex NumVertices() const noexcept
    {
        return mPrefix.empty() ? TVertexIndex{0} : static_cast<TVertexIndex>(mPrefix.size() - 1);
    }

    /**
     * @brief Read-only access to the data array
     * @return Const reference to the data vector
     */
    DataContainerType const& Data() const noexcept { return mData; }

    /**
     * @brief Read-write access to the data array
     * @return Reference to the data vector
     */
    DataContainerType& Data() noexcept { return mData; }

  private:
    template <class, common::CIndex, common::CIndex>
    friend class DenseAdjacencySet;

    /**
     * @brief Allocate a fresh data slot and return its id.
     * @return The allocated id
     */
    TIdIndex AllocateId();

    /**
     * @brief Release the given id, compacting the data array via swap-and-pop.
     * @param id The id to release
     */
    void ReleaseId([[maybe_unused]] TIdIndex id);

    /**
     * @brief Allocate ids and transfer data from @p other for each adjacency in
     * mAdjacenciesToAdd.
     *
     * Called by Merge() after the set-difference step has populated mAdjacenciesToAdd.
     * Specialise or branch on TData/TOtherData to control how data is moved.
     *
     * @tparam TOtherData Data type of the source set
     * @param other       The source set being merged from
     */
    template <class TOtherData>
    void AddAdjacencyDataFrom(DenseAdjacencySet<TOtherData, VertexIndexType, IdIndexType>& other);

    /**
     * @brief Release ids and invoke @p fOnRemoved for each adjacency in mAdjacenciesToRemove.
     *
     * Called by Update() after the set-difference step has populated mAdjacenciesToRemove.
     * When TData is void, only invokes fOnRemoved(u, v) and skips id/data operations.
     *
     * @tparam FOnRemoved Callback type
     * @param fOnRemoved  Invoked for each removed adjacency
     */
    template <class FOnRemoved>
    void RemoveOldAdjacencies(FOnRemoved&& fOnRemoved);

    /**
     * @brief Allocate ids and create data entries via @p fOnAdded for each adjacency in
     * mAdjacenciesToAdd.
     *
     * Called by Update() after the set-difference step has populated mAdjacenciesToAdd.
     * When TData is void, only invokes fOnAdded(u, v) and skips id/data operations.
     *
     * @tparam FOnAdded Callback type
     * @param fOnAdded  Invoked for each newly added adjacency
     */
    template <class FOnAdded>
    void AddNewAdjacencies(FOnAdded&& fOnAdded);

    // -- Committed state (read by AdjacenciesOf / ForAll) --
    std::vector<AdjacencyEntryType> mAdjacencies; ///< Current sorted unique (u,v,id)
    std::vector<TVertexIndex> mPrefix;            ///< Prefix sum over u for fast AdjacenciesOf()

    // -- Per-adjacency data and indirection --
    DataContainerType mData;            ///< Per-adjacency payload (dense, exactly Size() entries)
    IndirectionContainerType mIdToData; ///< id -> index into mData
    IndirectionContainerType mDataToId; ///< index into mData -> id

    // -- Staging buffers (used only during Add / Update) --
    std::vector<AdjacencyEntryType> mIncomingAdjacencies; ///< Incoming Add() calls
    std::vector<AdjacencyEntryType>
        mExistingAdjacencies; ///< Previous adjacencies (scratch during Update)
    std::vector<AdjacencyEntryType> mAdjacenciesToAdd;    ///< New \ Old (set difference)
    std::vector<AdjacencyEntryType> mAdjacenciesToRemove; ///< Old \ New (set difference)
    std::vector<AdjacencyEntryType> mCountingSortBuffer;  ///< Temporary buffer for counting sort
    std::vector<TVertexIndex> mCountingSortCounts;        ///< Count array for counting sort
};

/**
 * @brief Stores the reverse of a source DenseAdjacencySet of (u, v).
 *
 * This view enables efficient iteration over the reverse adjacency graph, where we query "which
 * sources u are adjacent to a given target v?" rather than "which targets v are adjacent to a
 * given source u?".
 *
 * The view is built from an existing DenseAdjacencySet via the Update() method using a linear-time
 * counting sort (O(n) where n is the number of adjacencies), exploiting the fact that the input
 * set is already sorted by (u, v).
 *
 * When the source set has associated data (TData is not void), the view stores triplets
 * (v, u, c) where c directly indexes into the source set's Data() array. When TData is void,
 * the view stores pairs (v, u) without the c index.
 *
 * @tparam TData        Data type of the source DenseAdjacencySet (void for no data)
 * @tparam TVertexIndex Integer type for vertex indices u, v (default: uint32_t)
 * @tparam TIdIndex     Integer type for data indices (default: uint32_t)
 *
 * @note This is a **view**: it does not own the data. The source DenseAdjacencySet must remain
 * valid and unmodified while this view is in use. If you need to access the data via this view,
 * you must pass the source set to methods like AdjacenciesOf() and ForAll().
 */
template <
    class TData,
    common::CIndex TVertexIndex = std::uint32_t,
    common::CIndex TIdIndex     = std::uint32_t>
class DenseReverseAdjacencySetView
{
  public:
    using SelfType           = DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>;
    using DataType           = TData;
    using VertexIndexType    = TVertexIndex;
    using IdIndexType        = TIdIndex;
    using SourceSetType      = DenseAdjacencySet<TData, TVertexIndex, TIdIndex>;
    using AdjacencyEntryType = std::conditional_t<
        std::is_void_v<DataType>,
        detail::AdjacencyPair<TVertexIndex>,
        detail::AdjacencyTriplet<TVertexIndex, TIdIndex>>;

    /**
     * @brief Build the reverse adjacency view from a finalized DenseAdjacencySet.
     *
     * Uses counting sort to reorder adjacencies by (v, u) in O(n) time, where n is the number of
     * adjacencies. The input set must have had Finalize() called on it.
     *
     * @param set The source adjacency set
     * @param n Number of target vertices `v`. If `v < 0`, inferred from max `v` in `set`.
     * @pre `Finalize()` has been called on `set` to establish its prefix array and enable counting
     * sort.
     */
    void Update(SourceSetType const& set, std::make_signed_t<TVertexIndex> n = -1);

    /**
     * @brief Check if the adjacency (u, v) exists in the view, optionally retrieving its
     * associated data index.
     *
     * @param u Source vertex
     * @param v Target vertex
     * @param c Optional pointer to store the data index (if TData is not void)
     * @param bUseBinarySearch If true, use binary search. Otherwise, linear search.
     * @return true if the adjacency exists, false otherwise
     */
    bool
    Has(TVertexIndex u, TVertexIndex v, TIdIndex* c = nullptr, bool bUseBinarySearch = true) const;

    /**
     * @brief Return the degree of vertex v (number of incoming adjacencies to v).
     * @param v Target vertex (from the perspective of this reverse view)
     * @return Number of adjacencies incident on v
     */
    TVertexIndex Degree(TVertexIndex v) const;

    /**
     * @brief Iterate over all adjacencies (u, v) incident on v.
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v)` if
     * `TData` is void, or `void(TVertexIndex u, TVertexIndex v, TIdIndex c)` if `TData` is not
     * void. The callback receives the source vertex `u`, the target vertex `v`, and optionally the
     * data index `c` for the adjacency.
     * @param v      Target vertex
     * @param fOnAdj Callback invoked for each source u adjacent to v
     */
    template <class FOnAdjacency>
    void AdjacenciesOf(TVertexIndex v, FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Iterate over all adjacencies in the view.
     *
     * @tparam FOnAdjacency Callable with signature `void(TVertexIndex u, TVertexIndex v)` when
     * TData is void, or `void(TVertexIndex u, TVertexIndex v, TIdIndex c)` when TData is not void.
     *
     * @param fOnAdj Callback invoked for each adjacency
     */
    template <class FOnAdjacency>
    void ForAll(FOnAdjacency&& fOnAdj) const;

    /**
     * @brief Clear all vectors, resetting the view to an empty state while preserving capacity.
     */
    void Clear();

    /**
     * @brief Number of adjacencies in the view
     * @return Number of adjacencies
     */
    std::size_t Size() const noexcept { return mAdjacencies.size(); }

    /**
     * @brief Check if the view is empty
     * @return true if empty, false otherwise
     */
    bool Empty() const noexcept { return mAdjacencies.empty(); }

    /**
     * @brief Number of target vertices inferred from the prefix array
     * @return Number of target vertices, or 0 if Update() has not been called
     */
    TVertexIndex NumVertices() const noexcept
    {
        return mPrefix.empty() ? TVertexIndex{0} : static_cast<TVertexIndex>(mPrefix.size() - 1);
    }

  private:
    std::vector<AdjacencyEntryType> mAdjacencies; ///< Adjacencies sorted by (v, u)
    std::vector<TVertexIndex> mPrefix;            ///< Prefix sum over v for fast AdjacenciesOf()
    std::vector<TVertexIndex> mCounts;            ///< Temporary storage for counting sort
};

// =============================================================================
// DenseAdjacencySet implementation
// ============================================================================

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Reserve(
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
    mAdjacenciesToRemove.reserve(nExpectedIncoming);
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Add(TVertexIndex u, TVertexIndex v)
{
    mIncomingAdjacencies.push_back({u, v});
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdded, class FOnRemoved>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Update(
    FOnAdded&& fOnAdded,
    FOnRemoved&& fOnRemoved,
    AdjacencySetUpdateOptions options)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.Update");

    // 1. Move the current adjacency set into mExistingAdjacencies
    assert(mExistingAdjacencies.empty() and "mExistingAdjacencies must be empty before Update");
    std::swap(mAdjacencies, mExistingAdjacencies);

    // 2. Sort and deduplicate incoming adjacencies. mExistingAdjacencies is already sorted (it was
    // mAdjacencies which we maintain sorted).
    if (not options.bAssumeSortedIncoming)
    {
        bool const bUseCountingSort =
            (options.nSourceVertices >= 0) and (options.nTargetVertices >= 0);
        if (bUseCountingSort)
        {
            detail::CountingSortAdjacencies(
                mIncomingAdjacencies,
                mCountingSortBuffer,
                mCountingSortCounts,
                static_cast<TVertexIndex>(options.nSourceVertices),
                static_cast<TVertexIndex>(options.nTargetVertices));
        }
        else if (options.bUseParallelSort)
        {
            tbb::parallel_sort(mIncomingAdjacencies);
        }
        else
        {
            std::ranges::sort(mIncomingAdjacencies);
        }
    }
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

            // 6a. Process removals and invoke fOnRemoved.
            RemoveOldAdjacencies(fOnRemoved);

            std::swap(mAdjacencies, mExistingAdjacencies);
            mAdjacencies.clear();
            break;
        }
        default: break;
    }

    // 7. Process additions: allocate ids and create data entries via fOnAdded.
    //    mAdjacenciesToAdd is already sorted (output of set_difference on sorted inputs),
    //    so we update each triplet's id in-place, preserving the sorted order.
    AddNewAdjacencies(fOnAdded);

    // 8. Merge the kept adjacencies (mAdjacencies) with the sorted additions
    //    (mAdjacenciesToAdd) into a single sorted array — O(n), no re-sort needed.
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
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
inline void
DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Update(AdjacencySetUpdateOptions options)
{
    this->Update([](auto&&...) {}, [](auto&&...) {}, options);
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Clear()
{
    // Committed state
    mAdjacencies.clear();
    mPrefix.clear();
    // Per-adjacency data and indirection
    mData.clear();
    mIdToData.clear();
    mDataToId.clear();
    // Staging buffers
    mIncomingAdjacencies.clear();
    mExistingAdjacencies.clear();
    mAdjacenciesToAdd.clear();
    mAdjacenciesToRemove.clear();
    mCountingSortBuffer.clear();
    mCountingSortCounts.clear();
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class TOtherData>
inline void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Merge(
    DenseAdjacencySet<TOtherData, VertexIndexType, IdIndexType>& other,
    bool bAssumeDisjoint)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.Merge");

    // 1. Determine which of other's adjacencies to add.
    assert(mAdjacenciesToAdd.empty() and "mAdjacenciesToAdd must be empty before Merge");
    if (bAssumeDisjoint)
    {
        // Fast path: all of other's adjacencies are new — skip set_difference entirely.
        if constexpr (std::is_same_v<TData, TOtherData>)
            std::swap(mAdjacenciesToAdd, other.mAdjacencies);
        else
            std::ranges::transform(
                other.mAdjacencies,
                std::back_inserter(mAdjacenciesToAdd),
                [](auto const& adj) { return AdjacencyEntryType{adj.u, adj.v}; });
    }
    else
    {
        // General path: other \ this
        if constexpr (std::is_same_v<TData, TOtherData>)
            std::ranges::set_difference(
                other.mAdjacencies,
                mAdjacencies,
                std::back_inserter(mAdjacenciesToAdd));
        else
            std::ranges::set_difference(
                std::views::transform(
                    other.mAdjacencies,
                    [](auto&& adj) -> AdjacencyEntryType { return {adj.u, adj.v}; }),
                mAdjacencies,
                std::back_inserter(mAdjacenciesToAdd));
    }

    // 2. Allocate ids and transfer data from other for each new adjacency.
    AddAdjacencyDataFrom(other);

    // 3. Merge sorted arrays
    if (not mAdjacenciesToAdd.empty())
    {
        assert(
            mExistingAdjacencies.empty() and
            "mExistingAdjacencies must be empty before merging additions");
        std::swap(mAdjacencies, mExistingAdjacencies);
        mAdjacencies.reserve(mExistingAdjacencies.size() + mAdjacenciesToAdd.size());
        std::ranges::merge(
            mExistingAdjacencies,
            mAdjacenciesToAdd,
            std::back_inserter(mAdjacencies));
    }

    // 4. Clear temporaries
    mExistingAdjacencies.clear();
    mAdjacenciesToAdd.clear();

    // 5. Leave other in an empty state, preserving its allocated capacity.
    other.Clear();
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <std::random_access_iterator TRandomIt>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Reduce(
    TRandomIt begin,
    TRandomIt end,
    bool bAssumeInputDisjoint,
    bool bAssumeOutputDisjoint)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.Reduce");

    using IterValueType = std::iter_value_t<TRandomIt>;
    using TOtherData    = typename IterValueType::DataType;
    static_assert(
        std::is_same_v<IterValueType, DenseAdjacencySet<TOtherData, VertexIndexType, IdIndexType>>,
        "Iterator value_type must be DenseAdjacencySet with compatible template parameters");
    auto const n = static_cast<std::size_t>(std::distance(begin, end));
    if (n == 0u)
        return;

    // Tree reduction: at each level, merge adjacent pairs in parallel.
    // stride = distance between a 'dst' and its 'src' partner.
    // After ceil(log2(n)) levels, begin[0] holds the fully merged result.
    for (std::size_t stride = 1u; stride < n; stride *= 2u)
    {
        std::size_t const step = stride * 2u;
        // Number of pairs at this level: every 'dst' at index k*step that has a
        // partner at k*step + stride (partner must be < n).
        std::size_t const nPairs = (n - stride + step - 1u) / step; // = ceil((n - stride) / step)
        tbb::parallel_for(
            std::size_t{0},
            nPairs,
            [&](std::size_t k) {
                std::size_t dst = k * step;
                std::size_t src = dst + stride;
                if (src < n)
                    begin[dst].Merge(begin[src], bAssumeInputDisjoint);
            },
            tbb::static_partitioner());
    }
    this->Merge(begin[0], bAssumeOutputDisjoint);
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Finalize(std::make_signed_t<TVertexIndex> n)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.Finalize");
    // Grow mPrefix if any adjacency source vertex exceeds the current range
    // (can happen after Merge with a larger set).
    if (not mAdjacencies.empty())
    {
        if (n < 0)
            n = static_cast<decltype(n)>(mAdjacencies.back().u) + 1;
        if (n >= mPrefix.size())
            mPrefix.resize(static_cast<std::size_t>(n) + 1u);
    }
    // Reset all counts to zero
    std::fill(mPrefix.begin(), mPrefix.end(), TVertexIndex{0});
    // Count adjacencies per source vertex u
    for (AdjacencyEntryType const& adj : mAdjacencies)
        ++mPrefix[static_cast<std::size_t>(adj.u)];
    // Exclusive prefix sum: mPrefix[i] = sum of counts for vertices [0, i)
    std::exclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin(), TVertexIndex{0});
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
inline bool DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Has(
    TVertexIndex u,
    TVertexIndex v,
    TIdIndex* c,
    bool bUseBinarySearch) const
{
    TVertexIndex const first = mPrefix[u];
    TVertexIndex const last  = mPrefix[static_cast<std::size_t>(u) + 1u];
    auto begin               = mAdjacencies.begin() + first;
    auto end                 = mAdjacencies.begin() + last;
    auto it                  = end;
    bool bFound{false};
    if (bUseBinarySearch)
    {
        it = std::lower_bound(begin, end, v, [](AdjacencyEntryType const& adj, TVertexIndex v_) {
            return adj.v < v_;
        });
        bFound = it != end and it->v == v;
    }
    else
    {
        it = std::find_if(begin, end, [&](AdjacencyEntryType const& adj) { return adj.v == v; });
        bFound = it != end;
    }
    if constexpr (not std::is_void_v<TData>)
        if (bFound and c != nullptr)
            *c = it->id;
    return bFound;
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
inline TVertexIndex DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::Degree(TVertexIndex u) const
{
    return mPrefix[static_cast<std::size_t>(u) + 1u] - mPrefix[u];
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::AdjacenciesOf(
    TVertexIndex u,
    FOnAdjacency&& fOnAdj)
{
    assert(u < NumVertices() and u >= 0 and "u out of range");
    TVertexIndex const begin = mPrefix[u];
    TVertexIndex const end   = mPrefix[static_cast<std::size_t>(u) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        AdjacencyEntryType const& adj = mAdjacencies[k];
        if constexpr (std::invocable<FOnAdjacency, TVertexIndex, TVertexIndex>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            TIdIndex c = mIdToData[adj.id];
            fOnAdj(adj.u, adj.v, mData[c]);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::AdjacenciesOf(
    TVertexIndex u,
    FOnAdjacency&& fOnAdj) const
{
    assert(u < NumVertices() and u >= 0 and "u out of range");
    TVertexIndex const begin = mPrefix[u];
    TVertexIndex const end   = mPrefix[static_cast<std::size_t>(u) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        AdjacencyEntryType const& adj = mAdjacencies[k];
        if constexpr (std::invocable<FOnAdjacency, TVertexIndex, TVertexIndex>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            TIdIndex c = mIdToData[adj.id];
            fOnAdj(adj.u, adj.v, mData[c]);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::ForAll(FOnAdjacency&& fOnAdj)
{
    for (AdjacencyEntryType& adj : mAdjacencies)
    {
        if constexpr (std::invocable<FOnAdjacency, TVertexIndex, TVertexIndex>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            TIdIndex c = mIdToData[adj.id];
            fOnAdj(adj.u, adj.v, mData[c]);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::ForAll(FOnAdjacency&& fOnAdj) const
{
    for (AdjacencyEntryType const& adj : mAdjacencies)
    {
        if constexpr (std::invocable<FOnAdjacency, TVertexIndex, TVertexIndex>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            TIdIndex c = mIdToData[adj.id];
            fOnAdj(adj.u, adj.v, mData[c]);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::CompactIds()
{
    // No-op when TData is void, since ids and data are unused.
    if constexpr (std::is_void_v<TData>)
    {
        return;
    }
    else
    {
        std::size_t const n = mAdjacencies.size();
        // Remap each triplet's id to its current data index
        for (auto& [tu, tv, tid] : mAdjacencies)
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
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
TIdIndex DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::AllocateId()
{
    TIdIndex id = static_cast<TIdIndex>(mIdToData.size());
    TIdIndex c  = static_cast<TIdIndex>(mData.size());
    mIdToData.push_back(c);
    mDataToId.push_back(id);
    mData.emplace_back(); // default-constructed; caller will overwrite
    return id;
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::ReleaseId(TIdIndex id)
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
template <class TOtherData>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::AddAdjacencyDataFrom(
    DenseAdjacencySet<TOtherData, VertexIndexType, IdIndexType>& other)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.AddAdjacencyDataFrom");
    // No-op when either this has void data, since ids and data are unused.
    if constexpr (std::is_void_v<TData>)
    {
        return;
    }
    else
    {
        for (AdjacencyEntryType& adj : mAdjacenciesToAdd)
        {
            auto aid = AllocateId();
            // Only assign if other's data is non-void and can be assigned into this's data.
            if constexpr (not std::is_void_v<TOtherData>)
            {
                if constexpr (std::is_assignable_v<TData&, TOtherData&&>)
                {
                    TIdIndex otherC = other.mIdToData[adj.id];
                    TIdIndex c      = mIdToData[aid];
                    mData[c]        = std::move(other.mData[otherC]);
                }
            }
            adj.id = aid;
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdded>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::AddNewAdjacencies(FOnAdded&& fOnAdded)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.AddNewAdjacencies");
    for (AdjacencyEntryType& adj : mAdjacenciesToAdd)
    {
        if constexpr (std::is_void_v<TData>)
        {
            fOnAdded(adj.u, adj.v);
        }
        else
        {
            adj.id     = AllocateId();
            TIdIndex c = mIdToData[adj.id];
            mData[c]   = fOnAdded(adj.u, adj.v);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnRemoved>
void DenseAdjacencySet<TData, TVertexIndex, TIdIndex>::RemoveOldAdjacencies(FOnRemoved&& fOnRemoved)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseAdjacencySet.RemoveOldAdjacencies");
    for (AdjacencyEntryType const& adj : mAdjacenciesToRemove)
    {
        if constexpr (std::is_void_v<TData>)
        {
            fOnRemoved(adj.u, adj.v);
        }
        else
        {
            TIdIndex c = mIdToData[adj.id];
            fOnRemoved(adj.u, adj.v, mData[c]);
            ReleaseId(adj.id);
        }
    }
}

// =============================================================================
// DenseReverseAdjacencySetView implementation
// ============================================================================

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::Update(
    SourceSetType const& set,
    std::make_signed_t<TVertexIndex> n)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DenseReverseAdjacencySetView.Update");

    // 1. Determine the number of target vertices
    // We need to scan all adjacencies to find the maximum v if not provided
    if (n < 0)
    {
        TVertexIndex maxV{0};
        set.ForAll([&maxV](TVertexIndex /*u*/, TVertexIndex v) { maxV = std::max(maxV, v); });
        n = static_cast<decltype(n)>(maxV) + 1;
    }

    // 2. Resize arrays
    std::size_t const nAdj = set.Size();
    mAdjacencies.resize(nAdj);
    auto const nV = static_cast<std::size_t>(n);
    mPrefix.resize(nV + 1u);
    mCounts.resize(nV);

    // 3. Count occurrences of each v (first pass)
    std::fill(mCounts.begin(), mCounts.end(), TVertexIndex{0});
    set.ForAll([this](TVertexIndex /*u*/, TVertexIndex v) { ++mCounts[v]; });

    // 4. Build prefix sum over v
    std::exclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin(), TVertexIndex{0});

    // 5. Reset counts to use as write cursors
    std::fill(mCounts.begin(), mCounts.end(), TVertexIndex{0});

    // 6. Scatter adjacencies into their correct positions (second pass)
    // Since the input is sorted by (u, v), within each bucket (same v), the elements
    // will be sorted by u automatically as we iterate in order.
    if constexpr (std::is_void_v<TData>)
    {
        set.ForAll([this](TVertexIndex u, TVertexIndex v) {
            std::size_t const pos = mPrefix[v] + mCounts[v];
            mAdjacencies[pos]     = AdjacencyEntryType{u, v};
            ++mCounts[v];
        });
    }
    else
    {
        // We need to get the data index c for each adjacency.
        // ForAll with 3-arg callback gives (u, v, data&), and we can compute c
        // from pointer arithmetic against set.Data()[0]. Alternatively, we could
        // make DenseReverseAdjacencySetView a friend of SourceSetType to access internal data
        // structures, but let's keep it simple.
        auto const* dataBegin = set.Data().data();
        set.ForAll([this, dataBegin](TVertexIndex u, TVertexIndex v, TData const& w) {
            std::size_t const pos = mPrefix[v] + mCounts[v];
            auto c                = static_cast<TIdIndex>(std::addressof(w) - dataBegin);
            mAdjacencies[pos]     = AdjacencyEntryType{u, v, c};
            ++mCounts[v];
        });
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
bool DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::Has(
    TVertexIndex u,
    TVertexIndex v,
    TIdIndex* c,
    bool bUseBinarySearch) const
{
    if (v >= NumVertices())
        return false;

    TVertexIndex const first = mPrefix[v];
    TVertexIndex const last  = mPrefix[static_cast<std::size_t>(v) + 1u];
    auto begin               = mAdjacencies.begin() + first;
    auto end                 = mAdjacencies.begin() + last;
    auto it                  = end;
    bool bFound{false};
    if (bUseBinarySearch)
    {
        // Within a bucket for target v, entries are sorted by u
        it = std::lower_bound(begin, end, u, [](AdjacencyEntryType const& adj, TVertexIndex u_) {
            return adj.u < u_;
        });
        bFound = it != end and (it->v == u); // adj.u is v (the target), adj.v is u (the source)
    }
    else
    {
        it = std::find_if(begin, end, [u](AdjacencyEntryType const& adj) { return adj.v == u; });
        bFound = it != end;
    }

    if constexpr (not std::is_void_v<TData>)
        if (bFound and c != nullptr)
            *c = it->id;
    return bFound;
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
TVertexIndex
DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::Degree(TVertexIndex v) const
{
    if (v >= NumVertices())
        return TVertexIndex{0};
    return mPrefix[static_cast<std::size_t>(v) + 1u] - mPrefix[v];
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::AdjacenciesOf(
    TVertexIndex v,
    FOnAdjacency&& fOnAdj) const
{
    assert(v < NumVertices() and "v out of range");
    TVertexIndex const begin = mPrefix[v];
    TVertexIndex const end   = mPrefix[static_cast<std::size_t>(v) + 1u];
    for (auto k = begin; k < end; ++k)
    {
        AdjacencyEntryType const& adj = mAdjacencies[k];
        if constexpr (std::is_void_v<TData>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            fOnAdj(adj.u, adj.v, adj.id);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
template <class FOnAdjacency>
void DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::ForAll(
    FOnAdjacency&& fOnAdj) const
{
    for (AdjacencyEntryType const& adj : mAdjacencies)
    {
        if constexpr (std::is_void_v<TData>)
        {
            fOnAdj(adj.u, adj.v);
        }
        else
        {
            fOnAdj(adj.u, adj.v, adj.id);
        }
    }
}

template <class TData, common::CIndex TVertexIndex, common::CIndex TIdIndex>
void DenseReverseAdjacencySetView<TData, TVertexIndex, TIdIndex>::Clear()
{
    mAdjacencies.clear();
    mPrefix.clear();
    mCounts.clear();
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ADJACENCYSET_H
