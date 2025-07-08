#ifndef PBAT_GEOMETRY_HIERARCHICALHASHGRID_H
#define PBAT_GEOMETRY_HIERARCHICALHASHGRID_H

#include "pbat/Aliases.h"
#include "pbat/common/BruteSet.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Modulo.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <limits>
#include <span>

namespace pbat::geometry {

/**
 * @brief Spatial partitioning data structure that divides 3D space into a set of sparse grids.
 * Allowing for efficient querying of point neighbours within a certain region.
 * Implements @cite eitz2007hierarchical
 * @tparam Dims Number of spatial dimensions (2 or 3).
 * @tparam TScalar Type of scalar values (e.g., float or double).
 * @tparam TIndex Type of index values (e.g., int or long).
 */
template <int Dims, common::CFloatingPoint TScalar = Scalar, common::CIndex TIndex = Index>
class HierarchicalHashGrid
{
  public:
    using ScalarType           = TScalar; ///< Type alias for scalar values (e.g., float or double).
    using IndexType            = TIndex;  ///< Type alias for index values (e.g., int or long).
    static constexpr int kDims = Dims;    ///< Number of spatial dimensions.
    static_assert(Dims >= 2 and Dims <= 3, "HierarchicalHashGrid only supports 2D and 3D");
    static constexpr int kMaxCellsPerPrimitive =
        Dims == 2 ? 4 : 8; ///< Maximum number of cells per primitive in the grid.

    /**
     * @brief Default constructor for HierarchicalHashGrid.
     */
    HierarchicalHashGrid() = default;
    /**
     * @brief Construct a HierarchicalHashGrid with a specific number of primitives.
     * @param nPrimitives Number of primitives in the hash grid.
     */
    HierarchicalHashGrid(IndexType nPrimitives);
    /**
     * @brief Configure the hash grid with a specific number of buckets.
     * @param nPrimitives Number of primitives in the hash grid.
     */
    void Configure(IndexType nPrimitives);
    /**
     * @brief Construct a HashGrid from lower and upper bounds of input axis-aligned bounding boxes
     * (aabbs).
     * @details Time complexity is \f$ O(n) \f$ where `n` is the number of buckets.
     * @note Does not handle zero aabb extents.
     * @tparam TDerivedL Type of the lower bounds matrix.
     * @tparam TDerivedU Type of the upper bounds matrix.
     * @param L `|# dims| x |# aabbs|` lower bounds of the aabbs.
     * @param U `|# dims| x |# aabbs|` upper bounds of the aabbs.
     */
    template <class TDerivedL, class TDerivedU>
    void Construct(Eigen::DenseBase<TDerivedL> const& L, Eigen::DenseBase<TDerivedU> const& U);
    /**
     * @brief Find all primitives whose cell overlaps with points `X`.
     *
     * @tparam FOnPair Function with signature `void(Index q, Index p)` where `q` is the index of a
     * query point and `p` is the index of a primitive that potentially overlaps with the query
     * point.
     * @tparam TDerivedX Eigen type of query points.
     * @param X `|# dims| x |# query points|` matrix of query points.
     * @param fOnPair Function to process a broad-phase pair
     */
    template <class FOnPair, class TDerivedX>
    void BroadPhase(Eigen::DenseBase<TDerivedX> const& X, FOnPair fOnPair) const;
    /**
     * @brief Get the number of buckets in the hash table.
     * @return The number of buckets in the hash table.
     */
    IndexType NumberOfBuckets() const { return static_cast<IndexType>(mPrefix.size()) - 1; }
    /**
     * @brief Get the number of levels in the hierarchical grid.
     * @return The number of levels in the hierarchical grid.
     */
    IndexType NumberOfLevels() const { return static_cast<IndexType>(mSetOfLevels.Size()); }
    /**
     * @brief Get the set of levels in the hierarchical grid.
     * @return `|# levels| x 1` vector of levels in the hierarchical grid.
     */
    auto Levels() const -> std::span<std::int16_t const>
    {
        return std::span<std::int16_t const>(mSetOfLevels.begin(), mSetOfLevels.Size());
    }
    /**
     * @brief Convert a point `X` to integer coordinates in the grid.
     * @tparam TDerivedX Eigen type of the point.
     * @param X `|# dims| x 1` point in space.
     * @param cellSize Size of each grid cell.
     * @return `|# dims| x 1` vector of integer coordinates in the grid.
     */
    template <class TDerivedX>
    auto ToIntegerCoordinates(Eigen::DenseBase<TDerivedX> const& X, ScalarType const cellSize) const
        -> Eigen::Vector<IndexType, kDims>;
    /**
     * @brief Hash a point `X` at level `l` in the grid.
     * See @cite eitz2007hierarchical for details on the hashing scheme.
     * @param X `|# dims| x 1` point in space.
     * @param l Level of the grid at which to hash the point.
     * @return Hash value for the point at level `l`.
     */
    template <class TDerivedX>
    auto Hash(Eigen::DenseBase<TDerivedX> const& X, std::int16_t l) const;

  protected:
    /**
     * @brief Clear the hash table, resetting all internal data structures.
     */
    void ClearHashTable();

  private:
    Eigen::Vector<std::uint8_t, Eigen::Dynamic>
        mCellCounts; ///< `|# primitives| x 1` number of overlapping cells per primitive.
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mBucketIds; ///< `|# (primitive,cell) pairs| x 1` bucket IDs for each
                    ///< (primitive,cell) pair in the grid
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mPrefix; ///< `|# buckets + 1| x 1` prefix sum holding hash table entries
                 ///< (i.e. primitives,cell pairs).
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mPrimitiveIds; ///< `|# primitive,cell) pairs| x 1` primitive IDs for each (primitive,cell)
                       ///< pair in the grid, in the order of the prefix sum.
    common::BruteSet<std::int16_t> mSetOfLevels; ///< Set of levels in the hierarchical grid.
};

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
HierarchicalHashGrid<Dims, TScalar, TIndex>::HierarchicalHashGrid(IndexType nPrimitives)
    : mCellCounts(), mBucketIds(), mPrefix(), mPrimitiveIds(), mSetOfLevels()
{
    Configure(nPrimitives);
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void HierarchicalHashGrid<Dims, TScalar, TIndex>::Configure(IndexType nPrimitives)
{
    auto nBuckets = nPrimitives * kMaxCellsPerPrimitive;
    mCellCounts.resize(nPrimitives);
    mBucketIds.resize(nBuckets);
    mPrefix.resize(nBuckets + 1);
    mPrimitiveIds.resize(nBuckets);
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedL, class TDerivedU>
void HierarchicalHashGrid<Dims, TScalar, TIndex>::Construct(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.HierarchicalHashGrid.Construct");
    assert(L.rows() == kDims and U.rows() == kDims); // L and U must have |# dims| rows
    assert(L.cols() == U.cols());                    // L and U must have the same number of columns
    auto const nPrimitives = static_cast<IndexType>(L.cols());
    auto const nBuckets    = NumberOfBuckets();
    assert(nPrimitives <= nBuckets);
    assert(mPrefix.size() == nBuckets + 1); // Prefix size must be nBuckets + 1
    // Build hash table from scratch
    ClearHashTable();
    // Map primitives to cells
    IndexType k = 0; // Index for mBucketIds and mPrimitiveIds
    for (IndexType i = 0; i < nPrimitives; ++i)
    {
        ScalarType const maxExtent = (U.col(i) - L.col(i)).maxCoeff();
        ScalarType const l         = std::ceil(std::log2(maxExtent));
        ScalarType const cellSize  = std::pow(ScalarType(2), l);
        Eigen::Vector<IndexType, kDims> const ib =
            ToIntegerCoordinates(L.col(i).segment<kDims>(0), cellSize);
        Eigen::Vector<IndexType, kDims> const ie =
            ToIntegerCoordinates(U.col(i).segment<kDims>(0), cellSize);
        auto const level = static_cast<std::int16_t>(l);
        Eigen::Vector<IndexType, kDims> ix;
        // Add primitive i to every cell at level l overlapping with i's bounding box
        for (ix(0) = ib(0); ix(0) <= ie(0); ++ix(0))
        {
            for (ix(1) = ib(1); ix(1) <= ie(1); ++ix(1))
            {
                if constexpr (kDims == 2)
                {
                    IndexType const hash = Hash(ix, level);
                    mBucketIds(k++)      = common::Modulo(hash, nBuckets);
                    ++mCellCounts(i);
                }
                else
                {
                    for (ix(2) = ib(2); ix(2) <= ie(2); ++ix(2))
                    {
                        IndexType const hash = Hash(ix, level);
                        mBucketIds(k++)      = common::Modulo(hash, nBuckets);
                        ++mCellCounts(i);
                    }
                }
            }
        }
        mSetOfLevels.Insert(level);
    }
    // Compute id counts in the prefix sum's memory
    mPrefix(mBucketIds.segment(0, k)).array() += 1;
    // Compute the shifted prefix sum
    std::inclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin());
    // Construct primitive IDs while unshifting prefix sum
    k = 0;
    for (IndexType i = 0; i < nPrimitives; ++i)
    {
        for (std::uint8_t j = 0; j < mCellCounts(i); ++j, ++k)
        {
            auto bucketId                      = mBucketIds(k);
            mPrimitiveIds(--mPrefix(bucketId)) = i;
        }
    }
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPair, class TDerivedX>
void HierarchicalHashGrid<Dims, TScalar, TIndex>::BroadPhase(
    Eigen::DenseBase<TDerivedX> const& X,
    FOnPair fOnPair) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.HierarchicalHashGrid.BroadPhase");
    assert(X.rows() == kDims); // X must have |# dims| rows
    auto const nQueries = static_cast<IndexType>(X.cols());
    auto const nBuckets = NumberOfBuckets();
    for (IndexType q = 0; q < nQueries; ++q)
    {
        for (std::int16_t l : mSetOfLevels)
        {
            auto cellSize = std::pow(ScalarType(2), static_cast<ScalarType>(l));
            Eigen::Vector<IndexType, kDims> iq = ToIntegerCoordinates(X.col(q), cellSize);
            auto bucketId    = (kDims == 2) ? common::Modulo(Hash(iq, l), nBuckets) :
                                              common::Modulo(Hash(iq, l), nBuckets);
            auto beginBucket = mPrefix(bucketId);
            auto endBucket   = mPrefix(bucketId + 1);
            for (auto k = beginBucket; k < endBucket; ++k)
            {
                auto const i = mPrimitiveIds(k);
                fOnPair(q, i);
            }
        }
    }
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline auto HierarchicalHashGrid<Dims, TScalar, TIndex>::ToIntegerCoordinates(
    Eigen::DenseBase<TDerivedX> const& X,
    ScalarType const cellSize) const -> Eigen::Vector<IndexType, kDims>
{
    return Eigen::Vector<ScalarType, kDims>(X.derived().array() / static_cast<ScalarType>(cellSize))
        .array()
        .floor()
        .template cast<IndexType>();
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline auto HierarchicalHashGrid<Dims, TScalar, TIndex>::Hash(
    Eigen::DenseBase<TDerivedX> const& X,
    std::int16_t l) const
{
    if constexpr (kDims == 2)
        return (X(0) * 73856093) ^ (X(1) * 19349663) ^ (l * 67867979);
    else
        return (X(0) * 73856093) ^ (X(1) * 19349663) ^ (X(2) * 834927911) ^ (l * 67867979);
}

template <int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
void HierarchicalHashGrid<Dims, TScalar, TIndex>::ClearHashTable()
{
    mCellCounts.setZero();
    mPrimitiveIds.setConstant(IndexType(-1));
    mPrefix.setZero();
    mSetOfLevels.Clear();
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_HIERARCHICALHASHGRID_H