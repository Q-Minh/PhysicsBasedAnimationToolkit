/**
 * @file HashGrid.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the definition of the HashGrid class, which is used for spatial
 * partitioning of 3D points in a grid structure.
 * @date 2025-06-21
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_HASHGRID_H
#define PBAT_GEOMETRY_HASHGRID_H

#include "pbat/common/Concepts.h"
#include "pbat/common/Modulo.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>

namespace pbat::geometry {

/**
 * @brief HashGrid is a spatial partitioning data structure that divides 3D space into a grid of
 * cells, allowing for efficient querying of points within a certain region.
 */
template <common::CFloatingPoint TScalar = Scalar, common::CIndex TIndex = Index>
class HashGrid
{
  public:
    using ScalarType           = TScalar; ///< Type alias for scalar values (e.g., float or double).
    using IndexType            = TIndex;  ///< Type alias for index values (e.g., int or size_t).
    static constexpr int kDims = 3;       ///< Number of spatial dimensions.

    /**
     * @brief Default constructor for HashGrid.
     */
    HashGrid() = default;
    /**
     * @brief Construct a HashGrid with a specific cell size and number of buckets.
     * @param cellSize Size of each grid cell along each dimension.
     * @param nBuckets Number of buckets in the hash table.
     */
    HashGrid(ScalarType cellSize, IndexType nBuckets);
    /**
     * @brief Configure the hash grid with a specific cell size and number of buckets.
     * @param cellSize Size of each grid cell along each dimension.
     * @param nBuckets Number of buckets in the hash table.
     */
    void Configure(ScalarType cellSize, IndexType nBuckets);
    /**
     * @brief Reserve space for a specific number of primitives in the hash grid.
     * @param nPrimitives Number of primitives to reserve space for.
     */
    void Reserve(IndexType nPrimitives);
    /**
     * @brief Construct a HashGrid from lower and upper bounds of input axis-aligned bounding boxes
     * (aabbs).
     * @tparam FHash Type of the hash function with signature `template <class TDerivedIX>
     * IndexType(Eigen::DenseBase<TDerivedIX> const& ixyz)`.
     * @tparam TDerivedL Type of the lower bounds matrix.
     * @tparam TDerivedU Type of the upper bounds matrix.
     * @param L `|# dims| x |# aabbs|` lower bounds of the aabbs.
     * @param U `|# dims| x |# aabbs|` upper bounds of the aabbs.
     * @param fHash Hash function for IndexType `# dims` point coordinates.
     */
    template <class FHash, class TDerivedL, class TDerivedU>
    void Construct(
        Eigen::DenseBase<TDerivedL> const& L,
        Eigen::DenseBase<TDerivedU> const& U,
        FHash fHash);
    /**
     * @brief Construct a HashGrid from points.
     * @tparam FHash Type of the hash function with signature `template <class TDerivedIX>
     * IndexType(Eigen::DenseBase<TDerivedIX> const& ixyz)`.
     * @tparam TDerivedX Type of the points matrix.
     * @param X `|# dims| x |# points|` points.
     * @param fHash Hash function for IndexType `# dims` point coordinates.
     */
    template <class FHash, class TDerivedX>
    void Construct(Eigen::DenseBase<TDerivedX> const& X, FHash fHash);
    /**
     * @brief Find all primitives whose cell overlaps with points `X`.
     *
     * @tparam FOnOverlap Function with signature `void(Index n, Index o)`
     * @tparam FHash Function with signature `IndexType(Eigen::DenseBase<TDerivedIX> const& ixyz)`
     * @tparam TDerivedX Eigen type of query points.
     * @param X `|# dims| x |# query points|` matrix of query points.
     * @param fOnOverlap Function to process an overlap
     * @param fHash Hash function for IndexType `# dims` point coordinates.
     */
    template <class FOnOverlap, class FHash, class TDerivedX>
    void Overlaps(Eigen::DenseBase<TDerivedX> const& X, FOnOverlap fOnOverlap, FHash fHash) const;
    /**
     * @brief Get the number of buckets in the hash table.
     * @return The number of buckets in the hash table.
     */
    IndexType NumberOfBuckets() const { return static_cast<IndexType>(mPrefix.size()) - 1; }
    /**
     * @brief Get the hash grid's cell (i.e. aabb) corresponding to a point `X`.
     * @param X `|# dims| x 1` point in space.
     * @return `|# dims| x 2` matrix representing the cell's lower and upper bounds in columns.
     */
    template <class TDerivedX>
    auto Cell(Eigen::DenseBase<TDerivedX> const& X) const -> Eigen::Matrix<ScalarType, kDims, 2>;
    /**
     * @brief Convert a point `X` to integer coordinates in the grid.
     * @tparam TDerivedX Eigen type of the point.
     * @param X `|# dims| x 1` point in space.
     * @return `|# dims| x 1` vector of integer coordinates in the grid.
     */
    template <class TDerivedX>
    auto ToIntegerCoordinates(Eigen::DenseBase<TDerivedX> const& X) const
        -> Eigen::Vector<IndexType, kDims>;

  private:
    ScalarType mCellSize; ///< Size of each grid cell along each dimension.
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mBucketIds; ///< `|# primitives| x 1` bucket IDs for each primitive in the grid
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mPrefix; ///< `|# buckets + 1| x 1` prefix sum holding hash table entries (i.e. primitives).
    Eigen::Vector<IndexType, Eigen::Dynamic>
        mPrimitiveIds; ///< `|# buckets + 1| x 1` primitive IDs for each primitive in the grid
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline HashGrid<TScalar, TIndex>::HashGrid(ScalarType cellSize, IndexType nBuckets)
    : HashGrid<TScalar, TIndex>()
{
    Configure(cellSize, nBuckets);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void HashGrid<TScalar, TIndex>::Configure(ScalarType cellSize, IndexType nBuckets)
{
    mCellSize = cellSize;
    mPrefix.resize(nBuckets + 1);
    mPrimitiveIds.resize(nBuckets + 1);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void HashGrid<TScalar, TIndex>::Reserve(IndexType nPrimitives)
{
    mBucketIds.resize(nPrimitives);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FHash, class TDerivedL, class TDerivedU>
void HashGrid<TScalar, TIndex>::Construct(
    Eigen::DenseBase<TDerivedL> const& L,
    Eigen::DenseBase<TDerivedU> const& U,
    FHash fHash)
{
    assert(L.rows() == U.rows());
    assert(L.cols() == U.cols());
    Construct(ScalarType(0.5) * (L.derived() + U.derived()), fHash);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FHash, class TDerivedX>
inline void HashGrid<TScalar, TIndex>::Construct(Eigen::DenseBase<TDerivedX> const& X, FHash fHash)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.HashGrid.Construct");
    auto const nPrimitives = static_cast<IndexType>(X.cols());
    auto const nBuckets    = NumberOfBuckets();
    assert(nBuckets >= nPrimitives);
    assert(nPrimitives == static_cast<IndexType>(X.cols()));
    assert(X.rows() == kDims);
    Reserve(nPrimitives);
    // Map primitives to hash IDs
    for (IndexType i = 0; i < nPrimitives; ++i)
    {
        auto ixyz     = ToIntegerCoordinates(X.col(i));
        mBucketIds[i] = common::Modulo(fHash(ixyz), nBuckets);
    }
    // Compute id counts in the prefix sum's memory
    mPrefix.setZero();
    mPrefix(mBucketIds).array() += 1;
    // Compute the shifted prefix sum
    std::inclusive_scan(mPrefix.begin(), mPrefix.end(), mPrefix.begin());
    // Construct primitive IDs while unshifting prefix sum
    for (IndexType i = 0; i < nPrimitives; ++i)
    {
        auto bucketId                      = mBucketIds[i];
        mPrimitiveIds[--mPrefix[bucketId]] = i;
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnOverlap, class FHash, class TDerivedX>
inline void HashGrid<TScalar, TIndex>::Overlaps(
    Eigen::DenseBase<TDerivedX> const& X,
    FOnOverlap fOnOverlap,
    FHash fHash) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.HashGrid.Overlaps");
    assert(X.rows() == kDims);
    auto const nQueries = static_cast<IndexType>(X.cols());
    auto const nBuckets = NumberOfBuckets();
    for (IndexType i = 0; i < nQueries; ++i)
    {
        auto ixyz = ToIntegerCoordinates(X.col(i));
        for (auto cx = -1; cx <= 1; ++cx)
        {
            for (auto cy = -1; cy <= 1; ++cy)
            {
                for (auto cz = -1; cz <= 1; ++cz)
                {
                    Eigen::Vector<IndexType, kDims> dixyz{cx, cy, cz};
                    auto bucketId = common::Modulo(fHash(ixyz + dixyz), nBuckets);
                    // Iterate over all primitives in the bucket
                    auto begin = mPrefix[bucketId];
                    auto end   = mPrefix[bucketId + 1];
                    for (auto j = begin; j < end; ++j)
                        fOnOverlap(i, mPrimitiveIds[j]);
                }
            }
        }
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline auto HashGrid<TScalar, TIndex>::Cell(Eigen::DenseBase<TDerivedX> const& X) const
    -> Eigen::Matrix<ScalarType, kDims, 2>
{
    Eigen::Matrix<ScalarType, kDims, 2> cell{};
    auto const ixyz = ToIntegerCoordinates(X);
    cell.col(0)     = ixyz.template cast<ScalarType>() * mCellSize;
    cell.col(1)     = cell.col(0).array() + static_cast<ScalarType>(mCellSize);
    return cell;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline auto
HashGrid<TScalar, TIndex>::ToIntegerCoordinates(Eigen::DenseBase<TDerivedX> const& X) const
    -> Eigen::Vector<IndexType, kDims>
{
    return Eigen::Vector<ScalarType, kDims>(
               X.derived().array() / static_cast<ScalarType>(mCellSize))
        .array()
        .floor()
        .template cast<IndexType>();
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_HASHGRID_H