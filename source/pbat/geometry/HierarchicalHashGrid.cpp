#include "HierarchicalHashGrid.h"

namespace pbat::geometry {
} // namespace pbat::geometry

#include "pbat/geometry/AxisAlignedBoundingBox.h"
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] HierarchicalHashGrid")
{
    using namespace pbat::geometry;
    // Arrange
    using ScalarType               = float;
    using IndexType                = int;
    auto constexpr kDims           = 3;
    using HierarchicalHashGridType = HierarchicalHashGrid<kDims, ScalarType, IndexType>;

    Eigen::Vector<IndexType, kDims> const gridDims{5, 5, 5};
    ScalarType cellSize = ScalarType(1);
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> L(kDims, gridDims.prod());
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> U(kDims, gridDims.prod());

    // Create regular grid of axis-aligned bounding boxes
    for (auto i = 0, n = 0; i < gridDims(0); ++i)
    {
        for (auto j = 0; j < gridDims(1); ++j)
        {
            for (auto k = 0; k < gridDims(2); ++k)
            {
                L.col(n) = Eigen::Vector<ScalarType, kDims>(
                    static_cast<ScalarType>(i) * cellSize,
                    static_cast<ScalarType>(j) * cellSize,
                    static_cast<ScalarType>(k) * cellSize);
                U.col(n) = L.col(n) + Eigen::Vector<ScalarType, kDims>::Constant(cellSize);
                ++n;
            }
        }
    }

    // Act
    HierarchicalHashGridType grid{};
    grid.Configure(static_cast<IndexType>(L.cols()));
    ScalarType const error = 0.01f;
    grid.Construct(L.array() - error, U.array() + error);

    // Assert
    CHECK_GE(grid.NumberOfBuckets(), L.cols());
    using Aabb = Eigen::AlignedBox<ScalarType, kDims>;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> bIsBroadPhasePair(L.cols(), L.cols());
    bIsBroadPhasePair.setConstant(false);
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> const Q = ScalarType(0.5) * (L + U);
    // Broad-phase pairs must be a super-set of overlapping pairs.
    grid.BroadPhase(Q, [&](IndexType q, IndexType p) { bIsBroadPhasePair(q, p) = true; });
    for (IndexType i = 0; i < L.cols(); ++i)
    {
        Aabb aabbi(L.col(i), U.col(i));
        for (IndexType j = 0; j < L.cols(); ++j)
        {
            Aabb aabbj(L.col(j), U.col(j));
            bool const bCellOverlapsPrimitive = aabbi.intersects(aabbj);
            if (bCellOverlapsPrimitive)
                CHECK(bIsBroadPhasePair(i, j));
        }
    }
}