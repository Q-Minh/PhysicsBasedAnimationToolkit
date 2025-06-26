#include "HashGrid.h"

#include "pbat/geometry/AxisAlignedBoundingBox.h"
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] HashGrid")
{
    using namespace pbat::geometry;
    // Arrange
    using ScalarType     = float;
    using IndexType      = int;
    auto constexpr kDims = 3;
    using HashGridType   = HashGrid<kDims, ScalarType, IndexType>;

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
                L.col(n) =
                    Eigen::Vector<ScalarType, kDims>(i * cellSize, j * cellSize, k * cellSize);
                U.col(n) = L.col(n) + Eigen::Vector<ScalarType, kDims>::Constant(cellSize);
                ++n;
            }
        }
    }

    IndexType const nBuckets = static_cast<IndexType>(L.cols() * 3);
    auto const fHash         = HashByXorOfPrimeMultiples<IndexType>();

    // Act
    HashGridType grid{};
    ScalarType tolerance = std::numeric_limits<ScalarType>::epsilon();
    grid.Configure(cellSize + tolerance, nBuckets);
    grid.Construct(L, U, fHash);

    // Assert
    CHECK_EQ(grid.NumberOfBuckets(), nBuckets);
    using Aabb = Eigen::AlignedBox<ScalarType, kDims>;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> allPairs(L.cols(), L.cols());
    allPairs.setConstant(false);
    Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> const Q = ScalarType(0.5) * (L + U);
    // Broad-phase pairs must be a super-set of overlapping pairs.
    grid.BroadPhase(Q, [&](IndexType q, IndexType p) { allPairs(q, p) = true; }, fHash);
    for (IndexType i = 0; i < L.cols(); ++i)
    {
        Aabb aabbi(L.col(i), U.col(i));
        for (IndexType j = 0; j < L.cols(); ++j)
        {
            Aabb aabbj(L.col(j), U.col(j));
            bool const bCellOverlapsPrimitive = aabbi.intersects(aabbj);
            if (bCellOverlapsPrimitive)
                CHECK(allPairs(i, j));
        }
    }
}