#include "HashGrid.h"

#include "pbat/geometry/AxisAlignedBoundingBox.h"
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] HashGrid")
{
    using namespace pbat::geometry;
    // Arrange
    using ScalarType = float;
    using IndexType  = int;
    using HashGridType = HashGrid<ScalarType, IndexType>;
    auto constexpr kDims = HashGridType::kDims;

    auto fHash = [](Eigen::Vector<IndexType, 3> const& ixyz) {
        return ixyz.sum(); // Simple hash function for demonstration
    };

    auto const [V, C] = model::Cube(model::EMesh::Tetrahedral /*mesh type*/, 1 /*layer*/);
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> L(3, C.cols());
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> U(3, C.cols());
    MeshToAabbs<kDims, 4>(V.cast<ScalarType>(), C.cast<IndexType>(), L, U);
    ScalarType const cellSize = ScalarType(0.5) * (U - L).maxCoeff();
    IndexType const nBuckets  = static_cast<IndexType>(C.cols() * 3);

    // Act
    HashGrid<ScalarType, IndexType> grid{};
    grid.Configure(cellSize, nBuckets);
    grid.Construct(L, U, fHash);

    // Assert
    CHECK_EQ(grid.NumberOfBuckets(), nBuckets);
    using Aabb = Eigen::AlignedBox<ScalarType, kDims>;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> allPairs(C.cols(), C.cols());
    allPairs.setConstant(false);
    grid.BroadPhase(
        ScalarType(0.5) * (L + U),
        [&](IndexType q, IndexType p) {
            allPairs(q, p) = true;
            Eigen::Matrix<ScalarType, kDims, 2> const cell =
                grid.Cell(ScalarType(0.5) * (L.col(q) + U.col(q)));
            bool const bCellOverlapsPrimitive =
                Aabb(cell.col(0), cell.col(1)).intersects(Aabb(L.col(p), U.col(p)));
            CHECK(bCellOverlapsPrimitive);
        },
        fHash);
    for (IndexType i = 0; i < C.cols(); ++i)
    {
        for (IndexType j = 0; j < C.cols(); ++j)
        {
            bool const bCellOverlapsPrimitive =
                Aabb(L.col(i), U.col(i)).intersects(Aabb(L.col(j), U.col(j)));
            CHECK_EQ(bCellOverlapsPrimitive, allPairs(i, j));
        }
    }
}