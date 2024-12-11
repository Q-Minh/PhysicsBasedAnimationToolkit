#include "Color.h"

#include "Mesh.h"
#include "pbat/common/Eigen.h"

#include <array>
#include <doctest/doctest.h>

TEST_CASE("[graph] Color")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    MatrixX V(3, 8);
    IndexMatrixX E(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 0., 1., 1., 0., 0., 1., 1.,
            0., 0., 0., 0., 1., 1., 1., 1.;
    E << 0, 3, 5, 6, 0,
            1, 2, 4, 7, 5,
            3, 0, 6, 5, 3,
            5, 6, 0, 3, 6;
    // clang-format on
    auto G   = graph::MeshPrimalGraph(E, V.cols());
    auto ptr = Eigen::Map<IndexVectorX>(G.outerIndexPtr(), G.outerSize() + 1);
    auto adj = Eigen::Map<IndexVectorX>(G.innerIndexPtr(), G.nonZeros());
    std::array<graph::EGreedyColorOrderingStrategy, 3> eOrderingStrategies{
        graph::EGreedyColorOrderingStrategy::Natural,
        graph::EGreedyColorOrderingStrategy::SmallestDegree,
        graph::EGreedyColorOrderingStrategy::LargestDegree};

    std::array<graph::EGreedyColorSelectionStrategy, 2> eSelectionStrategies{
        graph::EGreedyColorSelectionStrategy::LeastUsed,
        graph::EGreedyColorSelectionStrategy::FirstAvailable};
    // Act
    for (auto eOrdering : eOrderingStrategies)
    {
        for (auto eSelection : eSelectionStrategies)
        {
            IndexVectorX C = graph::GreedyColor(ptr, adj, eOrdering, eSelection);
            // Assert
            CHECK_EQ(C.size(), V.cols());
            for (auto v = 0; v < V.cols(); ++v)
                for (auto k = ptr(v); k < ptr(v + Index(1)); ++k)
                    if (v != adj(k))
                        CHECK_NE(C(v), C(adj(k)));
        }
    }
}