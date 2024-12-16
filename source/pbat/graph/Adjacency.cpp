#include "Adjacency.h"

#include "Color.h"
#include "Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[graph] Adjacency")
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
    auto G                   = graph::MeshPrimalGraph(E, V.cols());
    auto [ptr, adj, weights] = graph::MatrixToWeightedAdjacency(G);
    auto C                   = graph::GreedyColor(ptr, adj);
    // Act
    auto [Cptr, Cadj] = graph::MapToAdjacency(C);
    // Assert
    auto nColors         = Cptr.size() - 1;
    auto nColorsExpected = C.maxCoeff() + 1;
    CHECK_EQ(nColors, nColorsExpected);
    for (auto c = 0; c < nColors; ++c)
    {
        auto kBegin = Cptr(c);
        auto kEnd   = Cptr(c + Index(1));
        for (auto k = kBegin; k < kEnd; ++k)
        {
            auto v = Cadj(k);
            CHECK_EQ(C(v), c);
        }
    }
}