#include "Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[graph] Mesh")
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
    // Act
    auto G  = graph::MeshAdjacencyMatrix(E, V.cols());
    auto GP = graph::MeshPrimalGraph(E, V.cols());
    auto GD = graph::MeshDualGraph(E, V.cols(), graph::EMeshDualGraphOptions::All);
    // Assert
    CHECK_EQ(GP.rows(), V.cols());
    CHECK_EQ(GP.cols(), V.cols());
    CHECK_EQ(GD.rows(), E.cols());
    CHECK_EQ(GD.cols(), E.cols());
    CHECK_EQ(G.rows(), V.cols());
    CHECK_EQ(G.cols(), E.cols());
}