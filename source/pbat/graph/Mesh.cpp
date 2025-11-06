#include "Mesh.h"

#include "pbat/geometry/model/Cube.h"

#include <algorithm>
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

TEST_CASE("[graph] Connected component ordering and reindexing")
{
    using namespace pbat;
    using namespace pbat::geometry::model;
    using graph::ReindexMeshByConnectedComponents;
    using graph::SortedConnectedComponentOrdering;

    // Build three disjoint tetrahedral cubes stacked along +Z
    auto [X1, T1] = Cube();
    auto [X2, T2] = Cube();
    auto [X3, T3] = Cube();
    X2.row(2).array() += Scalar(2);
    X3.row(2).array() += Scalar(4);

    // Concatenate into a single mesh (adjust indices)
    Index n1 = static_cast<Index>(X1.cols());
    Index n2 = static_cast<Index>(X2.cols());
    Index n3 = static_cast<Index>(X3.cols());
    MatrixX X(3, n1 + n2 + n3);
    X << X1, X2, X3;
    IndexMatrixX E(4, T1.cols() + T2.cols() + T3.cols());
    E << T1, (T2.array() + n1), (T3.array() + n1 + n2);

    // Sort by connected components and reindex mesh/labels
    IndexVectorX XCC(X.cols()), ECC(E.cols()), Xord(X.cols()), Eord(E.cols());
    Eigen::Index const nComponents = SortedConnectedComponentOrdering(X, E, XCC, ECC, Xord, Eord);
    CHECK_EQ(nComponents, 3);
    bool const bIsNodeOrderingSorted =
        std::is_sorted(Xord.begin(), Xord.end(), [&](Index a, Index b) { return XCC[a] < XCC[b]; });
    CHECK(bIsNodeOrderingSorted);
    bool const bIsElementOrderingSorted =
        std::is_sorted(Eord.begin(), Eord.end(), [&](Index a, Index b) { return ECC[a] < ECC[b]; });
    CHECK(bIsElementOrderingSorted);
    ReindexMeshByConnectedComponents(X, E, XCC, ECC, Xord, Eord);
    bool const bAreNodesSorted = std::is_sorted(XCC.data(), XCC.data() + XCC.size());
    CHECK(bAreNodesSorted);
    bool const bAreElementsSorted = std::is_sorted(ECC.data(), ECC.data() + ECC.size());
    CHECK(bAreElementsSorted);
}