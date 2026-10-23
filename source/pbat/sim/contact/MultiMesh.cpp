#include "MultiMesh.h"

namespace pbat::sim::contact {
} // namespace pbat::sim::contact

#include "pbat/geometry/HalfEdges.h"
#include "pbat/geometry/model/Cube.h"
#include "pbat/graph/Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] BoundaryTriangulation groups by component and matches boundary")
{
    using namespace pbat;
    using namespace pbat::geometry::model;

    // 1. Arrange

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
    IndexVectorX XCC(X.cols()), ECC(E.cols()), Xord(X.cols()), Eord(E.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, E, XCC, ECC, Xord, Eord);
    graph::ReindexMeshByConnectedComponents(X, E, XCC, ECC, Xord, Eord);

    // 2. Act

    // Compute boundary triangulation via API under test
    IndexVectorX V, VP(nComponents + 1), FP(nComponents + 1), GXV;
    IndexMatrixX F;
    sim::contact::BoundaryTriangulation(E.bottomRows<4>(), XCC, V, F, VP, FP, GXV);

    // 3. Assert

    // Prefix sizes and totals
    CHECK_EQ(VP.size(), nComponents + 1);
    CHECK_EQ(FP.size(), nComponents + 1);
    CHECK_EQ(VP(0), 0);
    CHECK_EQ(FP(0), 0);
    CHECK_EQ(VP(nComponents), V.size());
    CHECK_EQ(FP(nComponents), F.cols());

    // Component-wise counts match prefixes
    for (Eigen::Index c = 0; c < nComponents; ++c)
    {
        Index vcount = (XCC(V).array() == c).count();
        Index fcount = (XCC(F.row(0)).array() == c).count();
        CHECK_EQ(VP(c + 1) - VP(c), vcount);
        CHECK_EQ(FP(c + 1) - FP(c), fcount);
    }
}

TEST_CASE("[sim][contact] BoundaryTriangulationEdges adjacency and grouping are consistent")
{
    using namespace pbat;
    using namespace pbat::geometry::model;

    // 1. Arrange

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
    IndexMatrixX T(4, T1.cols() + T2.cols() + T3.cols());
    T << T1, (T2.array() + n1).matrix(), (T3.array() + n1 + n2).matrix();

    // Sort by connected components and reindex mesh/labels
    IndexVectorX XCC(X.cols()), TCC(T.cols()), Xord(X.cols()), Tord(T.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, T, XCC, TCC, Xord, Tord);
    CHECK_EQ(nComponents, 3);
    graph::ReindexMeshByConnectedComponents(X, T, XCC, TCC, Xord, Tord);

    // Compute boundary faces first
    IndexVectorX V, VP(nComponents + 1), FP(nComponents + 1), GXV;
    IndexMatrixX F;
    sim::contact::BoundaryTriangulation(T.bottomRows<4>(), XCC, V, F, VP, FP, GXV);

    // 2. Act

    // Compute boundary edges and adjacencies via API under test
    IndexMatrixX E;
    IndexVectorX EP(nComponents + 1);
    IndexVectorX GVHEp, GVHEadj;
    IndexMatrixX GHEF, EHE;
    sim::contact::BoundaryTriangulationEdges(
        F.bottomRows<3>(),
        XCC,
        E,
        EP,
        GVHEp,
        GVHEadj,
        GHEF,
        EHE);

    // 3. Assert

    // Basic sizes
    Index const nPoints = static_cast<Index>(XCC.size());
    CHECK_EQ(GVHEp.size(), nPoints + 1);
    CHECK_EQ(GHEF.cols(), 3 * F.cols());
    CHECK_EQ(GVHEadj.size(), 3 * F.cols());
    CHECK_EQ(E.rows(), 2);
    CHECK_EQ(EHE.rows(), 2);
    CHECK_EQ(EHE.cols(), E.cols());

    // All boundary edges should lie within a single component
    for (Eigen::Index e = 0; e < E.cols(); ++e)
    {
        Index i = E(0, e);
        Index j = E(1, e);
        CHECK_EQ(XCC(i), XCC(j));
    }

    // Prefix sizes and totals
    CHECK_EQ(EP.size(), nComponents + 1);
    CHECK_EQ(EP(0), 0);
    CHECK_EQ(EP(nComponents), E.cols());

    // Component-wise counts match prefixes
    for (Eigen::Index c = 0; c < nComponents; ++c)
    {
        Index ecount = (XCC(E.row(0)).array() == c).count();
        CHECK_EQ(EP(c + 1) - EP(c), ecount);
    }
}

TEST_CASE("[sim][contact] ConstructFromTriangleMesh lists only the referenced nodes as vertices")
{
    using namespace pbat;

    // 1. Arrange

    // Two unit cubes with an interior node each (their center), stacked along +Z: 2 tetrahedra per
    // face quad, to the center
    MatrixX X1(3, 9);
    for (Index i = 0; i < 8; ++i)
        X1.col(i) << Scalar(i & 1), Scalar((i >> 1) & 1), Scalar((i >> 2) & 1);
    X1.col(8).setConstant(Scalar(0.5));
    Index const quads[6][4] =
        {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
    IndexMatrixX T1(4, 12);
    for (Index q = 0; q < 6; ++q)
        for (Index k = 1; k <= 2; ++k)
            T1.col(2 * q + k - 1) << quads[q][0], quads[q][k], quads[q][k + 1], Index(8);
    MatrixX X2 = X1;
    IndexMatrixX T2 = T1;
    X2.row(2).array() += Scalar(2);
    Index n1 = static_cast<Index>(X1.cols());
    MatrixX X(3, X1.cols() + X2.cols());
    X << X1, X2;
    IndexMatrixX T(4, T1.cols() + T2.cols());
    T << T1, (T2.array() + n1).matrix();
    IndexVectorX XCC(X.cols()), TCC(T.cols()), Xord(X.cols()), Tord(T.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, T, XCC, TCC, Xord, Tord);
    graph::ReindexMeshByConnectedComponents(X, T, XCC, TCC, Xord, Tord);
    sim::contact::MultiMesh<Index> tets(T.bottomRows<4>(), XCC, nComponents);
    REQUIRE_LT(tets.V.size(), X.cols()); // the cubes have interior nodes

    auto const equal = [](auto const& A, auto const& B) {
        return A.rows() == B.rows() && A.cols() == B.cols() && (A.array() == B.array()).all();
    };

    SUBCASE("The boundary of a tetrahedral mesh over all its nodes, as the tetrahedral construction")
    {
        // 2. Act
        sim::contact::MultiMesh<Index> tris;
        tris.ConstructFromTriangleMesh(tets.F, XCC, nComponents);
        // 3. Assert
        CHECK(equal(tris.V, tets.V));
        CHECK(equal(tris.F, tets.F));
        CHECK(equal(tris.E, tets.E));
        CHECK(equal(tris.VP, tets.VP));
        CHECK(equal(tris.FP, tets.FP));
        CHECK(equal(tris.EP, tets.EP));
        CHECK(equal(tris.GVHEp, tets.GVHEp));
        CHECK(equal(tris.GVHEadj, tets.GVHEadj));
        CHECK(equal(tris.GHEF, tets.GHEF));
        CHECK(equal(tris.EHE, tets.EHE));
        CHECK(equal(tris.GXV, tets.GXV));
        for (Eigen::Index i = 0; i < X.cols(); ++i)
            if ((tets.V.array() != static_cast<Index>(i)).all())
                CHECK_EQ(tris.GXV(i), Index(-1));
    }
    SUBCASE("A triangle mesh that references every node keeps them all as vertices")
    {
        // The boundary over its own vertices only
        Eigen::Matrix<Index, 3, Eigen::Dynamic> F(3, tets.F.cols());
        for (Eigen::Index f = 0; f < F.cols(); ++f)
            for (Eigen::Index k = 0; k < 3; ++k)
                F(k, f) = tets.GXV(tets.F(k, f));
        IndexVectorX VCC = XCC(tets.V);
        // 2. Act
        sim::contact::MultiMesh<Index> tris;
        tris.ConstructFromTriangleMesh(F, VCC, nComponents);
        // 3. Assert
        auto const nVertices = static_cast<Index>(tets.V.size());
        IndexVectorX const all = IndexVectorX::LinSpaced(nVertices, Index(0), nVertices - Index(1));
        CHECK(equal(tris.V, all));
        CHECK(equal(tris.GXV, all));
        CHECK(equal(tris.VP, tets.VP));
    }
}
