#include "Ogc.h"

namespace pbat::sim::contact::ogc {
} // namespace pbat::sim::contact::ogc

#include <doctest/doctest.h>
#include <ranges>

namespace pbat::sim::contact::ogc::detail::test {

/**
 * @brief Computes the triangle face (vertex, edge or triangle) nearest to the point xv.
 * @param uvw Barycentric coordinates of the closest point on triangle to xv.
 * @return The pair (a, eFace), where a is either a local vertex index or edge index, and eFace
 * indicates the type of face, i.e. (0 | 1 | 2) -> (triangle | edge | vertex)
 */
template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w)
{
    int const nZeros     = (u == TScalar(0)) + (v == TScalar(0)) + (w == TScalar(0));
    bool const bIsVertex = (nZeros == 2);
    bool const bIsEdge   = (nZeros == 1);
    int eFace;
    if (bIsVertex)
        eFace = 2;
    else if (bIsEdge)
        eFace = 1;
    else // is triangle
        eFace = 0;
    int a;
    if (bIsVertex)
    {
        if (v == float(1))
            a = 1;
        else if (w == float(1))
            a = 2;
        else // uvw(0) == float(1) must be true
            a = 0;
    }
    else if (bIsEdge)
    {
        if (u == float(0))
            a = 1;
        else if (v == float(0))
            a = 2;
        else // uvw(2) == float(0) must be true
            a = 0;
    }
    else // is triangle
    {
        a = 0;
    }
    return {a, eFace};
}

} // namespace pbat::sim::contact::ogc::detail::test

#include "pbat/geometry/MeshBoundary.h"
#include "pbat/graph/Mesh.h"
#include "pbat/sim/contact/MultiMesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact][ogc] ClosestFaceFacetToVertex")
{
    using pbat::math::linalg::mini::SVector;
    using namespace pbat::sim::contact::ogc;
    auto const fCheck = [&](double u, double v, double w, int aExpected, int eFaceExpected) {
        auto got = ClosestFaceFacetToVertex(u, v, w);
        auto exp = detail::test::ClosestFaceFacetToVertex(u, v, w);
        CHECK_EQ(got.first, aExpected);
        CHECK_EQ(got.second, eFaceExpected);
        CHECK_EQ(got.first, exp.first);
        CHECK_EQ(got.second, exp.second);
    };
    // Vertices
    fCheck(1.0, 0.0, 0.0, 0, 2); // vertex 0 -> a=0, eFace=2
    fCheck(0.0, 1.0, 0.0, 1, 2); // vertex 1 -> a=1, eFace=2
    fCheck(0.0, 0.0, 1.0, 2, 2); // vertex 2 -> a=2, eFace=2

    // Edge interiors (midpoints)
    fCheck(0.0, 0.5, 0.5, 1, 1); // edge opposite vertex 0 -> a=1, eFace=1
    fCheck(0.5, 0.0, 0.5, 2, 1); // edge opposite vertex 1 -> a=2, eFace=1
    fCheck(0.5, 0.5, 0.0, 0, 1); // edge opposite vertex 2 -> a=0, eFace=1

    // Triangle interior (no zeros)
    fCheck(0.2, 0.3, 0.5, 0, 0);    // face -> a=0, eFace=0
    fCheck(0.1, 0.1, 0.8, 0, 0);    // face -> a=0, eFace=0
    fCheck(0.34, 0.33, 0.33, 0, 0); // face -> a=0, eFace=0
}

TEST_CASE("[sim][contact][ogc] IsVertexFeasible")
{
    using namespace pbat;

    // Arrange: single tetrahedral cube and its boundary triangulation and half-edge adjacency
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto [Vb, Fb] = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto [GVHEp, GVHEadj] =
        geometry::VertexHalfEdgeAdjacency(Fb.bottomRows<3>(), static_cast<Index>(X.cols()));
    Scalar const eps            = Scalar(1e-3);
    Index const i               = 7;
    Eigen::Vector<Scalar, 3> xv = X.col(i);

    SUBCASE("feasible: move along +(1,1,1)")
    {
        Eigen::Vector<Scalar, 3> x = xv + eps * Eigen::Vector<Scalar, 3>::Ones();
        CHECK(sim::contact::ogc::IsVertexFeasible(X, Fb.bottomRows<3>(), GVHEp, GVHEadj, x, i));
    }
    SUBCASE("infeasible: move along +(1,-1,-1) axis")
    {
        Eigen::Vector<Scalar, 3> x =
            xv + eps * Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(-1), Scalar(-1)};
        CHECK_FALSE(
            sim::contact::ogc::IsVertexFeasible(X, Fb.bottomRows<3>(), GVHEp, GVHEadj, x, i));
    }
}

TEST_CASE("[sim][contact][ogc] IsEdgeFeasible")
{
    using namespace pbat;

    // Arrange: single tetrahedral cube and its boundary triangulation and half-edge adjacency
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto [Vb, Fb]                = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto GHEF                    = geometry::HalfEdgeFaceAdjacency(Fb.bottomRows<3>());
    auto EHE                     = geometry::EdgeHalfEdgeAdjacency(Fb.bottomRows<3>(), GHEF);
    auto const fGetHalfEdgeIndex = [&](Eigen::Vector<Scalar, 3> xi, Eigen::Vector<Scalar, 3> xj) {
        for (auto f = 0; f < Fb.cols(); ++f)
        {
            Index hei = geometry::FirstHalfEdgeOfFace(f);
            Index hej = geometry::NextHalfEdge(hei);
            Index hek = geometry::NextHalfEdge(hej);
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hei)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hei)).isApprox(xj))
                return hei;
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hej)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hej)).isApprox(xj))
                return hej;
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hek)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hek)).isApprox(xj))
                return hek;
        }
        return Index(-1);
    };
    auto const fGetVertexIndex = [&](Eigen::Vector<Scalar, 3> xv) {
        for (Index v = 0; v < X.cols(); ++v)
            if (X.col(v).isApprox(xv))
                return v;
        return Index(-1);
    };
    Scalar const eps = Scalar(1e-3);
    SUBCASE("Half-edge (0,0,0) -> (1,0,0)")
    {
        Index const he =
            fGetHalfEdgeIndex(Eigen::Vector<Scalar, 3>{0, 0, 0}, Eigen::Vector<Scalar, 3>{1, 0, 0});
        Index const i = fGetVertexIndex(Eigen::Vector<Scalar, 3>{0, 0, 0});
        SUBCASE("feasible: move along +(1,-1,-1) axis")
        {
            Eigen::Vector<Scalar, 3> x =
                X.col(i) + eps * Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(-1), Scalar(-1)};
            CHECK(sim::contact::ogc::IsEdgeFeasible(X, Fb.bottomRows<3>(), GHEF, x, he, i));
        }
        SUBCASE("infeasible: move along +(-1,0,0) axis")
        {
            Eigen::Vector<Scalar, 3> x =
                X.col(i) + eps * Eigen::Vector<Scalar, 3>{Scalar(-1), Scalar(0), Scalar(0)};
            CHECK_FALSE(sim::contact::ogc::IsEdgeFeasible(X, Fb.bottomRows<3>(), GHEF, x, he, i));
        }
    }
}

TEST_CASE("[sim][contact][ogc] Ogc")
{
    using namespace pbat;
    using namespace pbat::sim::contact::ogc;

    // Arrange
    auto ogcParams = Params<Scalar>()
                         .WithRadii(Scalar(0.01), Scalar(0.02))
                         .WithMaxContactEstimates(16, 16, 16)
                         .Construct();

    // single tetrahedral cube
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    MatrixX X1 = X;
    // Stack second cube on top in z direction, then shift in (1,1,1) direction, but stay inside
    // contact radius.
    MatrixX X2 = X.colwise() + (X.row(2).maxCoeff() - X.row(2).minCoeff()) * Vector<3>::UnitZ();
    X2.colwise() += Scalar(0.9) * ogcParams.r * Vector<3>::Ones().normalized();
    X.resize(3, X1.cols() + X2.cols());
    X << X1, X2;
    IndexMatrixX T1 = T;
    IndexMatrixX T2 = T.array() + static_cast<Index>(X1.cols());
    T.resize(4, T1.cols() + T2.cols());
    T << T1, T2;

    // Sort by connected components and reindex mesh/labels
    IndexVectorX XCC(X.cols()), TCC(T.cols()), Xord(X.cols()), Tord(T.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, T, XCC, TCC, Xord, Tord);
    graph::ReindexMeshByConnectedComponents(X, T, XCC, TCC, Xord, Tord);

    IndexVectorX V;
    IndexMatrixX F;
    IndexVectorX VP(nComponents + 1);
    IndexVectorX FP(nComponents + 1);
    IndexVectorX GXV;
    sim::contact::BoundaryTriangulation(T.bottomRows<4>(), XCC, V, F, VP, FP, GXV);
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

    geometry::DeviceConfig config;
    config.threads     = 1;
    config.userThreads = 1;
    config.verbose     = 0 /* 3 for debugging */;
    geometry::Device device(config);
    auto ogcInput = Input<Scalar, Index>{}
                        .WithDynamicGeometry(X, V, F, E, VP, FP, EP, GVHEp, GVHEadj, GHEF, EHE)
                        .Construct();
    State<Scalar, Index> ogcState{};
    ogcState.Initialize(device, ogcInput, ogcParams);
    ogcState.PrepareForExecution(ogcInput, ogcParams);

    SUBCASE("Vertex-Facet Contact Detection")
    {
        // Act: prepare iteration and perform vertex-facet contact detection
        VertexFacetContactDetection(ogcInput, ogcParams, ogcState);
        ogcState.CollectContactPairs();

        // Assert: expect contacts between top vertices of bottom cube and bottom faces of top cube
        auto const nVertexVertexContacts   = ogcState.mXX.size();
        auto const nVertexEdgeContacts     = ogcState.mXE.size();
        auto const nVertexTriangleContacts = ogcState.mXF.size();
        CHECK_EQ(nVertexTriangleContacts, 1);
        CHECK_EQ(nVertexEdgeContacts, 5);
        CHECK_EQ(nVertexVertexContacts, 2);
    }
    SUBCASE("Edge-Edge Contact Detection")
    {
        // Act: prepare iteration and perform edge-edge contact detection
        EdgeEdgeContactDetection(ogcInput, ogcParams, ogcState);
        ogcState.CollectContactPairs();
        // Assert: expect some edge-edge contacts
        auto const nEdgeEdgeContacts = ogcState.mEE.size();
        CHECK_GT(nEdgeEdgeContacts, 0);
    }
    SUBCASE("All contact detection")
    {
        // Act
        VertexFacetContactDetection(ogcInput, ogcParams, ogcState);
        EdgeEdgeContactDetection(ogcInput, ogcParams, ogcState);
        UpdateDisplacementBounds(ogcInput, ogcParams, ogcState);
        // Assert: displacement bounds are less than rq
        // NOTE: This is a weak test, but at least ensures that some plausible computation was done.
        Scalar const minDisplacementBound = ogcState.bv.minCoeff();
        CHECK_LT(minDisplacementBound, ogcParams.rq);
    }
}

TEST_CASE("[sim][contact][ogc] ClosestFaceEdgeToEdge")
{
    using namespace pbat;

    // Setup: a single tetrahedral cube; use its boundary triangle mesh to build edges
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto const [V, F] = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto const GHEF   = geometry::HalfEdgeFaceAdjacency(F.bottomRows<3>());
    auto const EHE    = geometry::EdgeHalfEdgeAdjacency(F.bottomRows<3>(), GHEF);
    auto const E = geometry::Edges(F.bottomRows<3>(), EHE); // 2 x |#edges| undirected edge list

    // Pick two concrete edges (by index) and build their endpoint arrays
    REQUIRE(E.cols() >= 2);
    Index const e1 = 0;
    Index const e2 = 1;
    std::array<Index, 2> const e1v{E(0, e1), E(1, e1)};
    std::array<Index, 2> const e2v{E(0, e2), E(1, e2)};

    auto const fCheck =
        [&](Scalar s, Scalar t, int a1Exp, int eFace1Exp, int a2Exp, int eFace2Exp) {
            auto const [a1, eFace1, a2, eFace2] =
                sim::contact::ogc::ClosestFaceEdgeToEdge(s, t, e1, e2, e1v, e2v);
            CHECK_EQ(a1, a1Exp);
            CHECK_EQ(eFace1, eFace1Exp);
            CHECK_EQ(a2, a2Exp);
            CHECK_EQ(eFace2, eFace2Exp);
        };

    // Both interior points on edges -> edge-edge contact
    fCheck(Scalar(0.3), Scalar(0.7), static_cast<int>(e1), 0, static_cast<int>(e2), 0);

    // One vertex, one edge
    fCheck(Scalar(0.0), Scalar(0.5), static_cast<int>(e1v[0]), 1, static_cast<int>(e2), 0);
    fCheck(Scalar(1.0), Scalar(0.5), static_cast<int>(e1v[1]), 1, static_cast<int>(e2), 0);
    fCheck(Scalar(0.5), Scalar(0.0), static_cast<int>(e1), 0, static_cast<int>(e2v[0]), 1);
    fCheck(Scalar(0.5), Scalar(1.0), static_cast<int>(e1), 0, static_cast<int>(e2v[1]), 1);

    // Both vertices -> vertex-vertex contact
    fCheck(Scalar(0.0), Scalar(0.0), static_cast<int>(e1v[0]), 1, static_cast<int>(e2v[0]), 1);
    fCheck(Scalar(1.0), Scalar(1.0), static_cast<int>(e1v[1]), 1, static_cast<int>(e2v[1]), 1);
}