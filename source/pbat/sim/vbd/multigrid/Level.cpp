#include "Level.h"

#include "Kernels.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/profiling/Profiling.h"

#include <tbb/parallel_for.h>
#include <utility>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Level::Level(Data const& data, VolumeMesh meshIn)
    : mesh(std::move(meshIn)),
      u(),
      colors(),
      Pptr(),
      Padj(),
      ecVE(),
      NecVE(),
      ilocalE(),
      GEptr(),
      GEadj(),
      bActiveE(),
      wgE(),
      ecK(),
      NecK(),
      GKptr(),
      GKadj(),
      GKilocal(),
      bActiveK(),
      mK(),
      bIsDirichletVertex()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.Construct");

    // Coarse mesh
    u.resize(mesh.X.rows(), mesh.X.cols());
    auto G                  = graph::MeshPrimalGraph(mesh.E, mesh.X.cols());
    auto [Gptr, Gadj, Gwts] = graph::MatrixToWeightedAdjacency(G);
    colors                  = graph::GreedyColor(Gptr, Gadj, data.eOrdering, data.eSelection);
    std::tie(Pptr, Padj)    = graph::MapToAdjacency(colors);

    geometry::TetrahedralAabbHierarchy cbvh(mesh.X, mesh.E);

    // Elastic energy
    //
    // Objective:
    // Need to construct list of edges (vc, ef), where vc is coarse vertex and ef is fine element.
    // In addition, for each such edge, for each fine vertex vf of element ef, we want to know:
    // - (ec, Nec, ilocal) where Nec are ec's shape functions at vf and ilocal is vc's local vertex
    //   index in ec
    //
    // 1. We first construct 4x|#fine elements| indices ecVE which contain, in each column,
    //    the 4 coarse elements ec containing the 4 vertices of element ef.
    // 2. We then construct 4x|4*#fine elements| coarse element shape functions at those vertices.
    // 3. Then, we construct the graph (vc, ef) by traversing the graph with edge (ec, ef) if ec
    //    contains at least one fine vertex of ef.
    // 4. For each (vc,ef), we look at all 4 ec in the column ef of ecVE, and determine which local
    //    vertex index vc corresponds to, or set it to -1 if it doesn't apply.
    auto const nFineElements = data.mesh.E.cols();
    ecVE.resize(4, nFineElements);
    NecVE.resize(4, 4 * nFineElements);
    for (auto i = 0; i < 4; ++i)
    {
        auto ithVertexPositions = data.mesh.X(Eigen::placeholders::all, data.mesh.E.row(i));
        ecVE.row(i)             = cbvh.PrimitivesContainingPoints(ithVertexPositions);
        NecVE(Eigen::placeholders::all, Eigen::seqN(i, nFineElements, 4)) =
            fem::ShapeFunctionsAt(mesh, ecVE.row(i), ithVertexPositions);
    }
    Index constexpr nVertexPerElement = 4;
    std::vector<graph::WeightedEdge<Scalar, Index>> vc2ef{};
    vc2ef.reserve(static_cast<std::size_t>(nFineElements * nVertexPerElement * nVertexPerElement));
    for (auto ef = 0; ef < nFineElements; ++ef)
        for (auto ec : ecVE.col(ef))
            for (auto vc : mesh.E.col(ec))
                vc2ef.push_back({vc, ef});
    Index const nCoarseVertices = mesh.X.cols();
    std::tie(GEptr, GEadj)      = graph::MatrixToAdjacency(graph::AdjacencyMatrixFromEdges(
        vc2ef.begin(),
        vc2ef.end(),
        nCoarseVertices,
        nFineElements));
    ilocalE.setConstant(4, GEadj.size(), Index(-1));
    graph::ForEachEdge(GEptr, GEadj, [&](Index vc, Index ef, Index eid) {
        for (Index vf = 0; vf < 4; ++vf)
        {
            Index ec = ecVE(vf, ef);
            for (Index iclocal = 0; iclocal < 4; ++iclocal)
                if (vc == mesh.E(iclocal, ec))
                    ilocalE(vf, eid) = iclocal;
        }
    });

    // Kinetic energy
    ecK                 = cbvh.PrimitivesContainingPoints(data.mesh.X);
    NecK                = fem::ShapeFunctionsAt(mesh, ecK, data.mesh.X);
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, ecK.size());
    G                   = graph::MeshAdjacencyMatrix(
        mesh.E(Eigen::placeholders::all, ecK),
        ilocal,
        mesh.X.cols(),
        true);
    std::tie(GKptr, GKadj, GKilocal) = graph::MatrixToWeightedAdjacency(G);

    // Dirichlet energy
    auto const nFineVertices = data.mesh.X.cols();
    bIsDirichletVertex.setConstant(nFineVertices, false);
    bIsDirichletVertex(data.dbc).setConstant(true);

#ifdef PBAT_USE_METIS
    HyperReduce(data);
#else
    wgE = data.wg;
    mK  = data.m;
    bActiveE.setConstant(nFineElements, true);
    bActiveK.setConstant(nFineVertices, true);
#endif
}

void Level::HyperReduce(Data const& data)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.HyperReduce");

    // Hyper-reduction
    // 1. Partition root mesh elements into a set of clusters.
    // 2. In each cluster, pick a single element (i.e. the one closest to the cluster's geometric
    // center).
    // 3. Mark that element as active.
    // 4. Its quadrature weight should be the sum of its cluster's elements' quadrature weights.
    // 5. Mark each active elements' vertices as active.
    // 6. Scale each active vertex's mass such that the sum of active vertex masses = sum of vertex
    // masses.
    auto const nFineElements = data.mesh.E.cols();
    auto const nFineVertices = data.mesh.X.cols();
    bActiveE.setConstant(nFineElements, false);
    bActiveK.setConstant(nFineVertices, false);

    // Choose coarse level compute budget as 3x its elements, but not surpassing the number of fine
    // elements.
    // TODO: Implement smart way to choose computational budget at each coarse level.
    auto const nCoarseElements = mesh.E.cols();
    Index nSubsetElements      = std::min(3 * nCoarseElements, nFineElements);
    // Compute the graph of face-adjacent tetrahedra on the fine mesh
    auto Gdual = graph::MeshDualGraph(data.mesh.E, nFineVertices);
    Gdual.prune([]([[maybe_unused]] Index row, [[maybe_unused]] Index col, Index value) {
        bool const bIsFaceAdjacency = value == 3;
        return bIsFaceAdjacency;
    });
    auto Gdualptr = graph::AdjacencyMatrixPrefix(Gdual);
    auto Gdualadj = graph::AdjacencyMatrixIndices(Gdual);
    // Partition the fine mesh using scale-invariant geometric distance between element barycenters
    // as the weight function.
    // TODO: Use smarter weight function that exhibits shape and material awareness.
    MatrixX XEbary = data.mesh.QuadraturePoints<1>();
    Scalar const scale =
        (data.mesh.X.rowwise().maxCoeff() - data.mesh.X.rowwise().minCoeff()).norm();
    Scalar decimalPrecision = 1e4;
    IndexVectorX Wdual(Gdual.nonZeros());
    graph::ForEachEdge(Gdualptr, Gdualadj, [&](Index ei, Index ej, Index k) {
        Scalar dij = (XEbary.col(ei) - XEbary.col(ej)).norm() / scale;
        Wdual(k)   = static_cast<Index>(dij * decimalPrecision);
    });
    IndexVectorX clusters         = graph::Partition(Gdualptr, Gdualadj, Wdual, nSubsetElements);
    auto [clusterPtr, clusterAdj] = graph::MapToAdjacency(clusters);
    // Compute the coarse level quadrature weights of active fine elements as the sum of cluster
    // element weights.
    VectorX clusterQuadWeights = VectorX::Zero(nSubsetElements);
    graph::ForEachEdge(clusterPtr, clusterAdj, [&](Index c, Index e, [[maybe_unused]] Index k) {
        clusterQuadWeights(c) += data.wg(e);
    });
    // We use the previously computed weighting scheme to compute "shape-aware" cluster geometric
    // centers
    MatrixX clusterCenters = MatrixX::Zero(3, nSubsetElements);
    graph::ForEachEdge(clusterPtr, clusterAdj, [&](Index c, Index e, [[maybe_unused]] Index k) {
        clusterCenters.col(c) += data.wg(e) * XEbary.col(e) / clusterQuadWeights(c);
    });
    // Find active elements as those closest to cluster centers.
    // TODO:
    // We might want to make this adaptive per timestep. This could be
    // in the form of keeping the clusters pre-computed, but changing the
    // active element in each cluster at the beginning of each time step
    // based on strain rate.
    VectorX clusterMinDistToCenter =
        VectorX::Constant(nSubsetElements, std::numeric_limits<Scalar>::max());
    IndexVectorX activeElements(nSubsetElements);
    graph::ForEachEdge(clusterPtr, clusterAdj, [&](Index c, Index e, [[maybe_unused]] Index k) {
        Scalar d = (XEbary.col(e) - clusterCenters.col(c)).squaredNorm();
        if (d >= clusterMinDistToCenter(c))
            return;
        activeElements(c)         = e;
        clusterMinDistToCenter(c) = d;
    });
    // Make element vertices active
    IndexVectorX activeVertices = data.mesh.E(Eigen::placeholders::all, activeElements).reshaped();
    std::sort(activeVertices.begin(), activeVertices.end());
    auto const nActiveVertices = std::distance(
        activeVertices.begin(),
        std::unique(activeVertices.begin(), activeVertices.end()));
    activeVertices.conservativeResize(nActiveVertices);
    bActiveE(activeElements).setConstant(true);
    bActiveK(activeVertices).setConstant(true);
    // Compute coarse level quadrature weights
    // TODO:
    // Find smarter way to compute coarse quadrature weights.
    // This could potentially be an adaptive, per-timestep method
    // that computes a quadrature weight which, when multiplying
    // its corresponding active element's elasticity, would yield
    // the sum of its cluster's elements' elastic energies.
    wgE.setZero(nFineElements);
    wgE(activeElements) = clusterQuadWeights;
    // Compute coarse level lumped masses by scaling s.t. total mass is matched,
    // i.e. \alpha * \sum Mcoarse(i) = Mtotal -> \alpha = Mtotal / \sum Mcoarse(i)
    Scalar alpha = data.m.sum() / data.m(activeVertices).sum();
    mK.setZero(nFineVertices);
    mK(activeVertices) = alpha * data.m(activeVertices);

    // TODO:
    // Find smart way to reduce quadrature for Dirichlet boundary conditions and
    // contact constraints. This could be in the form of selecting only a fixed-size
    // subset of contacts with the deepest penetrations, while the Dirichlet reduction
    // should be based on shape preservation and can be precomputed.
}

void Level::Prolong(Data& data) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.Prolong");
    tbb::parallel_for(Index(0), data.x.cols(), [&](Index i) {
        Index ec = ecK(i);
        auto uec = u(Eigen::placeholders::all, mesh.E.col(ec));
        auto Nec = NecK.col(i);
        data.x.col(i) += uec * Nec;
    });
}

void Level::Smooth(Scalar dt, Index iters, Data& data)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.Smooth");
    u.setZero();
    Index const nPartitions = Pptr.size() - 1;
    Scalar dt2              = dt * dt;
    for (auto iter = 0; iter < iters; ++iter)
    {
        for (Index p = 0; p < nPartitions; ++p)
        {
            auto pBegin = Pptr(p);
            auto pEnd   = Pptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                Index i     = Padj(kp);
                auto gBegin = GEptr(i);
                auto gEnd   = GEptr(i + 1);
                // Use mini API
                using pbat::math::linalg::mini::ToEigen;
                using pbat::math::linalg::mini::FromEigen;
                using pbat::math::linalg::mini::SMatrix;
                using pbat::math::linalg::mini::SVector;
                using pbat::math::linalg::mini::Zeros;

                // Compute energy derivatives
                SMatrix<Scalar, 3, 3> Hu = Zeros<Scalar, 3, 3>();
                SVector<Scalar, 3> gu    = Zeros<Scalar, 3>();

                // Elastic energy
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index ef = GEadj(kg);
                    if (not bActiveE(ef))
                        continue;

                    IndexVector<4> ilocal = ilocalE.col(kg);
                    Scalar wg             = wgE(ef);
                    Scalar mug            = data.lame(0, ef);
                    Scalar lambdag        = data.lame(1, ef);
                    Matrix<4, 3> GNef     = data.GP.block<4, 3>(0, 3 * ef);
                    Matrix<4, 4> N        = NecVE.block<4, 4>(0, 4 * ef);
                    Matrix<3, 4> xe       = data.x(Eigen::placeholders::all, data.mesh.E.col(ef));
                    IndexMatrix<4, 4> ec  = mesh.E(Eigen::placeholders::all, ecVE.col(ef));
                    for (auto iflocal = 0; iflocal < 4; ++iflocal)
                    {
                        xe.col(iflocal) +=
                            u(Eigen::placeholders::all, ec.col(iflocal)) * N.col(iflocal);
                    }
                    using kernels::AccumulateElasticEnergy;
                    AccumulateElasticEnergy(
                        physics::StableNeoHookeanEnergy<3>{},
                        FromEigen(ilocal),
                        dt2 * wg,
                        mug,
                        lambdag,
                        FromEigen(xe),
                        FromEigen(GNef),
                        FromEigen(N),
                        gu,
                        Hu);
                }

                // Kinetic + Dirichlet energy
                gBegin = GKptr(i);
                gEnd   = GKptr(i + 1);
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    // Kinetic energy is 1/2 * mi * || (x^k + u*N) - xtilde ||_2^2
                    Index vf                = GKadj(kg);
                    bool const bIsActive    = bActiveK(vf);
                    bool const bIsDirichlet = bIsDirichletVertex(vf);
                    if (not bIsActive and not bIsDirichlet)
                        continue;
                    Index ilocal          = GKilocal(kg);
                    Index ec              = ecK(vf);
                    SVector<Scalar, 4> Ne = FromEigen(NecK.col(vf).head<4>());
                    SVector<Scalar, 3> xk = FromEigen(data.x.col(vf).head<3>());
                    SMatrix<Scalar, 3, 4> ue =
                        FromEigen(u(Eigen::placeholders::all, mesh.E.col(ec)).block<3, 4>(0, 0));
                    SVector<Scalar, 3> x = xk + ue * Ne;
                    if (bIsActive)
                    {
                        Scalar mvf                = mK(vf);
                        SVector<Scalar, 3> xtilde = FromEigen(data.xtilde.col(vf).head<3>());
                        gu += Ne(ilocal) * mvf * (x - xtilde);
                        Diag(Hu) += Ne(ilocal) * Ne(ilocal) * mvf;
                    }
                    // Dirichlet energy
                    if (bIsDirichlet)
                    {
                        // NOTE: We should have an explicit matrix of Dirichlet boundary conditions
                        // so that we can control them, rather than always having them as rest
                        // positions.
                        SVector<Scalar, 3> xD = FromEigen(data.mesh.X.col(vf).head<3>());
                        // Dirichlet energy is 1/2 muD * || (x^k + u*N) - d(x) ||_2^2
                        gu += Ne(ilocal) * data.muD * (x - xD);
                        Diag(Hu) += Ne(ilocal) * Ne(ilocal) * data.muD;
                    }
                }
                // Integrate
                if (std::abs(Determinant(Hu)) < data.detHZero)
                    return;
                SVector<Scalar, 3> du = -(Inverse(Hu) * gu);
                u.col(i) += ToEigen(du);
            });
        }
    }
    Prolong(data);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Level")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::VolumeMesh;
    using sim::vbd::multigrid::Level;

    // Arrange
    auto const [VR, CR] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 2);
    Data data           = Data().WithVolumeMesh(VR, CR).Construct();
    auto const [VL, CL] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 0);

    // Act
    Level level(data, VolumeMesh(VL, CL));
    CHECK_EQ(level.mesh.X.cols(), VL.cols());
    CHECK_EQ(level.mesh.E.cols(), CL.cols());
    CHECK_EQ(level.u.rows(), Index(3));
    CHECK_EQ(level.u.cols(), VL.cols());
    CHECK_EQ(level.ecVE.rows(), 4);
    CHECK_EQ(level.ecVE.cols(), CR.cols());
    CHECK_EQ(level.NecVE.cols(), 4 * CR.cols());
    CHECK_EQ(level.NecVE.rows(), 4);
    CHECK_EQ(level.ilocalE.rows(), 4);
    CHECK_EQ(level.ilocalE.cols(), level.GEadj.size());
    CHECK_EQ(level.bIsDirichletVertex.size(), VR.cols());
    CHECK_FALSE((level.ecVE.array() < 0).any());
}