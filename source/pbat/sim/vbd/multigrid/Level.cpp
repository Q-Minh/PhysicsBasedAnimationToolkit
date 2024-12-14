#include "Level.h"

#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <ranges>
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
      ecK(),
      NecK(),
      GKptr(),
      GKadj(),
      GKilocal(),
      bIsDirichletVertex()
{
    // Coarse mesh
    u.setZero(mesh.X.rows(), mesh.X.cols());
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
    vc2ef.reserve(nFineElements * nVertexPerElement * nVertexPerElement);
    for (auto ef = 0; ef < nFineElements; ++ef)
        for (auto ec : ecVE.col(ef))
            for (auto vc : mesh.E.col(ec))
                vc2ef.push_back({vc, ef});
    std::tie(GEptr, GEadj) =
        graph::MatrixToAdjacency(graph::AdjacencyMatrixFromEdges(vc2ef.begin(), vc2ef.end()));
    ilocalE.setConstant(4, GEadj.size(), Index(-1));
    graph::ForEachEdge(GEptr, GEadj, [&](Index vc, Index ef, Index eid) {
        for (Index iflocal = 0; iflocal < 4; ++iflocal)
        {
            Index ec = ecVE(iflocal, ef);
            for (Index iclocal = 0; iclocal < 4; ++iclocal)
                if (vc == mesh.E(iclocal, ec))
                    ilocalE(iflocal, eid) = iclocal;
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
    bIsDirichletVertex.setConstant(false);
    bIsDirichletVertex(data.dbc).setConstant(true);
}

void Level::Prolong(Data& data) const
{
    tbb::parallel_for(Index(0), data.x.cols(), [&](Index i) {
        Index ec = ecK(i);
        auto uec = u(Eigen::placeholders::all, mesh.E.col(ec));
        auto Nec = NecK.col(i);
        data.x.col(i) += uec * Nec;
    });
}

void Level::Smooth(Data const& data)
{
    u.setZero();
    Index const nPartitions = Pptr.size() - 1;
    for (Index p = 0; p < nPartitions; ++p)
    {
        auto pBegin = Pptr(p);
        auto pEnd   = Pptr(p + 1);
        tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
            Index i     = Padj(kp);
            auto gBegin = GEptr(i);
            auto gEnd   = GEptr(i + 1);
            // Elastic energy
            for (auto kg = gBegin; kg < gEnd; ++kg)
            {
                Index ef              = GEadj(kg);
                Matrix<4, 3> GNef     = data.GP.block<4, 3>(0, 3 * ef);
                Matrix<4, 4> N        = NecVE.block<4, 4>(0, 4 * ef);
                IndexVector<4> ilocal = ilocalE.col(kg);
                Matrix<3, 4> xe       = data.x(Eigen::placeholders::all, data.mesh.E.col(ef));
                IndexMatrix<4, 4> ec  = mesh.E(Eigen::placeholders::all, ecVE.col(ef));
                for (auto iflocal = 0; iflocal < 4; ++iflocal)
                {
                    xe.col(iflocal) +=
                        u(Eigen::placeholders::all, ec.col(iflocal)) * N.col(iflocal);
                }
                Matrix<3, 3> F = xe * GNef;
                using namespace pbat::math::linalg;
                Scalar mu                      = data.lame(0, ef);
                Scalar lambda                  = data.lame(1, ef);
                mini::SVector<Scalar, 9> gF    = mini::Zeros<Scalar, 9>();
                mini::SMatrix<Scalar, 9, 9> HF = mini::Zeros<Scalar, 9, 9>();
                physics::StableNeoHookeanEnergy<3> Psi{};
                Psi.gradAndHessian(mini::FromEigen(F), mu, lambda, gF, HF);
                mini::SMatrix<Scalar, 3, 3> Hu = mini::Zeros<Scalar, 3, 3>();
                mini::SVector<Scalar, 3> gu    = mini::Zeros<Scalar, 3, 1>();
                for (auto i = 0; i < 4; ++i)
                {
                    for (auto j = 0; j < 4; ++j)
                    {
                        if (ilocal(i) >= 0 and ilocal(j) >= 0)
                        {
                            Hu += N(ilocal(i), i) * N(ilocal(j), j) *
                                  fem::HessianBlockWrtDofs<VolumeMesh::ElementType, 3>(
                                      HF,
                                      mini::FromEigen(GNef),
                                      i,
                                      j);
                        }
                    }
                }
                for (auto i = 0; i < 4; ++i)
                {
                    if (ilocal(i) >= 0)
                    {
                        gu += N(ilocal(i), i) *
                              fem::GradientSegmentWrtDofs<VolumeMesh::ElementType, 3>(
                                  gF,
                                  mini::FromEigen(GNef),
                                  i);
                    }
                }
                Scalar wg = data.wg(ef);
                Hu *= wg;
                gu *= wg;
            }
        });
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat