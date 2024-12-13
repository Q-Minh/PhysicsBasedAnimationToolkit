#include "Level.h"

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

Level::Level(VolumeMesh CM)
    : x(),
      colors(),
      ptr(),
      adj(),
      mesh(std::move(CM)),
      Qcage(),
      Qdirichlet(),
      Ekinetic(),
      Epotential(),
      Edirichlet()
{
    // Initialize coarse vertex positions from rest pose
    x = mesh.X;
    // Compute vertex parallelization scheme
    auto GVV                = graph::MeshPrimalGraph(mesh.E, mesh.X.cols());
    auto [GVVp, GVVv, GVVw] = graph::MatrixToAdjacency(GVV);
    colors                  = graph::GreedyColor(
        GVVp,
        GVVv,
        graph::EGreedyColorOrderingStrategy::LargestDegree,
        graph::EGreedyColorSelectionStrategy::LeastUsed);
    std::tie(ptr, adj) = graph::MapToAdjacency(colors);
}

Level& Level::WithCageQuadrature(Data const& problem, CageQuadratureParameters const& params)
{
    Qcage = CageQuadrature(problem.mesh, mesh, params);
    return *this;
}

Level& Level::WithDirichletQuadrature(Data const& problem)
{
    Qdirichlet = DirichletQuadrature(problem.mesh, mesh, problem.m, problem.dbc);
    return *this;
}

Level& Level::WithMomentumEnergy(Data const& problem)
{
    Ekinetic = MomentumEnergy(problem, Qcage);
    return *this;
}

Level& Level::WithElasticEnergy(Data const& problem)
{
    Epotential = ElasticEnergy(problem, Qcage);
    return *this;
}

Level& Level::WithDirichletEnergy(Data const& problem)
{
    Edirichlet = DirichletEnergy(problem, Qdirichlet);
    return *this;
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat