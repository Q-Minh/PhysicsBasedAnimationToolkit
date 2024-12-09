#include "Level.h"

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

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
    x                       = mesh.X;
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

Level& Level::WithCageQuadrature(Data const& problem, ECageQuadratureStrategy eStrategy)
{
    Qcage = CageQuadrature(problem.mesh, mesh, eStrategy);
    return *this;
}

Level& Level::WithDirichletQuadrature(Data const& problem)
{
    Qdirichlet = DirichletQuadrature(problem.mesh, mesh, problem.m, problem.dbc);
    return *this;
}

Level& Level::WithMomentumEnergy(Data const& problem)
{
    Ekinetic = MomentumEnergy(problem, mesh, Qcage);
    return *this;
}

Level& Level::WithElasticEnergy(Data const& problem)
{
    Epotential = ElasticEnergy(problem, mesh, Qcage);
    return *this;
}

Level& Level::WithDirichletEnergy(Data const& problem)
{
    Edirichlet = DirichletEnergy(problem, mesh, Qdirichlet);
    return *this;
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat