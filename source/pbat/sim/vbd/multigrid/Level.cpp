#include "Level.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Level::Level(VolumeMesh CM)
    : mesh(std::move(CM)), Qcage(), Qdirichlet(), Ekinetic(), Epotential(), Edirichlet()
{
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