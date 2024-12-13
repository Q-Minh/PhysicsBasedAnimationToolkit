#include "DirichletEnergy.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

DirichletEnergy::DirichletEnergy(
    Data const& problem,
    DirichletQuadrature const& DQ)
    : muD(problem.muD), dg(DQ.Xg)
{
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat