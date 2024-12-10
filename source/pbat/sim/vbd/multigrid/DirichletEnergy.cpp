#include "DirichletEnergy.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

DirichletEnergy::DirichletEnergy(
    Data const& problem,
    VolumeMesh const& CM,
    DirichletQuadrature const& DQ)
    : muD(problem.muD), Ncg(), dg()
{
    dg  = DQ.Xg;
    Ncg = fem::ShapeFunctionsAt(CM, DQ.eg, DQ.Xg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat