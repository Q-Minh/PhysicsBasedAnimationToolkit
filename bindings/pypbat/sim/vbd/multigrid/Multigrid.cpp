#include "Multigrid.h"

#include "DirichletEnergy.h"
#include "ElasticEnergy.h"
#include "Hierarchy.h"
#include "Integrator.h"
#include "Level.h"
#include "MomentumEnergy.h"
#include "Prolongation.h"
#include "Quadrature.h"
#include "Restriction.h"
#include "Smoother.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void Bind(pybind11::module& m)
{
    BindQuadrature(m);
    BindDirichletEnergy(m);
    BindElasticEnergy(m);
    BindMomentumEnergy(m);
    BindLevel(m);
    BindRestriction(m);
    BindProlongation(m);
    BindSmoother(m);
    BindHierarchy(m);
    BindIntegrator(m);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat