#include "Multigrid.h"

#include "Hierarchy.h"
#include "HyperReduction.h"
#include "Integrator.h"
#include "Level.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void Bind(pybind11::module& m)
{
    BindHyperReduction(m);
    BindLevel(m);
    BindHierarchy(m);
    BindIntegrator(m);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat