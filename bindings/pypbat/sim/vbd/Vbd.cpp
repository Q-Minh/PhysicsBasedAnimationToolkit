#include "Vbd.h"

#include "Data.h"
#include "Hierarchy.h"
#include "Integrator.h"
#include "Level.h"
#include "MultiScaleIntegrator.h"
#include "Prolongation.h"
#include "Restriction.h"
#include "Smoother.h"
#include "multigrid/Multigrid.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void Bind(pybind11::module& m)
{
    BindData(m);
    BindIntegrator(m);
    BindLevel(m);
    BindProlongation(m);
    BindRestriction(m);
    BindSmoother(m);
    BindHierarchy(m);
    BindMultiScaleIntegrator(m);
    auto mmultigrid = m.def_submodule("multigrid");
    multigrid::Bind(mmultigrid);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat