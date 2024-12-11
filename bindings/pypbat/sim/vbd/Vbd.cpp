#include "Vbd.h"

#include "Data.h"
#include "Integrator.h"
#include "multigrid/Multigrid.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void Bind(pybind11::module& m)
{
    BindData(m);
    BindIntegrator(m);
    auto mmultigrid = m.def_submodule("multigrid");
    multigrid::Bind(mmultigrid);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat