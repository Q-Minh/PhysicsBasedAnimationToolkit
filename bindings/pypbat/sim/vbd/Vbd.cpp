#include "Vbd.h"

#include "Data.h"
#include "Integrator.h"
#include "lod/Lod.h"
#include "multigrid/Multigrid.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void Bind(pybind11::module& m)
{
    BindData(m);
    BindIntegrator(m);
    auto mlod = m.def_submodule("lod");
    lod::Bind(mlod);
    auto mmultigrid = m.def_submodule("multigrid");
    multigrid::Bind(mmultigrid);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat