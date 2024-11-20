#include "Vbd.h"

#include "Data.h"
#include "Integrator.h"
#include "Restriction.h"

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void Bind(pybind11::module& m)
{
    BindData(m);
    BindIntegrator(m);
    BindRestriction(m);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat