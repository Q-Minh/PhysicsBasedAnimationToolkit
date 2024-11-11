#include "Xpbd.h"

#include "Data.h"
#include "Integrator.h"

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    BindData(m);
    BindIntegrator(m);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat