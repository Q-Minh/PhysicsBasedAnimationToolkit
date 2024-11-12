#include "Xpbd.h"

#include "Integrator.h"

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    BindIntegrator(m);
}

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat