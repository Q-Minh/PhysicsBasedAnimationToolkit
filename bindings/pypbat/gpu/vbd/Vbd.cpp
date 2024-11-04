#include "Vbd.h"

#include "Integrator.h"

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void Bind(pybind11::module& m)
{
    BindIntegrator(m);
}

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat