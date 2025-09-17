#include "Vbd.h"

#include "Integrator.h"

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void Bind(nanobind::module_& m)
{
    BindIntegrator(m);
}

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat