#include "Xpbd.h"

#include "Integrator.h"

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void Bind(nanobind::module_& m)
{
    namespace nb = nanobind;
    BindIntegrator(m);
}

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat