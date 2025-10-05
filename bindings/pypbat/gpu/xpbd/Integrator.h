#ifndef PYPBAT_GPU_XPBD_INTEGRATOR_H
#define PYPBAT_GPU_XPBD_INTEGRATOR_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void BindIntegrator(nanobind::module_& m);

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_XPBD_INTEGRATOR_H
