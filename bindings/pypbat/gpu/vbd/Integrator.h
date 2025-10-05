#ifndef PYPBAT_GPU_VBD_INTEGRATOR_H
#define PYPBAT_GPU_VBD_INTEGRATOR_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void BindIntegrator(nanobind::module_& m);

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_VBD_INTEGRATOR_H
