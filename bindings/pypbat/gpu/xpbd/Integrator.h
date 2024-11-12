#ifndef PYPBAT_GPU_XPBD_INTEGRATOR_H
#define PYPBAT_GPU_XPBD_INTEGRATOR_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void BindIntegrator(pybind11::module& m);

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_XPBD_INTEGRATOR_H
