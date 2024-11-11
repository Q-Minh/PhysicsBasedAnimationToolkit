#ifndef PYPBAT_SIM_XPBD_INTEGRATOR_H
#define PYPBAT_SIM_XPBD_INTEGRATOR_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindIntegrator(pybind11::module& m);

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_XPBD_INTEGRATOR_H
