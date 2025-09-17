#ifndef PYPBAT_SIM_XPBD_INTEGRATOR_H
#define PYPBAT_SIM_XPBD_INTEGRATOR_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindIntegrator(nanobind::module_& m);

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_XPBD_INTEGRATOR_H
