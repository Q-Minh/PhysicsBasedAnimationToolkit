#ifndef PYPBAT_SIM_VBD_INTEGRATOR_H
#define PYPBAT_SIM_VBD_INTEGRATOR_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindIntegrator(nanobind::module_& m);

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_VBD_INTEGRATOR_H
