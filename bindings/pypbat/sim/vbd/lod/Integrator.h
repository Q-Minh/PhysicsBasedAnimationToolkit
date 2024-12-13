#ifndef PYPBAT_SIM_VBD_LOD_INTEGRATOR_H
#define PYPBAT_SIM_VBD_LOD_INTEGRATOR_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindIntegrator(pybind11::module& m);

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_VBD_LOD_INTEGRATOR_H
