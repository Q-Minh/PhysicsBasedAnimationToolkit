#ifndef PYPBAT_SIM_VBD_LOD_RESTRICTION_H
#define PYPBAT_SIM_VBD_LOD_RESTRICTION_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindRestriction(pybind11::module& m);

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_VBD_LOD_RESTRICTION_H
