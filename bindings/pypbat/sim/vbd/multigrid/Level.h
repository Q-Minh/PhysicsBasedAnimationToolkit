#ifndef PYPBAT_SIM_VBD_MULTIGRID_LEVEL_H
#define PYPBAT_SIM_VBD_MULTIGRID_LEVEL_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindLevel(pybind11::module& m);

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_VBD_MULTIGRID_LEVEL_H
