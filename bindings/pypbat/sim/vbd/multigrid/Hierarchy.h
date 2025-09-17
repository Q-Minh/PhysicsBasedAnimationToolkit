#ifndef PYPBAT_SIM_VBD_MULTIGRID_HIERARCHY_H
#define PYPBAT_SIM_VBD_MULTIGRID_HIERARCHY_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHierarchy(nanobind::module_& m);

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_VBD_MULTIGRID_HIERARCHY_H
