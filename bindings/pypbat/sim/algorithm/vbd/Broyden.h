#ifndef PYPBAT_SIM_ALGORITHM_VBD_BROYDEN_H
#define PYPBAT_SIM_ALGORITHM_VBD_BROYDEN_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::algorithm::vbd {

void BindBroyden(nanobind::module_& m);

} // namespace pbat::py::sim::algorithm::vbd

#endif // PYPBAT_SIM_ALGORITHM_VBD_BROYDEN_H
