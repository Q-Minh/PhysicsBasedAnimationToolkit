#ifndef PYPBAT_SIM_ALGORITHM_NEWTON_CORE_H
#define PYPBAT_SIM_ALGORITHM_NEWTON_CORE_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::algorithm::newton {

void BindCore(nanobind::module_& m);

} // namespace pbat::py::sim::algorithm::newton

#endif // PYPBAT_SIM_ALGORITHM_NEWTON_CORE_H
