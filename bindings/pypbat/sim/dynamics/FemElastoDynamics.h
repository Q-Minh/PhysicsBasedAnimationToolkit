#ifndef PYPBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
#define PYPBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::dynamics {

void BindFemElastoDynamics(nanobind::module_& m);

} // namespace pbat::py::sim::dynamics

#endif // PYPBAT_SIM_DYNAMICS_FEMELASTODYNAMICS_H
