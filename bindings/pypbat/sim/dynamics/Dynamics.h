#ifndef PYPBAT_SIM_DYNAMICS_DYNAMICS_H
#define PYPBAT_SIM_DYNAMICS_DYNAMICS_H

#include <pybind11/pybind11.h>

namespace pbat::py::sim::dynamics {

void Bind(pybind11::module& m);

} // namespace pbat::py::sim::dynamics

#endif // PYPBAT_SIM_DYNAMICS_DYNAMICS_H
