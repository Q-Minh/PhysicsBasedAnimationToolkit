/**
 * @file Dynamics.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_DYNAMICS_DYNAMICS_H
#define PYPBAT_SIM_DYNAMICS_DYNAMICS_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::dynamics {

void Bind(nanobind::module_& m);

} // namespace pbat::py::sim::dynamics

#endif // PYPBAT_SIM_DYNAMICS_DYNAMICS_H
