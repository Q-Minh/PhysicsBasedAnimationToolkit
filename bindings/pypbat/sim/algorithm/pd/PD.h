/**
 * @file PD.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_ALGORITHM_PD_PD_H
#define PYPBAT_SIM_ALGORITHM_PD_PD_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::algorithm::pd {

void Bind(nanobind::module_& m);

} // namespace pbat::py::sim::algorithm::pd

#endif // PYPBAT_SIM_ALGORITHM_PD_PD_H
