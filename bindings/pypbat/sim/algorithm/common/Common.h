/**
 * @file Common.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-10-28
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PYPBAT_SIM_ALGORITHM_COMMON_COMMON_H
#define PYPBAT_SIM_ALGORITHM_COMMON_COMMON_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::algorithm::common {

void Bind(nanobind::module_& m);

} // namespace pbat::py::sim::algorithm::common

#endif // PYPBAT_SIM_ALGORITHM_COMMON_COMMON_H
