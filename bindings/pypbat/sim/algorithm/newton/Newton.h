/**
 * @file Newton.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_ALGORITHM_NEWTON_NEWTON_H
#define PYPBAT_SIM_ALGORITHM_NEWTON_NEWTON_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::algorithm::newton {

void Bind(nanobind::module_& m);

} // namespace pbat::py::sim::algorithm::newton

#endif // PYPBAT_SIM_ALGORITHM_NEWTON_NEWTON_H
