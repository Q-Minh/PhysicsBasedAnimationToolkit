/**
 * @file Optimization.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for optimization bindings.
 * @version 0.1
 * @date 2025-10-29
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PYPBAT_MATH_OPTIMIZATION_OPTIMIZATION_H
#define PYPBAT_MATH_OPTIMIZATION_OPTIMIZATION_H

#include <nanobind/nanobind.h>

namespace pbat::py::math::optimization {

void Bind(nanobind::module_& m);

} // namespace pbat::py::math::optimization

#endif // PYPBAT_MATH_OPTIMIZATION_OPTIMIZATION_H
