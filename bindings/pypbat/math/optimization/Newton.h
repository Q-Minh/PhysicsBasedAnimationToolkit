/**
 * @file Newton.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for Newton's method bindings.
 * @version 0.1
 * @date 2025-10-29
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PYPBAT_MATH_OPTIMIZATION_NEWTON_H
#define PYPBAT_MATH_OPTIMIZATION_NEWTON_H

#include <nanobind/nanobind.h>

namespace pbat::py::math::optimization {

void BindNewton(nanobind::module_& m);

} // namespace pbat::py::math::optimization

#endif // PYPBAT_MATH_OPTIMIZATION_NEWTON_H
