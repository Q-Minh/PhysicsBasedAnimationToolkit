/**
 * @file TriangleConstrainedTrustRegionSr1.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for triangle-constrained trust-region SR1 optimizer bindings.
 * @version 0.1
 * @date 2025-11-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PYPBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
#define PYPBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H

#include <nanobind/nanobind.h>

namespace pbat::py::math::optimization {

void BindTriangleConstrainedTrustRegionSr1(nanobind::module_& m);

} // namespace pbat::py::math::optimization

#endif // PYPBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
