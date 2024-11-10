#ifndef PYPBAT_MATH_MOMENT_FITTING_H
#define PYPBAT_MATH_MOMENT_FITTING_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace math {

void BindMomentFitting(pybind11::module& m);

} // namespace math
} // namespace py
} // namespace pbat

#endif // PYPBAT_MATH_MOMENT_FITTING_H
