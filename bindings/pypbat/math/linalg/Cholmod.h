#ifndef PYPBAT_MATH_LINALG_CHOLMOD_H
#define PYPBAT_MATH_LINALG_CHOLMOD_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindCholmod(nanobind::module_& m);

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat

#endif // PYPBAT_MATH_LINALG_CHOLMOD_H
