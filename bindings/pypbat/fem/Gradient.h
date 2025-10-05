#ifndef PYPBAT_FEM_GRADIENT_H
#define PYPBAT_FEM_GRADIENT_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace fem {

void BindGradient(nanobind::module_& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_GRADIENT_H
