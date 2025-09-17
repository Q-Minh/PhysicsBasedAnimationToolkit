#ifndef PYPBAT_FEM_FEM_H
#define PYPBAT_FEM_FEM_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace fem {

void Bind(nanobind::module_& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_FEM_H