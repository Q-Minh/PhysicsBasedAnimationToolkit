#ifndef PYPBAT_FEM_MASS_H
#define PYPBAT_FEM_MASS_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace fem {

void BindMass(nanobind::module_& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MASS_H
