#ifndef PYPBAT_FEM_LOAD_VECTOR_H
#define PYPBAT_FEM_LOAD_VECTOR_H

#include <nanobind/nanobind.h>

namespace pbat::py::fem {

void BindLoadVector(nanobind::module_& m);

} // namespace pbat::py::fem

#endif // PYPBAT_FEM_LOAD_VECTOR_H
