#ifndef PYPBAT_FEM_LOAD_VECTOR_H
#define PYPBAT_FEM_LOAD_VECTOR_H

#include <pybind11/pybind11.h>

namespace pbat::py::fem {

void BindLoadVector(pybind11::module& m);

} // namespace pbat::py::fem

#endif // PYPBAT_FEM_LOAD_VECTOR_H
