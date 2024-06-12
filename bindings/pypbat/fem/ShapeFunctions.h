
#ifndef PYPBAT_FEM_SHAPE_FUNCTIONS_H
#define PYPBAT_FEM_SHAPE_FUNCTIONS_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_SHAPE_FUNCTIONS_H
