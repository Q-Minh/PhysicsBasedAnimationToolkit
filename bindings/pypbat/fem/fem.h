#ifndef PYPBAT_FEM_FEM_H
#define PYPBAT_FEM_FEM_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void bind(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_FEM_H