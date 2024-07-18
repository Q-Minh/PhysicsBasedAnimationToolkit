#ifndef PYPBAT_FEM_LAPLACIAN_H
#define PYPBAT_FEM_LAPLACIAN_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindLaplacian(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_LAPLACIAN_H
