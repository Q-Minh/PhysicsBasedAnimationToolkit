#ifndef PYPBAT_FEM_MESH_QUADRATURE_H
#define PYPBAT_FEM_MESH_QUADRATURE_H

#include <pybind11/pybind11.h>

namespace pbat::py::fem {

void BindMeshQuadrature(pybind11::module& m);

} // namespace pbat::py::fem

#endif // PYPBAT_FEM_MESH_QUADRATURE_H