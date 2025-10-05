#ifndef PYPBAT_FEM_MESH_QUADRATURE_H
#define PYPBAT_FEM_MESH_QUADRATURE_H

#include <nanobind/nanobind.h>

namespace pbat::py::fem {

void BindMeshQuadrature(nanobind::module_& m);

} // namespace pbat::py::fem

#endif // PYPBAT_FEM_MESH_QUADRATURE_H