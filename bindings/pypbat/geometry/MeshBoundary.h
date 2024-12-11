#ifndef PYPBAT_GEOMETRY_MESH_BOUNDARY_H
#define PYPBAT_GEOMETRY_MESH_BOUNDARY_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace geometry {

void BindMeshBoundary(pybind11::module& m);

} // namespace geometry
} // namespace py
} // namespace pbat

#endif // PYPBAT_GEOMETRY_MESH_BOUNDARY_H
