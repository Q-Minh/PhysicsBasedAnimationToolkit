#ifndef PYPBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
#define PYPBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace geometry {

void BindTriangleAabbHierarchy(pybind11::module& m);

} // namespace geometry
} // namespace py
} // namespace pbat

#endif // PYPBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
