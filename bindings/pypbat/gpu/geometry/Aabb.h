#ifndef PYPBAT_GPU_GEOMETRY_AABB_H
#define PYPBAT_GPU_GEOMETRY_AABB_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindAabb(nanobind::module_& m);

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GEOMETRY_AABB_H
