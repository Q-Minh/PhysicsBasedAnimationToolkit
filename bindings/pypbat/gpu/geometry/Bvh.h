#ifndef PYPBAT_GPU_GEOMETRY_BVH_H
#define PYPBAT_GPU_GEOMETRY_BVH_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBvh(nanobind::module_& m);

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GEOMETRY_BVH_H
