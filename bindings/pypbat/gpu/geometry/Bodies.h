#ifndef PYPBAT_GPU_GEOMETRY_BODIES_H
#define PYPBAT_GPU_GEOMETRY_BODIES_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBodies(pybind11::module& m);

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GEOMETRY_BODIES_H
