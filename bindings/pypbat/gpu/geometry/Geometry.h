#ifndef PYPBAT_GPU_GEOMETRY_GEOMETRY_H
#define PYPBAT_GPU_GEOMETRY_GEOMETRY_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void Bind(pybind11::module& m);

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GEOMETRY_GEOMETRY_H
