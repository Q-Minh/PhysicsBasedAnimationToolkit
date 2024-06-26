#ifndef PYPBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H
#define PYPBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace geometry {

void BindAxisAlignedBoundingBox(pybind11::module& m);

} // namespace geometry
} // namespace py
} // namespace pbat

#endif // PYPBAT_GEOMETRY_AXIS_ALIGNED_BOUNDING_BOX_H
