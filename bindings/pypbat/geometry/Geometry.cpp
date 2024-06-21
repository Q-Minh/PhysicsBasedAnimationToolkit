#include "Geometry.h"

#include "AxisAlignedBoundingBox.h"
#include "TetrahedralAabbHierarchy.h"
#include "TriangleAabbHierarchy.h"

namespace pbat {
namespace py {
namespace geometry {

void Bind(pybind11::module& m)
{
    BindAxisAlignedBoundingBox(m);
    BindTetrahedralAabbHierarchy(m);
    BindTriangleAabbHierarchy(m);
}

} // namespace geometry
} // namespace py
} // namespace pbat
