#include "Geometry.h"

#include "AxisAlignedBoundingBox.h"
#include "TetrahedralAabbHierarchy.h"

namespace pbat {
namespace py {
namespace geometry {

void Bind(pybind11::module& m)
{
    BindAxisAlignedBoundingBox(m);
    BindTetrahedralAabbHierarchy(m);
}

} // namespace geometry
} // namespace py
} // namespace pbat
