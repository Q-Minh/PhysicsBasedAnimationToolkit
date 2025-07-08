#include "Geometry.h"

#include "AxisAlignedBoundingBox.h"
#include "HashGrid.h"
#include "HierarchicalHashGrid.h"
#include "MeshBoundary.h"
#include "TetrahedralAabbHierarchy.h"
#include "TriangleAabbHierarchy.h"

namespace pbat {
namespace py {
namespace geometry {

void Bind(pybind11::module& m)
{
    BindAxisAlignedBoundingBox(m);
    BindHashGrid(m);
    BindHierarchicalHashGrid(m);
    BindTetrahedralAabbHierarchy(m);
    BindTriangleAabbHierarchy(m);
    BindMeshBoundary(m);
}

} // namespace geometry
} // namespace py
} // namespace pbat
