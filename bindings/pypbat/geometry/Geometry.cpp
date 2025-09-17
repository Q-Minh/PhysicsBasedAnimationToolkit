#include "Geometry.h"

#include "AxisAlignedBoundingBox.h"
#include "HashGrid.h"
#include "HierarchicalHashGrid.h"
#include "MeshBoundary.h"
#include "TetrahedralAabbHierarchy.h"
#include "TriangleAabbHierarchy.h"
#include "sdf/Sdf.h"

namespace pbat::py::geometry {

void Bind(nanobind::module_& m)
{
    BindAxisAlignedBoundingBox(m);
    BindHashGrid(m);
    BindHierarchicalHashGrid(m);
    BindTetrahedralAabbHierarchy(m);
    BindTriangleAabbHierarchy(m);
    BindMeshBoundary(m);
    auto msdf = m.def_submodule("sdf", "Signed Distance Functions");
    sdf::Bind(msdf);
}

} // namespace pbat::py::geometry
