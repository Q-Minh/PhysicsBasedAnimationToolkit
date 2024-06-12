
#include "Mesh.h"

#include "pbatautogen/Mesh_Mesh_line_Order_1_Dims_1.h"
#include "pbatautogen/Mesh_Mesh_line_Order_1_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_line_Order_1_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_line_Order_2_Dims_1.h"
#include "pbatautogen/Mesh_Mesh_line_Order_2_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_line_Order_2_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_line_Order_3_Dims_1.h"
#include "pbatautogen/Mesh_Mesh_line_Order_3_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_line_Order_3_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_1_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_1_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_2_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_2_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_3_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_triangle_Order_3_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_1_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_1_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_2_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_2_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_3_Dims_2.h"
#include "pbatautogen/Mesh_Mesh_quadrilateral_Order_3_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_tetrahedron_Order_1_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_tetrahedron_Order_2_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_tetrahedron_Order_3_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_hexahedron_Order_1_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_hexahedron_Order_2_Dims_3.h"
#include "pbatautogen/Mesh_Mesh_hexahedron_Order_3_Dims_3.h"

namespace pbat {
namespace py {
namespace fem {

void BindMesh(pybind11::module& m)
{
    BindMesh_Mesh_line_Order_1_Dims_1(m);
BindMesh_Mesh_line_Order_1_Dims_2(m);
BindMesh_Mesh_line_Order_1_Dims_3(m);
BindMesh_Mesh_line_Order_2_Dims_1(m);
BindMesh_Mesh_line_Order_2_Dims_2(m);
BindMesh_Mesh_line_Order_2_Dims_3(m);
BindMesh_Mesh_line_Order_3_Dims_1(m);
BindMesh_Mesh_line_Order_3_Dims_2(m);
BindMesh_Mesh_line_Order_3_Dims_3(m);
BindMesh_Mesh_triangle_Order_1_Dims_2(m);
BindMesh_Mesh_triangle_Order_1_Dims_3(m);
BindMesh_Mesh_triangle_Order_2_Dims_2(m);
BindMesh_Mesh_triangle_Order_2_Dims_3(m);
BindMesh_Mesh_triangle_Order_3_Dims_2(m);
BindMesh_Mesh_triangle_Order_3_Dims_3(m);
BindMesh_Mesh_quadrilateral_Order_1_Dims_2(m);
BindMesh_Mesh_quadrilateral_Order_1_Dims_3(m);
BindMesh_Mesh_quadrilateral_Order_2_Dims_2(m);
BindMesh_Mesh_quadrilateral_Order_2_Dims_3(m);
BindMesh_Mesh_quadrilateral_Order_3_Dims_2(m);
BindMesh_Mesh_quadrilateral_Order_3_Dims_3(m);
BindMesh_Mesh_tetrahedron_Order_1_Dims_3(m);
BindMesh_Mesh_tetrahedron_Order_2_Dims_3(m);
BindMesh_Mesh_tetrahedron_Order_3_Dims_3(m);
BindMesh_Mesh_hexahedron_Order_1_Dims_3(m);
BindMesh_Mesh_hexahedron_Order_2_Dims_3(m);
BindMesh_Mesh_hexahedron_Order_3_Dims_3(m);  
}

} // namespace fem
} // namespace py
} // namespace pbat
