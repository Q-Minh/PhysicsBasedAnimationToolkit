
#include "ShapeFunctions.h"

#include "pbatautogen/ShapeFunctions_Mesh_line_Order_1_Dims_1.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_1_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_1_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_2_Dims_1.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_2_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_2_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_3_Dims_1.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_3_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_line_Order_3_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_1_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_1_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_2_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_2_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_3_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_triangle_Order_3_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_1_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_1_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_2_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_2_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_3_Dims_2.h"
#include "pbatautogen/ShapeFunctions_Mesh_quadrilateral_Order_3_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_tetrahedron_Order_1_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_tetrahedron_Order_2_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_tetrahedron_Order_3_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_hexahedron_Order_1_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_hexahedron_Order_2_Dims_3.h"
#include "pbatautogen/ShapeFunctions_Mesh_hexahedron_Order_3_Dims_3.h"

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m)
{
    BindShapeFunctions_Mesh_line_Order_1_Dims_1(m);
BindShapeFunctions_Mesh_line_Order_1_Dims_2(m);
BindShapeFunctions_Mesh_line_Order_1_Dims_3(m);
BindShapeFunctions_Mesh_line_Order_2_Dims_1(m);
BindShapeFunctions_Mesh_line_Order_2_Dims_2(m);
BindShapeFunctions_Mesh_line_Order_2_Dims_3(m);
BindShapeFunctions_Mesh_line_Order_3_Dims_1(m);
BindShapeFunctions_Mesh_line_Order_3_Dims_2(m);
BindShapeFunctions_Mesh_line_Order_3_Dims_3(m);
BindShapeFunctions_Mesh_triangle_Order_1_Dims_2(m);
BindShapeFunctions_Mesh_triangle_Order_1_Dims_3(m);
BindShapeFunctions_Mesh_triangle_Order_2_Dims_2(m);
BindShapeFunctions_Mesh_triangle_Order_2_Dims_3(m);
BindShapeFunctions_Mesh_triangle_Order_3_Dims_2(m);
BindShapeFunctions_Mesh_triangle_Order_3_Dims_3(m);
BindShapeFunctions_Mesh_quadrilateral_Order_1_Dims_2(m);
BindShapeFunctions_Mesh_quadrilateral_Order_1_Dims_3(m);
BindShapeFunctions_Mesh_quadrilateral_Order_2_Dims_2(m);
BindShapeFunctions_Mesh_quadrilateral_Order_2_Dims_3(m);
BindShapeFunctions_Mesh_quadrilateral_Order_3_Dims_2(m);
BindShapeFunctions_Mesh_quadrilateral_Order_3_Dims_3(m);
BindShapeFunctions_Mesh_tetrahedron_Order_1_Dims_3(m);
BindShapeFunctions_Mesh_tetrahedron_Order_2_Dims_3(m);
BindShapeFunctions_Mesh_tetrahedron_Order_3_Dims_3(m);
BindShapeFunctions_Mesh_hexahedron_Order_1_Dims_3(m);
BindShapeFunctions_Mesh_hexahedron_Order_2_Dims_3(m);
BindShapeFunctions_Mesh_hexahedron_Order_3_Dims_3(m);  
}

} // namespace fem
} // namespace py
} // namespace pbat
