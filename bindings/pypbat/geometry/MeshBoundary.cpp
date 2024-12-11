#include "MeshBoundary.h"

#include <pbat/geometry/MeshBoundary.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace geometry {

void BindMeshBoundary(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "simplex_mesh_boundary",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index n) {
            return pbat::geometry::SimplexMeshBoundary(C, n);
        },
        pyb::arg("C"),
        pyb::arg("n") = Index(-1),
        "Extracts the boundary of simplex mesh with n vertices and simplices C.\n"
        "Args:\n"
        "C (np.ndarray): |#simplex vertices|x|#simplices| array of simplices. Only triangles "
        "(#simplex vertices=3) and tetrahedra (#simplex vertices=4) are supported.\n"
        "n (int): Number of vertices\n"
        "Returns:\n"
        "(np.ndarray, np.ndarray): (V,F) where V is a |#boundary vertices| array of vertex indices "
        "and F is a |#dims|x|#boundary facets| array of boundary facets");
}

} // namespace geometry
} // namespace py
} // namespace pbat