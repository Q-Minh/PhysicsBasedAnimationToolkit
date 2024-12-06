#include "MeshBoundary.h"

#include <pbat/geometry/MeshBoundary.h>
#include <pybind11/eigen.h>

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
        "n (int): Number of vertices");
}

} // namespace geometry
} // namespace py
} // namespace pbat