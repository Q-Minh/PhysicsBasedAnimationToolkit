#include "Mesh.h"

#include <pybind11/eigen.h>
#include <string>
#include <utility>

namespace pbat {
namespace py {
namespace fem {

void BindMesh(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::enum_<EElement>(m, "Element")
        .value("Line", EElement::Line)
        .value("Triangle", EElement::Triangle)
        .value("Quadrilateral", EElement::Quadrilateral)
        .value("Tetrahedron", EElement::Tetrahedron)
        .value("Hexahedron", EElement::Hexahedron)
        .export_values();

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            using MatrixType      = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;
            using IndexMatrixType = Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>;
            m.def(
                "mesh",
                [](pyb::EigenDRef<MatrixType const> V,
                   pyb::EigenDRef<IndexMatrixType const> C,
                   EElement element,
                   int order,
                   int dims) {
                    MatrixType X;
                    IndexMatrixType E;
                    ApplyToMesh<TScalar, TIndex>(
                        element,
                        order,
                        dims,
                        [&]<pbat::fem::CMesh TMesh>() {
                            using MeshType = TMesh;
                            MeshType mesh(V, C);
                            X = std::move(mesh.X);
                            E = std::move(mesh.E);
                        });
                    return std::make_pair(X, E);
                },
                pyb::arg("V"),
                pyb::arg("C"),
                pyb::arg("element"),
                pyb::arg("order") = 1,
                pyb::arg("dims")  = 3,
                "Compute an FEM mesh from the input geometric mesh.\n\n"
                "Args:\n"
                "    V (numpy.ndarray): Vertex coordinates of the geometric mesh.\n"
                "    C (numpy.ndarray): Connectivity of the geometric mesh.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    dims (int): Number of dimensions of the mesh.\n\n"
                "Returns:\n"
                "    Tuple[numpy.ndarray, numpy.ndarray]: (X, E) where X are the `|# dims| x |# "
                "nodes|` nodal coordinates and E are the `|# elem. nodes.| x |# elements|` element "
                "connectivity.\n");
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat