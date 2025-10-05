#include "Mesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <utility>

namespace pbat {
namespace py {
namespace fem {

void BindMesh(nanobind::module_& m)
{
    namespace nb = nanobind;

    nb::enum_<EElement>(m, "Element")
        .value("Line", EElement::Line)
        .value("Triangle", EElement::Triangle)
        .value("Quadrilateral", EElement::Quadrilateral)
        .value("Tetrahedron", EElement::Tetrahedron)
        .value("Hexahedron", EElement::Hexahedron)
        .export_values();

    m.def(
        "dims",
        [](EElement eElement) {
            int dims = 0;
            ApplyToElement(eElement, 1, [&]<class ElementType>() { dims = ElementType::kDims; });
            if (dims == 0)
            {
                throw std::invalid_argument(
                    fmt::format("Invalid finite element type: {}", static_cast<int>(eElement)));
            }
            return dims;
        },
        nb::arg("element"),
        "Return the reference dimensionality of the given finite element type.\n\n"
        "Args:\n"
        "    element (EElement): Type of the finite element.\n\n"
        "Returns:\n"
        "    int: Number of dimensions of the finite element type.");

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;

    m.def(
        "mesh",
        [](nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> V,
           nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> C,
           EElement element,
           int order,
           int dims) {
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> X;
            Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> E;
            ApplyToMesh<TScalar, TIndex>(element, order, dims, [&]<pbat::fem::CMesh TMesh>() {
                using MeshType = TMesh;
                MeshType mesh(V, C);
                X = std::move(mesh.X);
                E = std::move(mesh.E);
            });
            return std::make_pair(X, E);
        },
        nb::arg("V"),
        nb::arg("C"),
        nb::arg("element"),
        nb::arg("order") = 1,
        nb::arg("dims")  = 3,
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
}

} // namespace fem
} // namespace py
} // namespace pbat