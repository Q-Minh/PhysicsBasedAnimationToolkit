#include "Mesh.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>

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

    pyb::class_<Mesh>(m, "Mesh")
        .def(
            pyb::init<
                Eigen::Ref<MatrixX const> const&,
                Eigen::Ref<IndexMatrixX const> const&,
                EElement,
                int,
                int>(),
            pyb::arg("V"),
            pyb::arg("C"),
            pyb::arg("element"),
            pyb::arg("order") = 1,
            pyb::arg("dims")  = 3)
        .def("quadrature_points", &Mesh::QuadraturePoints)
        .def("quadrature_weights", &Mesh::QuadratureWeights)
        .def_property(
            "X",
            [](Mesh const& M) { return M.X(); },
            [](Mesh& M, Eigen::Ref<MatrixX const> const& X) { M.X() = X; })
        .def_property(
            "E",
            [](Mesh const& M) { return M.E(); },
            [](Mesh& M, Eigen::Ref<IndexMatrixX const> const& E) { M.E() = E; })
        .def_readonly("element", &Mesh::eElement)
        .def_readonly("order", &Mesh::kOrder)
        .def_readonly("dims", &Mesh::kDims);
}

Mesh::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C,
    EElement element,
    int order,
    int dims)
    : eElement(element), kOrder(order), kDims(dims), mMesh(nullptr)
{
    Apply([&]<class MeshType>([[maybe_unused]] MeshType* mesh) { mMesh = new MeshType(V, C); });
}

MatrixX Mesh::QuadraturePoints(int qOrder) const
{
    static auto constexpr kMaxQuadratureOrder = 8;
    MatrixX XG{};
    ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
            XG = mesh->template QuadraturePoints<QuadratureOrder>();
        },
        qOrder);
    return XG;
}

VectorX Mesh::QuadratureWeights(int qOrder) const
{
    static auto constexpr kMaxQuadratureOrder = 8;
    VectorX WG{};
    ApplyWithQuadrature<kMaxQuadratureOrder>(
        [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
            WG = mesh->template QuadratureWeights<QuadratureOrder>();
        },
        qOrder);
    return WG;
}

MatrixX const& Mesh::X() const
{
    MatrixX* XN = nullptr;
    Apply([&]<class MeshType>(MeshType* mesh) { XN = std::addressof(mesh->X); });
    return *XN;
}

IndexMatrixX const& Mesh::E() const
{
    IndexMatrixX* EN = nullptr;
    Apply([&]<class MeshType>(MeshType* mesh) { EN = std::addressof(mesh->E); });
    return *EN;
}

MatrixX& Mesh::X()
{
    MatrixX* XN = nullptr;
    Apply([&]<class MeshType>(MeshType* mesh) { XN = std::addressof(mesh->X); });
    return *XN;
}

IndexMatrixX& Mesh::E()
{
    IndexMatrixX* EN = nullptr;
    Apply([&]<class MeshType>(MeshType* mesh) { EN = std::addressof(mesh->E); });
    return *EN;
}

Mesh::~Mesh()
{
    if (mMesh != nullptr)
        Apply([]<class MeshType>(MeshType* mesh) { delete mesh; });
}

} // namespace fem
} // namespace py
} // namespace pbat