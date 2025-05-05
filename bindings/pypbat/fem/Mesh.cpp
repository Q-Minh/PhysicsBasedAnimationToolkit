#include "Mesh.h"

#include <pybind11/eigen.h>
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
            pyb::arg("dims")  = 3,
            "Construct FEM mesh of the given shape function order and dimensions given some input "
            "geometric mesh V,C.")
        .def("quadrature_points", &Mesh::QuadraturePoints)
        .def("quadrature_weights", &Mesh::QuadratureWeights)
        .def_property(
            "X",
            [](Mesh const& M) { return M.X(); },
            [](Mesh& M, Eigen::Ref<MatrixX const> const& X) { M.X() = X; },
            "|#dims|x|#nodes| array of nodal positions")
        .def_property(
            "E",
            [](Mesh const& M) { return M.E(); },
            [](Mesh& M, Eigen::Ref<IndexMatrixX const> const& E) { M.E() = E; },
            "|#element nodes|x|#elements| array of element nodal indices")
        .def_readonly("element", &Mesh::eElement)
        .def_readonly("order", &Mesh::mOrder, "Shape function order")
        .def_readonly("dims", &Mesh::mDims, "Domain dimensions");
}

Mesh::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C,
    EElement element,
    int order,
    int dims)
    : eElement(element), mOrder(order), mDims(dims), mMesh(nullptr), bOwnMesh(true)
{
    Apply([&]<class MeshType>([[maybe_unused]] MeshType* mesh) { mMesh = new MeshType(V, C); });
}

Mesh::Mesh(void* meshImpl, EElement element, int order, int dims)
    : eElement(element), mOrder(order), mDims(dims), mMesh(meshImpl), bOwnMesh{false}
{
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

Eigen::Map<MatrixX> Mesh::X() const
{
    Scalar* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Apply([&]<class MeshType>(MeshType* mesh) {
        data = mesh->X.data();
        rows = mesh->X.rows();
        cols = mesh->X.cols();
    });
    return Eigen::Map<MatrixX>(data, rows, cols);
}

Eigen::Map<IndexMatrixX> Mesh::E() const
{
    Index* data{nullptr};
    Index rows{-1};
    Index cols{-1};
    Apply([&]<class MeshType>(MeshType* mesh) {
        data = mesh->E.data();
        rows = mesh->E.rows();
        cols = mesh->E.cols();
    });
    return Eigen::Map<IndexMatrixX>(data, rows, cols);
}

Mesh::~Mesh()
{
    if (mMesh != nullptr and bOwnMesh)
        Apply([]<class MeshType>(MeshType* mesh) { delete mesh; });
}

} // namespace fem
} // namespace py
} // namespace pbat