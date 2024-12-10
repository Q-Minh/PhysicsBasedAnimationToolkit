#include "ShapeFunctions.h"

#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m)
{
    namespace pyb = pybind11;

    m.def(
        "shape_functions_at",
        [](Mesh const& M,
           Eigen::Ref<IndexVectorX const> const& eg,
           Eigen::Ref<MatrixX const> const& Xg,
           bool bXgInReferenceSpace) {
            MatrixX N;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                N = pbat::fem::ShapeFunctionsAt(*mesh, eg, Xg, bXgInReferenceSpace);
            });
            return N;
        },
        pyb::arg("mesh"),
        pyb::arg("eg"),
        pyb::arg("Xg"),
        pyb::arg("in_reference_space") = false,
        "|#elem. nodes|x|Xg.shape[1]| matrix of nodal shape functions at evaluation points Xg");

    m.def(
        "shape_function_matrix",
        [](Mesh const& M, int qOrder) {
            auto constexpr kMaxQuadratureOrder = 6;
            CSRMatrix N;
            M.ApplyWithQuadrature<kMaxQuadratureOrder>(
                [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
                    N = pbat::fem::ShapeFunctionMatrix<QuadratureOrder>(*mesh);
                },
                qOrder);
            return N;
        },
        pyb::arg("mesh"),
        pyb::arg("quadrature_order") = 1,
        "|#elements * #quad.pts.| x |#nodes| shape function matrix");

    m.def(
        "shape_function_matrix",
        [](Mesh const& M,
           Eigen::Ref<IndexVectorX const> const& eg,
           Eigen::Ref<MatrixX const> const& Xg,
           bool bXgInReferenceSpace) {
            CSRMatrix N;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                N = pbat::fem::ShapeFunctionMatrix(*mesh, eg, Xg, bXgInReferenceSpace);
            });
            return N;
        },
        pyb::arg("mesh"),
        pyb::arg("eg"),
        pyb::arg("Xg"),
        pyb::arg("in_reference_space") = false,
        "|#quad.pts.| x |#nodes| shape function matrix");

    m.def(
        "shape_function_gradients",
        [](Mesh const& M, int qOrder) {
            auto constexpr kMaxQuadratureOrder = 8;
            MatrixX GN;
            M.ApplyWithQuadrature<kMaxQuadratureOrder>(
                [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
                    GN = pbat::fem::ShapeFunctionGradients<QuadratureOrder>(*mesh);
                },
                qOrder);
            return GN;
        },
        pyb::arg("mesh"),
        pyb::arg("quadrature_order") = 1,
        "|#element nodes| x |#dims * #quad.pts. * #elements| matrix of shape functions at each "
        "element quadrature point");

    m.def(
        "shape_function_gradients_at",
        [](Mesh const& M,
           Eigen::Ref<IndexVectorX const> const& E,
           Eigen::Ref<MatrixX const> const& Xg,
           bool bXgInReferenceSpace) {
            MatrixX GN;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                GN = pbat::fem::ShapeFunctionGradientsAt(*mesh, E, Xg, bXgInReferenceSpace);
            });
            return GN;
        },
        pyb::arg("mesh"),
        pyb::arg("E"),
        pyb::arg("Xg"),
        pyb::arg("in_reference_space"),
        "|#element nodes| x |E.shape[0] * mesh.dims| nodal shape function gradients at evaluation "
        "points Xg");
}

} // namespace fem
} // namespace py
} // namespace pbat
