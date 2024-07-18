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
        pyb::arg("quadrature_order") = 1);

    m.def(
        "shape_function_gradients_at",
        [](Mesh const& M,
           Eigen::Ref<IndexVectorX const> const& E,
           Eigen::Ref<MatrixX const> const& Xi) {
            MatrixX GN;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                GN = pbat::fem::ShapeFunctionGradientsAt(*mesh, E, Xi);
            });
            return GN;
        },
        pyb::arg("mesh"),
        pyb::arg("E"),
        pyb::arg("Xi"));

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
        pyb::arg("quadrature_order") = 1);

    m.def(
        "shape_functions_at",
        [](Mesh const& M, Eigen::Ref<MatrixX const> const& Xi) {
            MatrixX N;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                using ElementType = typename MeshType::ElementType;
                N                 = pbat::fem::ShapeFunctionsAt<ElementType>(Xi);
            });
            return N;
        },
        pyb::arg("mesh"),
        pyb::arg("Xi"));
}

} // namespace fem
} // namespace py
} // namespace pbat
