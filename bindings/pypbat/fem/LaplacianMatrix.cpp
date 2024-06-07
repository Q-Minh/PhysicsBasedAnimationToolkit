#include "LaplacianMatrix.h"

#include "For.h"
#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/LaplacianMatrix.h>
#include <pybind11/eigen.h>
#include <utility>

namespace pbat {
namespace py {
namespace fem {

void BindLaplacianMatrix(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kQuadratureOrderMax = 6;
        pbat::common::ForRange<1, kQuadratureOrderMax + 1>([&]<auto QuadratureOrder>() {
            using LaplacianMatrixType =
                pbat::fem::SymmetricLaplacianMatrix<MeshType, QuadratureOrder>;
            std::string const className = "SymmetricLaplacianMatrix_QuadratureOrder_" +
                                          std::to_string(QuadratureOrder) + "_" +
                                          MeshTypeName<MeshType>();
            pyb::class_<LaplacianMatrixType>(m, className.data())
                .def(
                    pyb::init([](MeshType const& mesh,
                                 Eigen::Ref<MatrixX const> const& detJe,
                                 Eigen::Ref<MatrixX const> const& GNe) {
                        return LaplacianMatrixType(mesh, detJe, GNe);
                    }),
                    pyb::arg("mesh"),
                    pyb::arg("detJe"),
                    pyb::arg("GNe"))
                .def_property_readonly_static(
                    "dims",
                    [](pyb::object /*self*/) { return LaplacianMatrixType::kDims; })
                .def_property_readonly_static(
                    "order",
                    [](pyb::object /*self*/) { return LaplacianMatrixType::kOrder; })
                .def_property_readonly_static(
                    "quadrature_order",
                    [](pyb::object /*self*/) { return LaplacianMatrixType::kQuadratureOrder; })
                .def("to_matrix", &LaplacianMatrixType::ToMatrix)
                .def_property_readonly(
                    "shape",
                    [](LaplacianMatrixType const& L) {
                        return std::make_pair(L.OutputDimensions(), L.InputDimensions());
                    })
                .def_readonly("deltae", &LaplacianMatrixType::deltaE);
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat