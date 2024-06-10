#include "MassMatrix.h"

#include "For.h"
#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/MassMatrix.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

void BindMassMatrix(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kDimsMax            = 3;
        auto constexpr kQuadratureOrderMax = 6;
        pbat::common::ForRange<1, kDimsMax + 1>([&]<auto kDims>() {
            pbat::common::ForRange<1, kQuadratureOrderMax + 1>([&]<auto kQuadratureOrder>() {
                using MassMatrixType = pbat::fem::MassMatrix<MeshType, kDims, kQuadratureOrder>;
                std::string const className =
                    "MassMatrix_Dims_" + std::to_string(kDims) + "_QuadratureOrder_" +
                    std::to_string(kQuadratureOrder) + "_" + MeshTypeName<MeshType>();
                pyb::class_<MassMatrixType>(m, className.data())
                    .def(
                        pyb::init([](MeshType const& mesh, Eigen::Ref<MatrixX const> const& detJe) {
                            return MassMatrixType(mesh, detJe);
                        }),
                        pyb::arg("mesh"),
                        pyb::arg("detJe"))
                    .def(
                        pyb::init([](MeshType const& mesh,
                                     Eigen::Ref<MatrixX const> const& detJe,
                                     Scalar rho) { return MassMatrixType(mesh, detJe, rho); }),
                        pyb::arg("mesh"),
                        pyb::arg("detJe"),
                        pyb::arg("rho"))
                    .def(
                        pyb::init(
                            [](MeshType const& mesh,
                               Eigen::Ref<MatrixX const> const& detJe,
                               VectorX const& rhoe) { return MassMatrixType(mesh, detJe, rhoe); }),
                        pyb::arg("mesh"),
                        pyb::arg("detJe"),
                        pyb::arg("rhoe"))
                    .def_property_readonly_static(
                        "dims",
                        [](pyb::object /*self*/) { return MassMatrixType::kDims; })
                    .def_property_readonly_static(
                        "order",
                        [](pyb::object /*self*/) { return MassMatrixType::kOrder; })
                    .def_property_readonly_static(
                        "quadrature_order",
                        [](pyb::object /*self*/) { return MassMatrixType::kQuadratureOrder; })
                    .def_readonly("Me", &MassMatrixType::Me)
                    .def("rows", &MassMatrixType::OutputDimensions)
                    .def("cols", &MassMatrixType::InputDimensions)
                    .def("to_matrix", &MassMatrixType::ToMatrix)
                    .def(
                        "compute_element_mass_matrices",
                        [](MassMatrixType& M, VectorX const& rhoe) {
                            M.ComputeElementMassMatrices(rhoe);
                        },
                        pyb::arg("rhoe"));
            });
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
