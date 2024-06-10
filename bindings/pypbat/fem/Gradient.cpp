#include "Gradient.h"

#include "For.h"
#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Gradient.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

void BindGradient(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kQuadratureOrderMax = 5;
        pbat::common::ForRange<1, kQuadratureOrderMax + 1>([&]<auto QuadratureOrder>() {
            using GradientMatrixType    = pbat::fem::GalerkinGradient<MeshType, QuadratureOrder>;
            std::string const className = "GalerkinGradientMatrix_QuadratureOrder_" +
                                          std::to_string(QuadratureOrder) + "_" +
                                          MeshTypeName<MeshType>();
            pyb::class_<GradientMatrixType>(m, className.data())
                .def(
                    pyb::init([](MeshType const& mesh,
                                 Eigen::Ref<MatrixX const> const& detJe,
                                 Eigen::Ref<MatrixX const> const& GNe) {
                        return GradientMatrixType(mesh, detJe, GNe);
                    }),
                    pyb::arg("mesh"),
                    pyb::arg("detJe"),
                    pyb::arg("GNe"))
                .def_property_readonly_static(
                    "dims",
                    [](pyb::object /*self*/) { return GradientMatrixType::kDims; })
                .def_property_readonly_static(
                    "order",
                    [](pyb::object /*self*/) { return GradientMatrixType::kOrder; })
                .def_property_readonly_static(
                    "quadrature_order",
                    [](pyb::object /*self*/) { return GradientMatrixType::kQuadratureOrder; })
                .def("rows", &GradientMatrixType::OutputDimensions)
                .def("cols", &GradientMatrixType::InputDimensions)
                .def("to_matrix", &GradientMatrixType::ToMatrix)
                .def_readonly("Ge", &GradientMatrixType::Ge);
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
