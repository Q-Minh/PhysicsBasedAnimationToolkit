#include "LoadVector.h"

#include "For.h"
#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/LoadVector.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

void BindLoadVector(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kDimsMax            = 3;
        auto constexpr kQuadratureOrderMax = 3;
        pbat::common::ForRange<1, kDimsMax + 1>([&]<auto Dims>() {
            pbat::common::ForRange<1, kQuadratureOrderMax + 1>([&]<auto QuadratureOrder>() {
                using LoadVectorType = pbat::fem::LoadVector<MeshType, Dims, QuadratureOrder>;
                std::string const className =
                    "LoadVector_Dims_" + std::to_string(Dims) + "_QuadratureOrder_" +
                    std::to_string(QuadratureOrder) + "_" + MeshTypeName<MeshType>();
                pyb::class_<LoadVectorType>(m, className.data())
                    .def(
                        pyb::init([](MeshType const& mesh,
                                     Eigen::Ref<MatrixX const> const& detJe,
                                     Eigen::Ref<VectorX const> const& fe) {
                            return LoadVectorType(mesh, detJe, fe);
                        }),
                        pyb::arg("mesh"),
                        pyb::arg("detJe"),
                        pyb::arg("fe"))
                    .def_property_readonly_static(
                        "dims",
                        [](pyb::object /*self*/) { return LoadVectorType::kDims; })
                    .def_property_readonly_static(
                        "order",
                        [](pyb::object /*self*/) { return LoadVectorType::kOrder; })
                    .def_property_readonly_static(
                        "quadrature_order",
                        [](pyb::object /*self*/) { return LoadVectorType::kQuadratureOrder; })
                    .def_readonly("fe", &LoadVectorType::fe)
                    .def_property_readonly(
                        "shape",
                        [](LoadVectorType const& f) { return std::make_tuple(f.mesh.X.cols()); })
                    .def("to_vector", &LoadVectorType::ToVector)
                    .def(
                        "set_load",
                        [](LoadVectorType& f, Eigen::Ref<VectorX const> const& fe) {
                            f.SetLoad(fe);
                        },
                        pyb::arg("fe"))
                    .def_readonly("N", &LoadVectorType::N);
            });
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat