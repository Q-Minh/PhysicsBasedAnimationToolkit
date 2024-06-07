#include "HyperElasticPotential.h"

#include "For.h"
#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

void BindHyperElasticPotential(pybind11::module& m)
{
    namespace pyb = pybind11;
    ForMeshTypes([&]<class MeshType>() {
        auto constexpr kQuadratureOrderMax = 2;
        auto constexpr kDimsMax            = 3;
        using ElementType                  = typename MeshType::ElementType;

        pbat::common::ForRange<1, kDimsMax + 1>([&]<auto Dims>() {
            pbat::common::ForTypes<
                pbat::physics::SaintVenantKirchhoffEnergy<Dims>,
                pbat::physics::StableNeoHookeanEnergy<Dims>>([&]<class HyperElasticEnergy>() {
                // Eigen cannot stack-allocate a large matrix, so cubic hexahedra are probably too
                // big.
                bool constexpr bIsHexahedron          = ElementType::Vertices.size() == 8;
                bool constexpr bIs3DimensionalProblem = Dims == 3;
                if constexpr (not(bIsHexahedron and bIs3DimensionalProblem and
                                  (ElementType::kOrder >= 3)))
                {
                    std::string const psiName = []() {
                        if constexpr (std::is_same_v<
                                          HyperElasticEnergy,
                                          pbat::physics::SaintVenantKirchhoffEnergy<Dims>>)
                            return "StVk";
                        if constexpr (std::is_same_v<
                                          HyperElasticEnergy,
                                          pbat::physics::StableNeoHookeanEnergy<Dims>>)
                            return "StableNeohookean";
                    }();

                    pbat::common::ForRange<1, kQuadratureOrderMax + 1>([&]<auto QuadratureOrder>() {
                        using ElasticPotentialType = pbat::fem::
                            HyperElasticPotential<MeshType, HyperElasticEnergy, QuadratureOrder>;
                        std::string const className =
                            "HyperElasticPotential_" + psiName + "_QuadratureOrder_" +
                            std::to_string(QuadratureOrder) + "_Dims_" + std::to_string(Dims) +
                            "_" + MeshTypeName<MeshType>();
                        pyb::class_<ElasticPotentialType>(m, className.data())
                            .def(
                                pyb::init([](MeshType const& mesh,
                                             Eigen::Ref<MatrixX const> const& detJe,
                                             Eigen::Ref<MatrixX const> const& GNe,
                                             Eigen::Ref<VectorX const> const& x,
                                             Scalar Y,
                                             Scalar nu) {
                                    return ElasticPotentialType(mesh, detJe, GNe, x, Y, nu);
                                }),
                                pyb::arg("mesh"),
                                pyb::arg("detJe"),
                                pyb::arg("GNe"),
                                pyb::arg("x"),
                                pyb::arg("Y"),
                                pyb::arg("nu"))
                            .def(
                                pyb::init([](MeshType const& mesh,
                                             Eigen::Ref<MatrixX const> const& detJe,
                                             Eigen::Ref<MatrixX const> const& GNe,
                                             Eigen::Ref<VectorX const> const& x,
                                             Eigen::Ref<VectorX const> const& Y,
                                             Eigen::Ref<VectorX const> const& nu) {
                                    return ElasticPotentialType(mesh, detJe, GNe, x, Y, nu);
                                }),
                                pyb::arg("mesh"),
                                pyb::arg("detJe"),
                                pyb::arg("GNe"),
                                pyb::arg("x"),
                                pyb::arg("Y"),
                                pyb::arg("nu"))
                            .def_property_readonly_static(
                                "dims",
                                [](pyb::object /*self*/) { return ElasticPotentialType::kDims; })
                            .def_property_readonly_static(
                                "order",
                                [](pyb::object /*self*/) { return ElasticPotentialType::kOrder; })
                            .def_property_readonly_static(
                                "quadrature_order",
                                [](pyb::object /*self*/) {
                                    return ElasticPotentialType::kQuadratureOrder;
                                })
                            .def(
                                "precompute_hessian_sparsity",
                                &ElasticPotentialType::PrecomputeHessianSparsity)
                            .def(
                                "compute_element_elasticity",
                                [](ElasticPotentialType& U, Eigen::Ref<VectorX const> const& x) {
                                    U.ComputeElementElasticity(x);
                                },
                                pyb::arg("x"))
                            .def("to_matrix", &ElasticPotentialType::ToMatrix)
                            .def("to_vector", &ElasticPotentialType::ToVector)
                            .def("eval", &ElasticPotentialType::Eval)
                            .def_property_readonly(
                                "shape",
                                [](ElasticPotentialType const& U) {
                                    return std::make_tuple(
                                        U.OutputDimensions(),
                                        U.InputDimensions());
                                })
                            .def_readwrite("mue", &ElasticPotentialType::mue)
                            .def_readwrite("lambdae", &ElasticPotentialType::lambdae)
                            .def_readonly("He", &ElasticPotentialType::He)
                            .def_readonly("Ge", &ElasticPotentialType::Ge)
                            .def_readonly("Ue", &ElasticPotentialType::Ue);
                    });
                }
            });
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat