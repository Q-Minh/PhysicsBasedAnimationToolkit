#include "Jacobian.h"

#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Jacobian.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindJacobian(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "jacobian_determinants",
        [](Mesh const& M, int qOrder) {
            auto constexpr kMaxQuadratureOrder = 8;
            MatrixX detJe;
            M.ApplyWithQuadrature<kMaxQuadratureOrder>(
                [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
                    detJe = pbat::fem::DeterminantOfJacobian<QuadratureOrder>(*mesh);
                },
                qOrder);
            return detJe;
        },
        pyb::arg("mesh"),
        pyb::arg("quadrature_order") = 1);

    m.def(
        "inner_product_weights",
        [](Mesh const& M, int qOrder) {
            auto constexpr kMaxQuadratureOrder = 8;
            MatrixX I;
            M.ApplyWithQuadrature<kMaxQuadratureOrder>(
                [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
                    I = pbat::fem::InnerProductWeights<QuadratureOrder>(*mesh);
                },
                qOrder);
            return I;
        },
        pyb::arg("mesh"),
        pyb::arg("quadrature_order") = 1);

    m.def(
        "inner_product_weights",
        [](Mesh const& M, Eigen::Ref<MatrixX const> const& detJe, int qOrder) {
            auto constexpr kMaxQuadratureOrder = 8;
            MatrixX I;
            M.ApplyWithQuadrature<kMaxQuadratureOrder>(
                [&]<class MeshType, auto QuadratureOrder>(MeshType* mesh) {
                    I = pbat::fem::InnerProductWeights<QuadratureOrder>(*mesh, detJe);
                },
                qOrder);
            return I;
        },
        pyb::arg("mesh"),
        pyb::arg("detJe"),
        pyb::arg("quadrature_order") = 1);

    m.def(
        "inner_product_weights",
        [](Mesh const& M,
           Eigen::Ref<IndexVectorX const> const& E,
           Eigen::Ref<MatrixX const> const& X,
           int maxIterations,
           Scalar eps) {
            MatrixX XR;
            M.Apply([&]<class MeshType>(MeshType* mesh) {
                XR = pbat::fem::ReferencePositions(*mesh, E, X, maxIterations, eps);
            });
            return XR;
        },
        pyb::arg("mesh"),
        pyb::arg("E"),
        pyb::arg("X"),
        pyb::arg("max_iters") = 5,
        pyb::arg("eps")       = 1e-10);
}

} // namespace fem
} // namespace py
} // namespace pbat
