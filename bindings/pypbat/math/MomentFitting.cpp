#include "MomentFitting.h"

#include <exception>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/MomentFitting.h>

namespace pbat {
namespace py {
namespace math {

void BindMomentFitting(nanobind::module_& m)
{
    namespace nb = nanobind;
    m.def(
        "transfer_quadrature",
        [](Eigen::Ref<IndexVectorX const> const& S1,
           Eigen::Ref<MatrixX const> const& X1,
           Eigen::Ref<IndexVectorX const> const& S2,
           Eigen::Ref<MatrixX const> const& X2,
           Eigen::Ref<VectorX const> const& w2,
           Index nSimplices,
           Index order,
           bool bEvaluateError,
           Index maxIterations,
           Scalar precision) {
            VectorX w1, err;
            common::ForRange<1, 5>([&]<auto Order>() {
                if (order == Order)
                {
                    std::tie(w1, err) = pbat::math::TransferQuadrature<Order>(
                        S1,
                        X1,
                        S2,
                        X2,
                        w2,
                        nSimplices,
                        bEvaluateError,
                        maxIterations,
                        precision);
                }
            });
            if (w1.size() == 0)
                throw std::invalid_argument("transfer_quadrature only accepts 1 <= order <= 4.");
            return std::make_pair(w1, err);
        },
        nb::arg("S1"),
        nb::arg("X1"),
        nb::arg("S2"),
        nb::arg("X2"),
        nb::arg("w2"),
        nb::arg("nSimplices") = -1,
        nb::arg("order")      = 1,
        nb::arg("with_error") = false,
        nb::arg("max_iters")  = 20,
        nb::arg("precision")  = std::numeric_limits<Scalar>::epsilon(),
        "Obtain weights w1 by transferring an existing quadrature rule (X2,w2) "
        "defined on a domain composed of simplices onto a new quadrature rule "
        "(X1,w1) defined on the same domain, given fixed quadrature points X1. "
        "S1 is a |X1.shape[1]| array of simplex indices associated with the "
        "corresponding quadrature point in columns (i.e. the quadrature points) "
        "of X1. S2 is the same for columns of X2. w2 are the quadrature weights "
        "associated with X2. If with_error==True, the polynomial integration error "
        "is computed and returned.");

    m.def(
        "reference_moment_fitting_systems",
        [](Eigen::Ref<IndexVectorX const> const& S1,
           Eigen::Ref<MatrixX const> const& X1,
           Eigen::Ref<IndexVectorX const> const& S2,
           Eigen::Ref<MatrixX const> const& X2,
           Eigen::Ref<VectorX const> const& w2,
           Index nSimplices,
           Index order) {
            MatrixX M, B;
            IndexVectorX P;
            common::ForRange<1, 5>([&]<auto Order>() {
                if (order == Order)
                {
                    std::tie(M, B, P) = pbat::math::ReferenceMomentFittingSystems<Order>(
                        S1,
                        X1,
                        S2,
                        X2,
                        w2,
                        nSimplices);
                }
            });
            if (M.size() == 0)
                throw std::invalid_argument("transfer_quadrature only accepts 1 <= order <= 4.");
            return std::make_tuple(M, B, P);
        },
        nb::arg("S1"),
        nb::arg("X1"),
        nb::arg("S2"),
        nb::arg("X2"),
        nb::arg("w2"),
        nb::arg("nSimplices") = -1,
        nb::arg("order")      = 1,
        "Obtain a collection of reference moment fitting systems (M, B, P), where M[:, "
        "P[s]:P[s+1]] is the reference moment fitting matrix for simplex s, and b[:,s] is its "
        "corresponding right-hand side. X1, S1 are the |#dims|x|#quad.pts.| array of quadrature "
        "points and corresponding simplices, and X2,w2,S2 are the |#dims|x|#old quad.pts.| array "
        "of existing quadrature points and corresponding simplices, with weights w2. order "
        "specifies the polynomial order of integration that we wish to reproduce exactly.");

    m.def(
        "block_diagonalize_moment_fitting",
        [](Eigen::Ref<MatrixX const> const& M, Eigen::Ref<IndexVectorX const> const& P) {
            return pbat::math::BlockDiagonalReferenceMomentFittingSystem(M, P);
        },
        nb::arg("M"),
        nb::arg("P"),
        "Assemble the block diagonal row sparse matrix GM, such that GM @ w = B.flatten(order='F') "
        "contains all the reference moment fitting systems in (M,B,P).");
}

} // namespace math
} // namespace py
} // namespace pbat