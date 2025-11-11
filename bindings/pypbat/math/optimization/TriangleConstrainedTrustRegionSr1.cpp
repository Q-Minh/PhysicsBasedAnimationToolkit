#include "TriangleConstrainedTrustRegionSr1.h"

#include <Eigen/Core>
#include <functional>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <optional>
#include <pbat/Aliases.h>
#include <pbat/math/linalg/mini/Eigen.h>
#include <pbat/math/linalg/mini/Matrix.h>
#include <pbat/math/optimization/TriangleConstrainedTrustRegionSr1.h>
#include <tuple>

namespace pbat::py::math::optimization {

void BindTriangleConstrainedTrustRegionSr1(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using ParamsType =
        pbat::math::optimization::TriangleConstrainedTrustRegionSr1Params<ScalarType>;

    nb::class_<ParamsType>(m, "TriangleConstrainedTrustRegionSr1Params")
        .def(nb::init<>(), "Default construct with uninitialized numeric fields.")
        .def_rw("R0", &ParamsType::R0, "Initial trust-region radius")
        .def_rw("eta", &ParamsType::eta, "Trust-region minimal energy reduction ratio")
        .def_rw("trlo", &ParamsType::trlo, "Largest ratio under which to shrink the trust region")
        .def_rw("trhi", &ParamsType::trhi, "Smallest ratio over which to grow the trust region")
        .def_rw(
            "trbound",
            &ParamsType::trbound,
            "Step size multiple threshold (0 < trbound <= 1) to allow growth")
        .def_rw("trgrow", &ParamsType::trgrow, "Trust-region growth factor")
        .def_rw("trshrink", &ParamsType::trshrink, "Trust-region shrink factor")
        .def_rw("sigmaB", &ParamsType::sigmaB, "Initial Hessian approximation scaling")
        .def_rw(
            "deltaf",
            &ParamsType::deltaf,
            "Numerical offset to avoid division by zero in objective function reduction ratio "
            "computation")
        .def_rw(
            "deltas",
            &ParamsType::deltas,
            "Numerical offset to avoid division by zero in step truncation")
        .def_rw("n_max_iters", &ParamsType::nMaxIters, "Maximum number of iterations")
        .def_rw("gzero", &ParamsType::gzero, "Gradient norm convergence tolerance")
        .def_ro("k", &ParamsType::k, "Iteration")
        .def_ro("Rk", &ParamsType::Rk, "Current trust-region radius")
        .def_ro("fk", &ParamsType::fk, "Current objective value")
        .def_prop_ro(
            "gk",
            [](ParamsType const& self) { return Eigen::Vector<Scalar, 2>{self.gk(0), self.gk(1)}; },
            "Current gradient (2-vector)")
        .def_prop_ro(
            "Bk",
            [](ParamsType const& self) {
                Eigen::Matrix<Scalar, 2, 2> Bk;
                Bk << self.Bk(0, 0), self.Bk(0, 1), self.Bk(1, 0), self.Bk(1, 1);
                return Bk;
            },
            "Current Hessian approximation (2x2)");

    m.def(
        "triangle_constrained_trust_region_sr1",
        [](std::function<ScalarType(Eigen::Vector<ScalarType, 2> const&)> f,
           std::function<Eigen::Vector<ScalarType, 2>(Eigen::Vector<ScalarType, 2> const&)> gradf,
           Eigen::Vector<ScalarType, 2>& xk,
           ParamsType& params,
           std::optional<std::function<
               bool(Eigen::Vector<ScalarType, 2> const&, ScalarType, ScalarType, bool)>>
               fCheckConvergence) {
            using namespace pbat::math::linalg::mini;
            auto const fwrap = [&](SVector<ScalarType, 2> const& x) {
                return f(ToEigen(x));
            };
            auto const gwrap = [&](SVector<ScalarType, 2> const& x) {
                return FromEigen(gradf(ToEigen(x)));
            };
            auto const cwrap = [&](SVector<ScalarType, 2> const& x,
                                   ScalarType ared,
                                   ScalarType pred,
                                   bool stepAccepted) {
                return fCheckConvergence.value()(ToEigen(x), ared, pred, stepAccepted);
            };
            SVector<ScalarType, 2> xk_wrap = FromEigen(xk);
            bool converged;
            if (fCheckConvergence)
            {
                converged = pbat::math::optimization::TriangleConstrainedTrustRegionSr1(
                    fwrap,
                    gwrap,
                    cwrap,
                    xk_wrap,
                    params);
            }
            else
            {
                converged = pbat::math::optimization::TriangleConstrainedTrustRegionSr1(
                    fwrap,
                    gwrap,
                    xk_wrap,
                    params);
            }
            Eigen::Vector<ScalarType, 2> xstar = ToEigen(xk_wrap);
            return std::make_tuple(xstar, converged);
        },
        nb::arg("f"),
        nb::arg("gradf"),
        nb::arg("xk"),
        nb::arg("params"),
        nb::arg("check_convergence").none(),
        "Solve a triangle-constrained trust-region SR1 optimization problem with a custom "
        "convergence callback.\n\n"
        "Args:\n"
        "    f (Callable[[numpy.ndarray], float]): Objective function taking a 2-vector `x`.\n"
        "    gradf (Callable[[numpy.ndarray], numpy.ndarray]): Gradient function taking a 2-vector "
        "`x` and returning 2-vector.\n"
        "    xk (numpy.ndarray): Initial iterate (updated in-place).\n"
        "    params (TriangleConstrainedTrustRegionSr1Params): Parameter struct (read/write).\n"
        "    check_convergence (Callable[[numpy.ndarray, bool], bool]): Callback receiving current "
        "`x` and a bool `step_accepted`; returns True to stop.\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, bool]: The solution and True if converged, False otherwise.");
}

} // namespace pbat::py::math::optimization
