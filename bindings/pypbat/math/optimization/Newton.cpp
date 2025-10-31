#include "Newton.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/variant.h>
#include <pbat/Aliases.h>
#include <pbat/math/optimization/LineSearch.h>
#include <pbat/math/optimization/Newton.h>

namespace pbat::py::math::optimization {

void BindNewton(nanobind::module_& m)
{
    namespace nb = nanobind;

    using ScalarType        = Scalar;
    using NewtonType        = pbat::math::optimization::Newton<ScalarType>;
    using LineSearchVariant = typename NewtonType::LineSearchType;
    nb::class_<NewtonType>(m, "Newton")
        .def(
            nb::init<int, ScalarType, Index, LineSearchVariant>(),
            nb::arg("n_max_iters") = 10,
            nb::arg("gtol")        = ScalarType(1e-4),
            nb::arg("n")           = 0,
            nb::arg("line_search") = pbat::math::optimization::BackTrackingLineSearch<ScalarType>{},
            "Construct a Newton optimizer with a backtracking line search.\n\n"
            "Args:\n"
            "    n_max_iters (int, optional): Maximum Newton iterations. Defaults to 10.\n"
            "    gtol (float, optional): Gradient norm tolerance. Defaults to 1e-4.\n"
            "    n (int, optional): Degrees of freedom (allocates buffers). Defaults to 0.\n"
            "    line_search (None | BackTrackingLineSearch): Line search instance.")
        .def_rw("n_max_iters", &NewtonType::nMaxIters, "Maximum Newton iterations")
        .def_rw("gtol2", &NewtonType::gtol2, "Squared gradient norm tolerance")
        .def_ro("dxk", &NewtonType::dxk, "Step direction (internal work vector)")
        .def_ro("gk", &NewtonType::gk, "Gradient at current iterate (internal work vector)")
        .def_ro("fk", &NewtonType::fk, "Objective value at current iterate")
        .def_ro("gknorm2", &NewtonType::gknorm2, "Squared gradient norm at current iterate")
        .def_ro("k", &NewtonType::k, "Current iteration")
        .def_rw("line_search", &NewtonType::lineSearch, "Line search object")
        // Bindings for these methods are a bit complicated, so don't bother for now.
        /*.def(
            "prepare_next_iteration",
            [](NewtonType& self,
               std::function<ScalarType(nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const>)>
                   fPrepareDerivatives,
               std::function<Eigen::Vector<ScalarType, Eigen::Dynamic>(
                   nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const>)> g,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> const& xk) {
                auto const gAdapt =
                    [&](nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> const& x,
                        Eigen::Vector<ScalarType, Eigen::Dynamic>& gk) {
                        gk = g(x);
                    };
                self.PrepareNextIteration(fPrepareDerivatives, gAdapt, xk);
            },
            nb::arg("prepare_derivatives"),
            nb::arg("g"),
            nb::arg("xk"),
            "Call derivative preparation and compute gradient at xk. Updates fk, gk, gknorm2.\n\n"
            "Args:\n"
            "    prepare_derivatives (Callable[[numpy.ndarray], float]): Precompute shared terms "
            "and return f(xk).\n"
            "    g (Callable[[numpy.ndarray], numpy.ndarray]): Gradient function returning "
            "vector.\n"
            "    xk (numpy.ndarray): Current iterate.")
        .def(
            "iterate",
            [](NewtonType& self,
               std::function<ScalarType(Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> f,
               std::function<Eigen::Vector<ScalarType, Eigen::Dynamic>(
                   Eigen::Vector<ScalarType, Eigen::Dynamic> const&,
                   Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> hinv,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic>> xk) {
                auto hinv_adapt = [&](Eigen::Vector<ScalarType, Eigen::Dynamic> const& x,
                                      Eigen::Vector<ScalarType, Eigen::Dynamic> const& g,
                                      Eigen::Vector<ScalarType, Eigen::Dynamic>& dx) {
                    dx = hinv(x, g);
                };
                return self.Iterate(f, hinv_adapt, xk);
            },
            nb::arg("f"),
            nb::arg("hinv"),
            nb::arg("xk"),
            "Perform one Newton iteration using Python callbacks. Returns True if a step was "
            "taken.")
        .def(
            "solve",
            [](NewtonType& self,
               std::function<ScalarType(Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> fprepare,
               std::function<ScalarType(Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> f,
               std::function<Eigen::Vector<ScalarType, Eigen::Dynamic>(
                   Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> g,
               std::function<Eigen::Vector<ScalarType, Eigen::Dynamic>(
                   Eigen::Vector<ScalarType, Eigen::Dynamic> const&,
                   Eigen::Vector<ScalarType, Eigen::Dynamic> const&)> hinv,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic>> xk) {
                auto g_adapt = [&](Eigen::Vector<ScalarType, Eigen::Dynamic> const& x,
                                   Eigen::Vector<ScalarType, Eigen::Dynamic>& gout) {
                    gout = g(x);
                };
                auto hinv_adapt = [&](Eigen::Vector<ScalarType, Eigen::Dynamic> const& x,
                                      Eigen::Vector<ScalarType, Eigen::Dynamic> const& gin,
                                      Eigen::Vector<ScalarType, Eigen::Dynamic>& dx) {
                    dx = hinv(x, gin);
                };
                return self.Solve(fprepare, f, g_adapt, hinv_adapt, xk);
            },
            nb::arg("f_prepare_derivatives"),
            nb::arg("f"),
            nb::arg("g"),
            nb::arg("hinv"),
            nb::arg("xk"),
            "Run Newton's method to convergence (or max iters) using Python callbacks.\n\n"
            "Args:\n"
            "    f_prepare_derivatives (Callable[[numpy.ndarray], float]): Precompute shared terms "
            "and return f(xk).\n"
            "    f (Callable[[numpy.ndarray], float]): Objective function used for line search.\n"
            "    g (Callable[[numpy.ndarray], numpy.ndarray]): Gradient function returning "
            "vector.\n"
            "    hinv (Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]): Hessian-inverse "
            "product returning dx given x and g.\n"
            "    xk (numpy.ndarray): Current iterate (updated in-place).\n"
            "Returns:\n"
            "    bool: True if converged (stationarity), False otherwise.")*/
        ;
}

} // namespace pbat::py::math::optimization
