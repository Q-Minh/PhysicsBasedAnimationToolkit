#include "LineSearch.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <pbat/Aliases.h>
#include <pbat/math/optimization/LineSearch.h>

namespace pbat::py::math::optimization {

void BindLineSearch(nanobind::module_& m)
{
    namespace nb = nanobind;

    using ScalarType   = Scalar;
    using BackTracking = pbat::math::optimization::BackTrackingLineSearch<ScalarType>;
    nb::class_<BackTracking>(m, "BackTrackingLineSearch")
        .def(
            nb::init<int, ScalarType, ScalarType, ScalarType, Eigen::Index>(),
            nb::arg("n_max_iters") = 20,
            nb::arg("tau")         = ScalarType(0.5),
            nb::arg("c")           = ScalarType(1e-4),
            nb::arg("alpha")       = ScalarType(1),
            nb::arg("n")           = 0,
            "Construct a backtracking line search object.\n\n"
            "Args:\n"
            "    n_max_iters (int, optional): Maximum number of iterations. Defaults to 20.\n"
            "    tau (float, optional): Step reduction factor in (0,1). Defaults to 0.5.\n"
            "    c (float, optional): Armijo slope scale in (0,1). Defaults to 1e-4.\n"
            "    alpha (float, optional): Initial step size. Defaults to 1.\n"
            "    n (int, optional): Degrees of freedom (allocates internal buffers). Defaults to "
            "0.")
        .def_rw("n_max_iters", &BackTracking::nMaxIters, "Maximum iterations for line search")
        .def_rw("tau", &BackTracking::tau, "Step size decrease factor")
        .def_rw("c", &BackTracking::c, "Armijo slope scale")
        .def_rw("alpha", &BackTracking::alpha, "Initial step size")
        .def_ro("alphaj", &BackTracking::alphaj, "Current step size")
        .def_ro("fj", &BackTracking::fj, "Current objective value")
        .def_ro("xj", &BackTracking::xj, "Current candidate iterate")
        .def_ro("niters", &BackTracking::niters, "Current iteration count")
        // Bindings for this are a bit complicated, so don't bother for now.
        /*.def(
            "solve",
            [](BackTracking& self,
               std::function<ScalarType(nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const&>)>
                   f,
               ScalarType fk,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> gk,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> dx,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> xk) {
                return self.Solve(f, fk, gk, dx, xk);
            },
            nb::arg("f"),
            nb::arg("fk"),
            nb::arg("gk"),
            nb::arg("dx"),
            nb::arg("xk"),
            "Perform a backtracking line search using a Python objective callback.\n\n"
            "Args:\n"
            "    f (Callable[[numpy.ndarray], float]): Objective function f(x).\n"
            "    fk (float): Objective value at current iterate.\n"
            "    gk (numpy.ndarray): `n x 1` gradient at current iterate.\n"
            "    dx (numpy.ndarray): `n x 1` descent direction.\n"
            "    xk (numpy.ndarray): `n x 1` current iterate.\n"
            "Returns:\n"
            "    bool: True if Armijo condition is met within max iterations.")*/
        ;
}

} // namespace pbat::py::math::optimization
