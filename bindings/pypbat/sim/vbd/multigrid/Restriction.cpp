#include "Restriction.h"

#include <pbat/sim/vbd/multigrid/Level.h>
#include <pbat/sim/vbd/multigrid/Quadrature.h>
#include <pbat/sim/vbd/multigrid/Restriction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindRestriction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::multigrid::CageQuadrature;
    using pbat::sim::vbd::multigrid::Restriction;
    pyb::class_<Restriction>(m, "Restriction")
        .def(
            pyb::init([](CageQuadrature const& CQ) { return Restriction(CQ); }),
            pyb::arg("cage_quadrature"))
        .def("apply", &Restriction::Apply, pyb::arg("iters"), pyb::arg("lf"), pyb::arg("lc"))
        .def(
            "do_apply",
            &Restriction::DoApply,
            pyb::arg("iters"),
            pyb::arg("xf"),
            pyb::arg("Ef"),
            pyb::arg("lc"))
        .def_readwrite("xfg", &Restriction::xfg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat