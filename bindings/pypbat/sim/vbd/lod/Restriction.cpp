#include "Restriction.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/lod/Level.h>
#include <pbat/sim/vbd/lod/Quadrature.h>
#include <pbat/sim/vbd/lod/Restriction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindRestriction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::lod::CageQuadrature;
    using pbat::sim::vbd::lod::Level;
    using pbat::sim::vbd::lod::Restriction;
    pyb::class_<Restriction>(m, "Restriction")
        .def(
            pyb::init([](CageQuadrature const& CQ) { return Restriction(CQ); }),
            pyb::arg("cage_quadrature"))
        .def(
            "apply",
            [](Restriction& R, Index iters, Data const& lf, Level& lc) { R.Apply(iters, lf, lc); },
            pyb::arg("iters"),
            pyb::arg("lf"),
            pyb::arg("lc"))
        .def(
            "apply",
            [](Restriction& R, Index iters, Level const& lf, Level& lc) { R.Apply(iters, lf, lc); },
            pyb::arg("iters"),
            pyb::arg("lf"),
            pyb::arg("lc"))
        .def(
            "do_apply",
            &Restriction::DoApply,
            pyb::arg("iters"),
            pyb::arg("xf"),
            pyb::arg("Ef"),
            pyb::arg("lc"))
        .def_readwrite("xfg", &Restriction::xfg);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat