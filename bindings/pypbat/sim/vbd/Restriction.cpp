#include "Restriction.h"

#include <pbat/sim/vbd/Restriction.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindRestriction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Restriction;
    pyb::class_<Restriction>(m, "Restriction")
        .def(pyb::init<>())
        .def(
            "set_target_shape",
            [](Restriction& R, Eigen::Ref<MatrixX const> const& xtildeg) {
                R.SetTargetShape(xtildeg);
            },
            pyb::arg("xtildeg"))
        .def(
            "apply",
            [](Restriction& R, Eigen::Ref<MatrixX const> const& x, Index iterations) {
                MatrixX xcpy = x;
                R.Apply(xcpy, iterations);
                return xcpy;
            },
            pyb::arg("x"),
            pyb::arg("iters") = 20)
        .def_readwrite("E", &Restriction::E)
        .def_readwrite("eg", &Restriction::eg)
        .def_readwrite("Ng", &Restriction::Ng)
        .def_readwrite("GNeg", &Restriction::GNeg)
        .def_readwrite("wg", &Restriction::wg)
        .def_readwrite("rhog", &Restriction::rhog)
        .def_readwrite("mug", &Restriction::mug)
        .def_readwrite("lambdag", &Restriction::lambdag)
        .def_readwrite("m", &Restriction::m)
        .def_readwrite("gtilde", &Restriction::gtilde)
        .def_readwrite("Psitildeg", &Restriction::Psitildeg)
        .def_readwrite("Kpsi", &Restriction::Kpsi)
        .def_readwrite("GVGp", &Restriction::GVGp)
        .def_readwrite("GVGg", &Restriction::GVGg)
        .def_readwrite("GVGe", &Restriction::GVGe)
        .def_readwrite("GVGilocal", &Restriction::GVGilocal)
        .def_readwrite("partitions", &Restriction::partitions);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat