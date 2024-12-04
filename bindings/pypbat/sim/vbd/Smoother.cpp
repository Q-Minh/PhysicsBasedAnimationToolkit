#include "Smoother.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Level.h>
#include <pbat/sim/vbd/Smoother.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindSmoother(pybind11::module& m)
{
    namespace pyb = pybind11;

    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::Level;
    using pbat::sim::vbd::Smoother;

    pyb::class_<Smoother>(m, "Smoother")
        .def(pyb::init([](Index iters) { return Smoother{iters}; }), pyb::arg("iters") = 10)
        .def(
            "apply",
            [](Smoother& S, Scalar dt, Level& L) { S.Apply(dt, L); },
            pyb::arg("dt"),
            pyb::arg("level"),
            "Smooth level's solution.")
        .def(
            "apply",
            [](Smoother& S, Scalar dt, Scalar rho, Data& data) { S.Apply(dt, rho, data); },
            pyb::arg("dt"),
            pyb::arg("rho"),
            pyb::arg("data"),
            "Smooth root's solution.")
        .def_readwrite("iters", &Smoother::iterations);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat