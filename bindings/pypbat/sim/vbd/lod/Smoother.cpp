#include "Smoother.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/lod/Level.h>
#include <pbat/sim/vbd/lod/Smoother.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindSmoother(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::lod::Level;
    using pbat::sim::vbd::lod::Smoother;
    pyb::class_<Smoother>(m, "Smoother")
        .def(pyb::init<>())
        .def(
            "apply",
            [](Smoother& smoother, Index iters, Scalar dt, Level& l) {
                smoother.Apply(iters, dt, l);
            },
            pyb::arg("iters"),
            pyb::arg("dt"),
            pyb::arg("level"))
        .def(
            "apply",
            [](Smoother& smoother, Index iters, Scalar dt, Data& data) {
                smoother.Apply(iters, dt, data);
            },
            pyb::arg("iters"),
            pyb::arg("dt"),
            pyb::arg("root"));
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat
