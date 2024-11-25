#include "Smoother.h"

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

    using pbat::sim::vbd::Smoother;

    pyb::class_<Smoother>(m, "Smoother")
        .def(pyb::init<Index>(), pyb::arg("iters") = 10)
        .def("apply", &Smoother::Apply, pyb::arg("level"), "Smooth level's solution.")
        .def_readwrite("iters", &Smoother::iterations);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat