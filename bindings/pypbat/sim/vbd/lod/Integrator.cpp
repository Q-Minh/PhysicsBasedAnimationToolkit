#include "Integrator.h"

#include <pbat/sim/vbd/lod/Hierarchy.h>
#include <pbat/sim/vbd/lod/Integrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindIntegrator(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::lod::Hierarchy;
    using pbat::sim::vbd::lod::Integrator;
    pyb::class_<Integrator>(m, "Integrator")
        .def(pyb::init<>())
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt"),
            pyb::arg("substeps"),
            pyb::arg("hierarchy"));
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat