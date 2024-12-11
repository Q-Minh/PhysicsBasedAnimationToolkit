#include "Integrator.h"

#include <pbat/sim/vbd/multigrid/Hierarchy.h>
#include <pbat/sim/vbd/multigrid/Integrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindIntegrator(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::multigrid::Hierarchy;
    using pbat::sim::vbd::multigrid::Integrator;
    pyb::class_<Integrator>(m, "Integrator")
        .def(pyb::init<>())
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt"),
            pyb::arg("substeps"),
            pyb::arg("hierarchy"));
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat