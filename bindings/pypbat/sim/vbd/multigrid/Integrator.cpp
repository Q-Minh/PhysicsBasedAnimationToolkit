#include "Integrator.h"

#include <pbat/sim/vbd/multigrid/Hierarchy.h>
#include <pbat/sim/vbd/multigrid/Integrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindIntegrator(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::vbd::multigrid::Hierarchy;
    using pbat::sim::vbd::multigrid::Integrator;
    nb::class_<Integrator>(m, "Integrator")
        .def(nb::init<>())
        .def(
            "step",
            &Integrator::Step,
            nb::arg("dt"),
            nb::arg("substeps"),
            nb::arg("hierarchy"));
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat