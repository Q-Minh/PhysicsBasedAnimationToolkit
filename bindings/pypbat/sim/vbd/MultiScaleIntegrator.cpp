#include "MultiScaleIntegrator.h"

#include <pbat/sim/vbd/Hierarchy.h>
#include <pbat/sim/vbd/MultiScaleIntegrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindMultiScaleIntegrator(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Hierarchy;
    using pbat::sim::vbd::MultiScaleIntegrator;

    pyb::class_<MultiScaleIntegrator>(m, "MultiScaleIntegrator")
        .def(pyb::init<>())
        .def(
            "step",
            &MultiScaleIntegrator::Step,
            pyb::arg("dt"),
            pyb::arg("substeps"),
            pyb::arg("hierarchy"));
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat