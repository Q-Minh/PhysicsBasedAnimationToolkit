#include "HyperReduction.h"

#include <pbat/sim/vbd/multigrid/Hierarchy.h>
#include <pbat/sim/vbd/multigrid/HyperReduction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHyperReduction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::multigrid::Hierarchy;
    using pbat::sim::vbd::multigrid::HyperReduction;

    pyb::class_<HyperReduction>(m, "HyperReduction")
        .def(
            pyb::init<Hierarchy const&, Index>(),
            pyb::arg("hierarchy"),
            pyb::arg("n_target_active_elements") = Index(-1));
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat