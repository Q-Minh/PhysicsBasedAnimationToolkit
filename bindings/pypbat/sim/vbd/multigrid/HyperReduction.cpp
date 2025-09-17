#include "HyperReduction.h"

#include <pbat/sim/vbd/multigrid/Hierarchy.h>
#include <pbat/sim/vbd/multigrid/HyperReduction.h>
#include <nanobind/eigen/dense.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHyperReduction(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::vbd::multigrid::Hierarchy;
    using pbat::sim::vbd::multigrid::HyperReduction;

    nb::class_<HyperReduction>(m, "HyperReduction")
        .def(
            nb::init<Hierarchy const&, Index>(),
            nb::arg("hierarchy"),
            nb::arg("n_target_active_elements") = Index(-1));
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat