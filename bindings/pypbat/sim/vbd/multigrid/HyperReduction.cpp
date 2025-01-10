#include "HyperReduction.h"

#include <pbat/sim/vbd/Data.h>
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
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::HyperReduction;

    pyb::class_<HyperReduction>(m, "HyperReduction")
        .def(
            pyb::init<Data const&, Index>(),
            pyb::arg("data"),
            pyb::arg("n_target_active_elements") = Index(-1))
        .def_readwrite(
            "active_elements",
            &HyperReduction::bActiveE,
            "|#fine elements| boolean mask identifying active elements at this coarse level")
        .def_readwrite(
            "active_vertices",
            &HyperReduction::bActiveK,
            "|#fine vertices| boolean mask identifying active vertices at this coarse level")
        .def_readwrite(
            "wgE",
            &HyperReduction::wgE,
            "|#fine elements| quadrature weights at this coarse level")
        .def_readwrite(
            "mK",
            &HyperReduction::mK,
            "|#fine vertices| lumped nodal masses at this coarse level")
        .def_readwrite(
            "nTargetActiveElements",
            &HyperReduction::nTargetActiveElements,
            "Target number of active elements at this coarse level");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat