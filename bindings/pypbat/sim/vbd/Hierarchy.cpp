#include "Hierarchy.h"

#include <pbat/sim/vbd/Hierarchy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindHierarchy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::Hierarchy;
    using pbat::sim::vbd::Level;
    using pbat::sim::vbd::Smoother;
    using Transition = pbat::sim::vbd::Hierarchy::Transition;

    pyb::class_<Hierarchy>(m, "Hierarchy")
        .def(
            pyb::init([](Data& data,
                         std::vector<Level>& levels,
                         std::vector<MatrixX>& Ng,
                         std::vector<Transition>& transitions,
                         std::vector<Smoother>& smoothers) {
                return Hierarchy(
                    std::move(data),
                    std::move(levels),
                    std::move(Ng),
                    std::move(transitions),
                    std::move(smoothers));
            }),
            pyb::arg("data"),
            pyb::arg("levels"),
            pyb::arg("Ng"),
            pyb::arg("transitions"),
            pyb::arg("smoothers"))
        .def_readwrite("root", &Hierarchy::root)
        .def_readwrite("levels", &Hierarchy::levels)
        .def_readwrite("Ng", &Hierarchy::Ng)
        .def_readwrite("transitions", &Hierarchy::transitions)
        .def_readwrite("smoothers", &Hierarchy::smoothers);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat