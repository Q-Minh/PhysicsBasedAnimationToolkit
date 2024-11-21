#include "Restriction.h"

#include <pbat/sim/vbd/Restriction.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindRestriction([[maybe_unused]] pybind11::module& m)
{
    // namespace pyb = pybind11;
    // using pbat::sim::vbd::Restriction;
    // pyb::class_<Restriction>(m, "Restriction").def(pyb::init<>());
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat
