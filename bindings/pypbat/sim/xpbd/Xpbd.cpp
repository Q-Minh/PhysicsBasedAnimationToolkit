#include "Xpbd.h"

#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <utility>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void Bind([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat