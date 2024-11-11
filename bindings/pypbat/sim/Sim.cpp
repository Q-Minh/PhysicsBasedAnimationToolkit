#include "Sim.h"

#include "vbd/Vbd.h"
#include "xpbd/Xpbd.h"

namespace pbat {
namespace py {
namespace sim {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;

    auto mxpbd = m.def_submodule("xpbd");
    xpbd::Bind(mxpbd);
    auto mvbd = m.def_submodule("vbd");
    vbd::Bind(mvbd);
}

} // namespace sim
} // namespace py
} // namespace pbat