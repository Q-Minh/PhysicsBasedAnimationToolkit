#include "Sim.h"

#include "contact/Contact.h"
#include "dynamics/Dynamics.h"
#include "integration/Integration.h"
#include "vbd/Vbd.h"
#include "xpbd/Xpbd.h"

namespace pbat {
namespace py {
namespace sim {

void Bind(nanobind::module_& m)
{
    namespace nb = nanobind;

    auto mcontact = m.def_submodule("contact");
    contact::Bind(mcontact);
    auto mintegration = m.def_submodule("integration");
    integration::Bind(mintegration);
    auto mdynamics = m.def_submodule("dynamics");
    dynamics::Bind(mdynamics);
    auto mxpbd = m.def_submodule("xpbd");
    xpbd::Bind(mxpbd);
    auto mvbd = m.def_submodule("vbd");
    vbd::Bind(mvbd);
}

} // namespace sim
} // namespace py
} // namespace pbat