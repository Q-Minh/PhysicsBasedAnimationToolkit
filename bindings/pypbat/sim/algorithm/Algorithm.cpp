#include "Algorithm.h"

#include "newton/Newton.h"
#include "pd/PD.h"
#include "vbd/Vbd.h"

namespace pbat::py::sim::algorithm {

void Bind(nanobind::module_& m)
{
    auto mnewton = m.def_submodule("newton");
    newton::Bind(mnewton);
    auto mpd = m.def_submodule("pd");
    pd::Bind(mpd);
    auto mvbd = m.def_submodule("vbd");
    vbd::Bind(mvbd);
}

} // namespace pbat::py::sim::algorithm