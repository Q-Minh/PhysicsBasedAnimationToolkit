#include "PD.h"

#include "ValanisLandelQuasiNewton.h"

namespace pbat::py::sim::algorithm::pd {

void Bind(nanobind::module_& m)
{
    [[maybe_unused]] nanobind::module_ mpd =
        m.def_submodule("pd", "Projective dynamics simulation algorithms.");
    BindValanisLandelQuasiNewton(mpd);
}

} // namespace pbat::py::sim::algorithm::pd