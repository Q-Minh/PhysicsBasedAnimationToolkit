#include "Newton.h"

namespace pbat::py::sim::algorithm::newton {

void Bind(nanobind::module_& m)
{
    [[maybe_unused]] nanobind::module_ mnewton =
        m.def_submodule("newton", "Newton simulation algorithms.");
}

} // namespace pbat::py::sim::algorithm::newton