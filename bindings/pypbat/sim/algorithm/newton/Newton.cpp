#include "Newton.h"

#include "Core.h"

namespace pbat::py::sim::algorithm::newton {

void Bind(nanobind::module_& m)
{
    auto mnewton = m.def_submodule("newton", "Newton simulation algorithms.");
    BindCore(mnewton);
}

} // namespace pbat::py::sim::algorithm::newton