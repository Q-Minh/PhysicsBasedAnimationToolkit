#include "Integration.h"

#include "Bdf.h"

namespace pbat::py::sim::integration {

void Bind(nanobind::module_& m)
{
    BindBdf(m);
}

} // namespace pbat::py::sim::integration