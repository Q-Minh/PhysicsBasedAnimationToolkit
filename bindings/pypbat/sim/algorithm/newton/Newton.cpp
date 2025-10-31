#include "Newton.h"

#include "Core.h"

namespace pbat::py::sim::algorithm::newton {

void Bind(nanobind::module_& m)
{
    BindCore(m);
}

} // namespace pbat::py::sim::algorithm::newton