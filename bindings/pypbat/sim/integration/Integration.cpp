#include "Integration.h"

#include "Bdf.h"

namespace pbat::py::sim::integration {

void Bind(pybind11::module& m)
{
    BindBdf(m);
}

} // namespace pbat::py::sim::integration