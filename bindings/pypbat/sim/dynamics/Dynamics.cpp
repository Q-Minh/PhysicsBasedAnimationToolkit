#include "Dynamics.h"

#include "FemElastoDynamics.h"

namespace pbat::py::sim::dynamics {

void Bind(nanobind::module_& m)
{
    BindFemElastoDynamics(m);
}

} // namespace pbat::py::sim::dynamics