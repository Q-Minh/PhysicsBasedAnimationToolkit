#include "Dynamics.h"

#include "FemElastoDynamics.h"

namespace pbat::py::sim::dynamics {

void Bind(pybind11::module& m)
{
    BindFemElastoDynamics(m);
}

} // namespace pbat::py::sim::dynamics