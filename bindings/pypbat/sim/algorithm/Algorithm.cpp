#include "Algorithm.h"

#include "newton/Newton.h"
#include "pd/PD.h"

namespace pbat::py::sim::algorithm {

void Bind(nanobind::module_& m)
{
    [[maybe_unused]] nanobind::module_ malgorithm =
        m.def_submodule("algorithm", "Simulation algorithms.");
    newton::Bind(malgorithm);
    pd::Bind(malgorithm);
}

} // namespace pbat::py::sim::algorithm