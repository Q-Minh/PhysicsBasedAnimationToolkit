#include "HyperElasticPotential.h"

#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindHyperElasticPotential(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::enum_<EHyperElasticEnergy>(m, "HyperElasticEnergy")
        .value("SaintVenantKirchhoff", EHyperElasticEnergy::SaintVenantKirchhoff)
        .value("StableNeoHookean", EHyperElasticEnergy::StableNeoHookean)
        .export_values();

}

} // namespace fem
} // namespace py
} // namespace pbat