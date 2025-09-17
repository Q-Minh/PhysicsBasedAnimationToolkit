#ifndef PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
#define PYPBAT_FEM_HYPERELASTICPOTENTIAL_H

#include "Mesh.h"

#include <nanobind/nanobind.h>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

enum class EHyperElasticEnergy { SaintVenantKirchhoff, StableNeoHookean };

void BindHyperElasticPotential(nanobind::module_& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
