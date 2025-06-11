#ifndef PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
#define PYPBAT_FEM_HYPERELASTICPOTENTIAL_H

#include "Mesh.h"

#include <pbat/fem/Hexahedron.h>
#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pybind11/pybind11.h>
#include <tuple>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

enum class EHyperElasticEnergy { SaintVenantKirchhoff, StableNeoHookean };

template <class Func>
inline void ApplyToElementInDimsWithHyperElasticEnergy(
    EElement eElement,
    int order,
    int dims,
    EHyperElasticEnergy eEnergy,
    Func&& f)
{
    ApplyToElementInDims(eElement, order, dims, [&]<pbat::fem::CElement ElementType, auto Dims>() {
        switch (eEnergy)
        {
            case EHyperElasticEnergy::SaintVenantKirchhoff:
                f.template
                operator()<ElementType, Dims, pbat::physics::SaintVenantKirchhoffEnergy<Dims>>();
                break;
            case EHyperElasticEnergy::StableNeoHookean:
                f.template
                operator()<ElementType, Dims, pbat::physics::StableNeoHookeanEnergy<Dims>>();
                break;
            default: break;
        }
    });
}

void BindHyperElasticPotential(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
