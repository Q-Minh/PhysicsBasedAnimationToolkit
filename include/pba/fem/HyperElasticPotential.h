#ifndef PBA_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PBA_FEM_HYPER_ELASTIC_POTENTIAL_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/physics/HyperElasticity.h"

#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
struct HyperElasticPotential
{
  public:
    using SelfType          = HyperElasticPotential<TMesh, THyperElasticEnergy>;
    using MeshType          = TMesh;
    using ElementType       = typename TMesh::ElementType;
    using ElasticEnergyType = THyperElasticEnergy;

    static auto constexpr kDims = THyperElasticEnergy::kDims;
    static int constexpr kOrder = ElementType::kOrder - 1;

    template <int OrderPrivate>
    struct OrderSelector
    {
        static auto constexpr kOrder = OrderPrivate - 1;
    };

    template <>
    struct OrderSelector<1>
    {
        static auto constexpr kOrder = 1;
    };

    using QuadratureRuleType =
        ElementType::template QuadratureType<OrderSelector<ElementType::kOrder>::kOrder>;
};

} // namespace fem
} // namespace pba

#endif // PBA_FEM_HYPER_ELASTIC_POTENTIAL_H