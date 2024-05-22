#include "pba/fem/HyperElasticPotential.h"

#include "pba/common/ConstexprFor.h"
#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"
#include "pba/physics/HyperElasticity.h"
#include "pba/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] HyperElasticPotential")
{
    using namespace pba;
    SUBCASE("Linear basis functions")
    {
        auto constexpr kDims   = 3;
        auto constexpr kOrder  = 1;
        using ElasticEnergy    = physics::StableNeoHookeanEnergy<3>;
        using ElementType      = fem::Tetrahedron<kOrder>;
        using MeshType         = fem::Mesh<ElementType, kDims>;
        using ElasticPotential = fem::HyperElasticPotential<MeshType, ElasticEnergy>;
        using QuadratureType   = ElasticPotential::QuadratureRuleType;
        // There is no order-0 polynomial quadrature, so hyper elastic energies on linear basis
        // functions should default to order-1 quadratures (where the point will actually not
        // matter).
        CHECK_EQ(QuadratureType::kOrder, 1);
    }
    SUBCASE("Higher order basis functions")
    {
        common::ForValues<2, 3>([]<auto kOrder>() {
            auto constexpr kDims   = 3;
            using ElasticEnergy    = physics::StableNeoHookeanEnergy<3>;
            using ElementType      = fem::Tetrahedron<kOrder>;
            using MeshType         = fem::Mesh<ElementType, kDims>;
            using ElasticPotential = fem::HyperElasticPotential<MeshType, ElasticEnergy>;
            using QuadratureType   = ElasticPotential::QuadratureRuleType;
            // For order-k basis functions, our quadrature should be order k-1
            CHECK_EQ(QuadratureType::kOrder, kOrder - 1);
        });
    }
}
