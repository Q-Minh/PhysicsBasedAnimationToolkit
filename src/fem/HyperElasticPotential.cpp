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
    // Cube tetrahedral mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 0., 1., 1., 0., 0., 1., 1.,
            0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
            1, 2, 4, 7, 5,
            3, 0, 6, 5, 3,
            5, 6, 0, 3, 6;
    // clang-format on
    Scalar constexpr Y  = 1e6;
    Scalar constexpr nu = 0.45;
    SUBCASE("Linear basis functions")
    {
        auto constexpr kDims       = 3;
        auto constexpr kOrder      = 1;
        using ElasticEnergyType    = physics::StableNeoHookeanEnergy<3>;
        using ElementType          = fem::Tetrahedron<kOrder>;
        using MeshType             = fem::Mesh<ElementType, kDims>;
        using ElasticPotentialType = fem::HyperElasticPotential<MeshType, ElasticEnergyType>;
        using QuadratureType       = ElasticPotentialType::QuadratureRuleType;
        // There is no order-0 polynomial quadrature, so hyper elastic energies on linear basis
        // functions should default to order-1 quadratures (where the point will actually not
        // matter).
        CHECK_EQ(QuadratureType::kOrder, 1);

        MeshType const M(V, C);
        VectorX const x = M.X.reshaped();
        ElasticPotentialType U(M, x, Y, nu);
        MatrixX const H = U.ToMatrix();
    }
    SUBCASE("Higher order basis functions")
    {
        common::ForValues<2, 3>([&]<auto kOrder>() {
            auto constexpr kDims       = 3;
            using ElasticEnergyType    = physics::StableNeoHookeanEnergy<3>;
            using ElementType          = fem::Tetrahedron<kOrder>;
            using MeshType             = fem::Mesh<ElementType, kDims>;
            using ElasticPotentialType = fem::HyperElasticPotential<MeshType, ElasticEnergyType>;
            using QuadratureType       = ElasticPotentialType::QuadratureRuleType;
            // For order-k basis functions, our quadrature should be order k-1
            CHECK_EQ(QuadratureType::kOrder, kOrder - 1);
            MeshType const M(V, C);
            VectorX const x = M.X.reshaped();
            ElasticPotentialType U(M, x, Y, nu);
            MatrixX const H = U.ToMatrix();
        });
    }
}
