#include "FemElastoDynamics.h"

#include "pbat/fem/Tetrahedron.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][dynamics] FemElastoDynamics")
{
    // Arrange
    using namespace pbat;
    // Cube mesh
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
    auto constexpr kOrder   = 1;
    auto constexpr kDims    = 3;
    using ElementType       = fem::Tetrahedron<kOrder>;
    using ElasticEnergyType = physics::StableNeoHookeanEnergy<kDims>;
    using ElastoDynamics = sim::dynamics::FemElastoDynamics<ElementType, kDims, ElasticEnergyType>;

    // Act
    ElastoDynamics dynamics{};
    dynamics.Construct(V, C);
    auto const nNodes = dynamics.mesh.X.cols();
    Eigen::Vector<bool, Eigen::Dynamic> D(nNodes);
    D.setConstant(false);
    D(2) = true;
    dynamics.Constrain(D);
    dynamics.SetupTimeIntegrationOptimization();

    // Assert
    CHECK_EQ(dynamics.FreeNodes().size(), nNodes - 1);
    CHECK_EQ(dynamics.DirichletNodes().size(), 1);
    CHECK_EQ(dynamics.aext().rows(), kDims);
    CHECK_EQ(dynamics.aext().cols(), nNodes);
    CHECK_EQ(dynamics.fext.rows(), kDims);
    CHECK_EQ(dynamics.fext.cols(), nNodes);
    CHECK_EQ(dynamics.x.rows(), kDims);
    CHECK_EQ(dynamics.x.cols(), nNodes);
    CHECK_EQ(dynamics.v.rows(), kDims);
    CHECK_EQ(dynamics.v.cols(), nNodes);
    CHECK_EQ(dynamics.m.size(), nNodes);
    CHECK_EQ(dynamics.M().size(), nNodes * kDims);
    CHECK_EQ(dynamics.xtilde.rows(), kDims);
    CHECK_EQ(dynamics.xtilde.cols(), nNodes);
}