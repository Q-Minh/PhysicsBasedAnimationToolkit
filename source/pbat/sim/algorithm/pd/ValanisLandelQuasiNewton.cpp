#include "ValanisLandelQuasiNewton.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][pd] ValanisLandelQuasiNewtonStiffness")
{
    using namespace pbat::sim::algorithm::pd;
    // Arrange
    double Y = 1e6;
    double nu = 0.45;
    double mu = Y / (2 * (1 + nu));
    double lambda = (Y * nu) / ((1 + nu) * (1 - 2 * nu));
    Eigen::Vector3d sigma{1.0, 1.0, 1.0};
    double sigmalo   = 0.5;
    double sigmahi   = 1.5;
    auto energy      = pbat::physics::EHyperElasticEnergy::StableNeoHookean;
    // Act
    double k = ValanisLandelQuasiNewtonStiffness(
        mu, lambda, sigma, sigmalo, sigmahi, energy);
    // Assert
    CHECK_GT(k, 0.0);
}