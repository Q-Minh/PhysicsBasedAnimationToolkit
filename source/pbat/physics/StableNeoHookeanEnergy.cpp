#include "pbat/physics/StableNeoHookeanEnergy.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/physics/HyperElasticity.h"

#include <Eigen/LU>
#include <doctest/doctest.h>

TEST_CASE("[physics] StableNeoHookeanEnergy")
{
    using namespace pbat;
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        physics::StableNeoHookeanEnergy<Dims> psi{};
        Matrix<Dims, Dims> const F = Matrix<Dims, Dims>::Identity();
        Scalar constexpr Y         = 1e6;
        Scalar constexpr nu        = 0.45;
        auto const [mu, lambda]    = physics::LameCoefficients(Y, nu);
        auto const ePsi            = psi.eval(F.reshaped(), mu, lambda);
        auto const eGradPsi        = psi.evalWithGrad(F.reshaped(), mu, lambda);
        Scalar const ePsiFromGrad  = std::get<0>(eGradPsi);
        auto const eGradHessPsi    = psi.evalWithGradAndHessian(F.reshaped(), mu, lambda);
        Scalar const ePsiFromHess  = std::get<0>(eGradHessPsi);
        bool const bIsEnergyNonNegative =
            (ePsi >= 0.) && (ePsiFromGrad >= 0.) && (ePsiFromHess >= 0.);
        CHECK(bIsEnergyNonNegative);

        Scalar const gamma = 1. + mu / lambda;
        Scalar const I2    = (F.array() * F.array()).sum();
        Scalar const I3    = F.determinant();
        Scalar const ePsiExpected =
            0.5 * mu * (I2 - Dims) + 0.5 * lambda * (I3 - gamma) * (I3 - gamma);
        Scalar const ePsiError = std::abs(ePsi - ePsiExpected) +
                                 std::abs(ePsiFromGrad - ePsiExpected) +
                                 std::abs(ePsiFromHess - ePsiExpected);
        Scalar constexpr zero = 1e-9;
        CHECK_LE(ePsiError, zero);
    });
}