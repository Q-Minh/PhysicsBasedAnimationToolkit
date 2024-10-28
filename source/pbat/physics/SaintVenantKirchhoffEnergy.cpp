#include "SaintVenantKirchhoffEnergy.h"

#include "HyperElasticity.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/linalg/mini/Eigen.h>

TEST_CASE("[physics] SaintVenantKirchhoffEnergy")
{
    using namespace pbat;
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        using math::linalg::mini::FromEigen;
        physics::SaintVenantKirchhoffEnergy<Dims> psi{};
        Matrix<Dims, Dims> const F = Matrix<Dims, Dims>::Identity();
        Scalar constexpr Y         = 1e6;
        Scalar constexpr nu        = 0.45;
        auto const [mu, lambda]    = physics::LameCoefficients(Y, nu);
        auto vecF                  = FromEigen(F.reshaped());
        Scalar const ePsi          = psi.eval(vecF, mu, lambda);
        auto const eGradPsi        = psi.evalWithGrad(vecF, mu, lambda);
        Scalar const ePsiFromGrad  = std::get<0>(eGradPsi);
        auto const eGradHessPsi    = psi.evalWithGradAndHessian(vecF, mu, lambda);
        Scalar const ePsiFromHess  = std::get<0>(eGradHessPsi);
        bool const bIsEnergyNonNegative =
            (ePsi >= 0.) && (ePsiFromGrad >= 0.) && (ePsiFromHess >= 0.);
        CHECK(bIsEnergyNonNegative);

        Matrix<Dims, Dims> const E = 0.5 * (F.transpose() * F - Matrix<Dims, Dims>::Identity());
        Scalar const trE           = E.trace();
        Scalar const ePsiExpected  = mu * (E.array() * E.array()).sum() + 0.5 * lambda * trE * trE;
        Scalar const ePsiError     = std::abs(ePsi - ePsiExpected) +
                                 std::abs(std::get<0>(eGradPsi) - ePsiExpected) +
                                 std::abs(std::get<0>(eGradHessPsi) - ePsiExpected);
        Scalar constexpr zero = 1e-15;
        CHECK_LE(ePsiError, zero);
    });
}