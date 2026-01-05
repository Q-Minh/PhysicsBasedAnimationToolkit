#include "StableNeoHookeanEnergy.h"

#include "HyperElasticity.h"

#include <Eigen/LU>
#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/linalg/mini/Eigen.h>

TEST_CASE("[physics] StableNeoHookeanEnergy")
{
    using namespace pbat;
    namespace mini = pbat::math::linalg::mini;
    // WARNING:
    // Make tests robust to numerical precision issues.
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        using mini::FromEigen;
        physics::StableNeoHookeanEnergy<Dims> psi{};
        Matrix<Dims, Dims> const F = Matrix<Dims, Dims>::Identity();
        Scalar constexpr Y         = Scalar(1e6);
        Scalar constexpr nu        = Scalar(0.45);
        auto const [mu, lambda]    = physics::LameCoefficients(Y, nu);
        auto vecF                  = FromEigen(F.reshaped());
        auto const ePsi            = psi.Eval(vecF, mu, lambda);
        mini::SVector<Scalar, Dims * Dims> gF;
        Scalar const ePsiFromGrad = psi.EvalWithGrad(vecF, mu, lambda, gF);
        gF.SetZero();
        mini::SMatrix<Scalar, Dims * Dims, Dims * Dims> HF;
        Scalar const ePsiFromHess = psi.EvalWithGradAndHessian(vecF, mu, lambda, gF, HF);
        bool const bIsEnergyNonNegative =
            (ePsi >= 0.) and (ePsiFromGrad >= 0.) and (ePsiFromHess >= 0.);
        CHECK(bIsEnergyNonNegative);

        Scalar const gamma = Scalar(1) + mu / lambda;
        Scalar const I2    = (F.array() * F.array()).sum();
        Scalar const I3    = F.determinant();
        Scalar const ePsiExpected =
            Scalar(0.5) * mu * (I2 - Dims) + Scalar(0.5) * lambda * (I3 - gamma) * (I3 - gamma);
        Scalar const ePsiError = std::abs(ePsi - ePsiExpected) +
                                 std::abs(ePsiFromGrad - ePsiExpected) +
                                 std::abs(ePsiFromHess - ePsiExpected);
        auto constexpr zero = Scalar(1e-9);
        CHECK_LE(ePsiError, zero);
    });
}