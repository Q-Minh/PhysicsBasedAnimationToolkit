#include "StableNeoHookeanEnergy.h"

#include "HyperElasticity.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Eigenvalues.h"

#include <Eigen/LU>
#include <doctest/doctest.h>

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
        auto vecF                  = FromEigen(F);
        auto const ePsi            = psi.Eval(vecF, mu, lambda);
        mini::SVector<Scalar, Dims * Dims> gF;
        Scalar const ePsiFromGrad = psi.EvalWithGrad(vecF, mu, lambda, gF);
        gF.SetZero();
        mini::SMatrix<Scalar, Dims * Dims, Dims * Dims> HF;
        Scalar const ePsiFromHess = psi.EvalWithGradAndHessian(vecF, mu, lambda, gF, HF);
        bool const bIsEnergyNonNegative =
            (ePsi >= 0.) and (ePsiFromGrad >= 0.) and (ePsiFromHess >= 0.);
        CHECK(bIsEnergyNonNegative);
        auto eigs = mini::SymmetricEigenNxN(HF);
        bool const bHasNegativeEigenvalue = mini::Max(eigs.lambda < Scalar(0));
        CHECK_FALSE(bHasNegativeEigenvalue);

        Scalar const I2         = (F.array() * F.array()).sum();
        Scalar const I3         = F.determinant();
        Scalar const I3minAlpha = I3 - 1 - mu / lambda;
        Scalar const ePsiExpected =
            Scalar(0.5) * mu * (I2 - Dims) + Scalar(0.5) * lambda * (I3minAlpha * I3minAlpha);
        Scalar const ePsiError = std::abs(ePsi - ePsiExpected) +
                                 std::abs(ePsiFromGrad - ePsiExpected) +
                                 std::abs(ePsiFromHess - ePsiExpected);
        auto constexpr zero = Scalar(1e-9);
        CHECK_LE(ePsiError, zero);
    });
}