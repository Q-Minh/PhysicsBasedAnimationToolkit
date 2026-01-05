#include "SaintVenantKirchhoffEnergy.h"

#include "HyperElasticity.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/linalg/mini/Eigen.h>

TEST_CASE("[physics] SaintVenantKirchhoffEnergy")
{
    using namespace pbat;
    namespace mini = pbat::math::linalg::mini;
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        using mini::FromEigen;
        physics::SaintVenantKirchhoffEnergy<Dims> psi{};
        Matrix<Dims, Dims> const F = Matrix<Dims, Dims>::Identity();
        Scalar constexpr Y         = Scalar(1e6);
        Scalar constexpr nu        = Scalar(0.45);
        auto const [mu, lambda]    = physics::LameCoefficients(Y, nu);
        auto vecF                  = FromEigen(F.reshaped());
        Scalar const ePsi          = psi.Eval(vecF, mu, lambda);
        mini::SVector<Scalar, Dims * Dims> gF;
        Scalar const ePsiFromGrad = psi.EvalWithGrad(vecF, mu, lambda, gF);
        gF.SetZero();
        mini::SMatrix<Scalar, Dims * Dims, Dims * Dims> HF;
        Scalar const ePsiFromHess = psi.EvalWithGradAndHessian(vecF, mu, lambda, gF, HF);
        bool const bIsEnergyNonNegative =
            (ePsi >= 0.) and (ePsiFromGrad >= 0.) and (ePsiFromHess >= 0.);
        CHECK(bIsEnergyNonNegative);

        Matrix<Dims, Dims> const E = 0.5 * (F.transpose() * F - Matrix<Dims, Dims>::Identity());
        Scalar const trE           = E.trace();
        Scalar const ePsiExpected =
            mu * (E.array() * E.array()).sum() + Scalar(0.5) * lambda * trE * trE;
        Scalar const ePsiError = std::abs(ePsi - ePsiExpected) +
                                 std::abs(ePsiFromGrad - ePsiExpected) +
                                 std::abs(ePsiFromHess - ePsiExpected);
        auto constexpr zero = Scalar(1e-15);
        CHECK_LE(ePsiError, zero);
    });
}