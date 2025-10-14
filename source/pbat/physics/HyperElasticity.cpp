#include "HyperElasticity.h"

namespace pbat {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu)
{
    Scalar const mu     = Y / (Scalar(2) * (Scalar(1) + nu));
    Scalar const lambda = (Y * nu) / ((Scalar(1) + nu) * (Scalar(1) - Scalar(2) * nu));
    return {mu, lambda};
}

} // namespace physics
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[physics] HyperElasticity")
{
    using namespace pbat;
    Scalar constexpr Y                   = Scalar(1e6);
    Scalar constexpr nu                  = Scalar(0.45);
    auto const [mu, lambda]              = physics::LameCoefficients(Y, nu);
    auto constexpr kNumberOfCoefficients = 5;
    auto const [mus, lambdas]            = physics::LameCoefficients(
        VectorX::Constant(kNumberOfCoefficients, Y),
        VectorX::Constant(kNumberOfCoefficients, nu));
    CHECK_EQ(mus.size(), kNumberOfCoefficients);
    CHECK_EQ(lambdas.size(), kNumberOfCoefficients);
    bool const bAreCoefficientsSame =
        (mus.array() == mu).all() and (lambdas.array() == lambda).all();
    CHECK(bAreCoefficientsSame);
}