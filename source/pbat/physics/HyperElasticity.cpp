#include "HyperElasticity.h"

#include <doctest/doctest.h>

namespace pbat {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu)
{
    Scalar const mu     = Y / (2. * (1. + nu));
    Scalar const lambda = (Y * nu) / ((1. + nu) * (1. - 2. * nu));
    return {mu, lambda};
}

} // namespace physics
} // namespace pbat

namespace pbat {
namespace test {
struct HyperElasticEnergy
{
    static auto constexpr kDims = 3;

    template <class TDerived>
    Scalar eval(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class TDerived>
    Vector<kDims * kDims> grad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class TDerived>
    Matrix<kDims * kDims, kDims * kDims>
    hessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class TDerived>
    std::tuple<Scalar, Vector<kDims * kDims>>
    evalWithGrad(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class TDerived>
    std::tuple<Scalar, Vector<kDims * kDims>, Matrix<kDims * kDims, kDims * kDims>>
    evalWithGradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class TDerived>
    std::tuple<Vector<kDims * kDims>, Matrix<kDims * kDims, kDims * kDims>>
    gradAndHessian(Eigen::DenseBase<TDerived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }
};

} // namespace test
} // namespace pbat

TEST_CASE("[physics] HyperElasticity")
{
    using namespace pbat;
    CHECK(physics::CHyperElasticEnergy<test::HyperElasticEnergy>);
    Scalar constexpr Y                   = 1e6;
    Scalar constexpr nu                  = 0.45;
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