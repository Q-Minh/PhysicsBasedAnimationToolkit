#include "pba/physics/HyperElasticity.h"

#include <doctest/doctest.h>

namespace pba {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu)
{
    Scalar const mu     = Y / (2. * (1. + nu));
    Scalar const lambda = Y * nu / ((1. + nu) * (1. - 2. * nu));
    return {mu, lambda};
}

} // namespace physics
} // namespace pba

namespace pba {
namespace test {
struct HyperElasticEnergy
{
    static auto constexpr kDims = 3;

    template <class Derived>
    Scalar eval(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class Derived>
    Vector<kDims * kDims> grad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class Derived>
    Matrix<kDims * kDims, kDims * kDims>
    hessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class Derived>
    std::tuple<Scalar, Vector<kDims * kDims>>
    evalWithGrad(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class Derived>
    std::tuple<Scalar, Vector<kDims * kDims>, Matrix<kDims * kDims, kDims * kDims>>
    evalWithGradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }

    template <class Derived>
    std::tuple<Vector<kDims * kDims>, Matrix<kDims * kDims, kDims * kDims>>
    gradAndHessian(Eigen::DenseBase<Derived> const& F, Scalar mu, Scalar lambda) const
    {
        return {};
    }
};

} // namespace test
} // namespace pba

TEST_CASE("[physics] HyperElasticity")
{
    CHECK(pba::physics::CHyperElasticEnergy<pba::test::HyperElasticEnergy>);
}