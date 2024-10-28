#ifndef PBAT_PHYSICS_HYPER_ELASTICITY_H
#define PBAT_PHYSICS_HYPER_ELASTICITY_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <concepts>
#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <string>
#include <tuple>

namespace pbat {
namespace physics {

PBAT_API std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu);

template <class TDerivedY, class TDerivednu>
std::pair<VectorX, VectorX>
LameCoefficients(Eigen::DenseBase<TDerivedY> const& Y, Eigen::DenseBase<TDerivednu> const& nu);

template <class T>
concept CHyperElasticEnergy = requires(T t)
{
    {
        T::kDims
    } -> std::convertible_to<int>;
    {
        t.eval(math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{}, Scalar{}, Scalar{})
    } -> std::convertible_to<Scalar>;
    {
        t.grad(math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{}, Scalar{}, Scalar{})
    } -> std::convertible_to<math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>>;
    {
        t.hessian(math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{}, Scalar{}, Scalar{})
    } -> std::convertible_to<
        math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims, T::kDims * T::kDims>>;
    {
        t.evalWithGrad(
            math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{},
            Scalar{},
            Scalar{})
    } -> std::convertible_to<
        std::tuple<Scalar, math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>>>;
    {
        t.evalWithGradAndHessian(
            math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{},
            Scalar{},
            Scalar{})
    } -> std::convertible_to<std::tuple<
        Scalar,
        math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>,
        math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims, T::kDims * T::kDims>>>;
    {
        t.gradAndHessian(
            math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{},
            Scalar{},
            Scalar{})
    } -> std::convertible_to<std::tuple<
        math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>,
        math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims, T::kDims * T::kDims>>>;
};

template <class TDerivedY, class TDerivednu>
std::pair<VectorX, VectorX>
LameCoefficients(Eigen::DenseBase<TDerivedY> const& Y, Eigen::DenseBase<TDerivednu> const& nu)
{
    bool const bYHasExpectedDimensions  = (Y.rows() == 1) or (Y.cols() == 1);
    bool const bNuHasExpectedDimensions = (nu.rows() == 1) or (nu.cols() == 1);
    bool const bHaveSameDimensions      = Y.size() == nu.size();
    if (not(bYHasExpectedDimensions and bNuHasExpectedDimensions and bHaveSameDimensions))
    {
        std::string const what = fmt::format(
            "Expected equivalent dimensions in Y and nu, with Y and nu being 1D arrays, but got "
            "size(Y)={}, size(nu)={}",
            Y.size(),
            nu.size());
        throw std::invalid_argument(what);
    }
    VectorX mu(Y.size());
    VectorX lambda(Y.size());
    for (auto i = 0; i < Y.size(); ++i)
    {
        std::tie(mu(i), lambda(i)) = LameCoefficients(Y(i), nu(i));
    }
    return {mu, lambda};
}

} // namespace physics
} // namespace pbat

#endif // PBAT_PHYSICS_HYPER_ELASTICITY_H