#ifndef PBA_PHYSICS_HYPER_ELASTICITY_H
#define PBA_PHYSICS_HYPER_ELASTICITY_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/aliases.h"

#include <concepts>
#include <exception>
#include <format>
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
        t.eval(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    } -> std::convertible_to<Scalar>;
    {
        t.grad(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    } -> std::convertible_to<Vector<T::kDims * T::kDims>>;
    {
        t.hessian(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    } -> std::convertible_to<Matrix<T::kDims * T::kDims, T::kDims * T::kDims>>;
    {
        t.evalWithGrad(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    } -> std::convertible_to<std::tuple<Scalar, Vector<T::kDims * T::kDims>>>;
    {
        t.evalWithGradAndHessian(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    } -> std::convertible_to<std::tuple<
          Scalar,
          Vector<T::kDims * T::kDims>,
          Matrix<T::kDims * T::kDims, T::kDims * T::kDims>>>;
    {
        t.gradAndHessian(Matrix<T::kDims, T::kDims>{}.reshaped(), Scalar{}, Scalar{})
    }
    -> std::convertible_to<
        std::tuple<Vector<T::kDims * T::kDims>, Matrix<T::kDims * T::kDims, T::kDims * T::kDims>>>;
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
        std::string const what = std::format(
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

#endif // PBA_PHYSICS_HYPER_ELASTICITY_H