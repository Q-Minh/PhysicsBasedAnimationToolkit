/**
 * @file HyperElasticity.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup physics
 */

#ifndef PBAT_PHYSICS_HYPER_ELASTICITY_H
#define PBAT_PHYSICS_HYPER_ELASTICITY_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <concepts>
#include <exception>
#include <fmt/core.h>
#include <string>

namespace pbat {
namespace physics {

/**
 * @brief Compute the Lame coefficients from Young's modulus and Poisson's ratio
 *
 * @param Y Young's modulus
 * @param nu Poisson's ratio
 * @return std::pair<Scalar, Scalar> Lame coefficients (mu, lambda)
 * @ingroup physics
 */
PBAT_API std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu);

/**
 * @brief Compute the Lame coefficients from Young's modulus and Poisson's ratio
 *
 * @tparam TDerivedY Eigen dense expression of vector of Young's moduli
 * @tparam TDerivednu Eigen dense expression of vector of Poisson's ratios
 * @param Y Vector of Young's moduli
 * @param nu Vector of Poisson's ratios
 * @return std::pair<VectorX, VectorX> Lame coefficients (mu, lambda)
 * @ingroup physics
 */
template <class TDerivedY, class TDerivednu>
std::pair<VectorX, VectorX>
LameCoefficients(Eigen::DenseBase<TDerivedY> const& Y, Eigen::DenseBase<TDerivednu> const& nu);

/**
 * @brief Concept for hyperelastic energy
 *
 * @tparam T Type to check
 * @ingroup physics
 */
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
            Scalar{},
            std::declval<math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>&>())
    } -> std::convertible_to<Scalar>;
    {
        t.evalWithGradAndHessian(
            math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{},
            Scalar{},
            Scalar{},
            std::declval<math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>&>(),
            std::declval<
                math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims, T::kDims * T::kDims>&>())
    } -> std::convertible_to<Scalar>;
    {t.gradAndHessian(
        math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims>{},
        Scalar{},
        Scalar{},
        std::declval<math::linalg::mini::SVector<Scalar, T::kDims * T::kDims>&>(),
        std::declval<
            math::linalg::mini::SMatrix<Scalar, T::kDims * T::kDims, T::kDims * T::kDims>&>())};
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