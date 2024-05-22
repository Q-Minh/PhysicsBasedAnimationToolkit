#ifndef PBA_PHYSICS_HYPER_ELASTICITY_H
#define PBA_PHYSICS_HYPER_ELASTICITY_H

#include "pba/aliases.h"

#include <concepts>
#include <tuple>

namespace pba {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu);

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
    } -> std::convertible_to<
        std::tuple<Vector<T::kDims * T::kDims>, Matrix<T::kDims * T::kDims, T::kDims * T::kDims>>>;
};

} // namespace physics
} // namespace pba

#endif // PBA_PHYSICS_HYPER_ELASTICITY_H