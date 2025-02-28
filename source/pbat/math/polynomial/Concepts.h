/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Concepts for polynomial basis
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_POLYNOMIAL_CONCEPTS_H
#define PBAT_MATH_POLYNOMIAL_CONCEPTS_H

#include "pbat/Aliases.h"

#include <concepts>

namespace pbat::math::polynomial {

/**
 * @brief
 *
 * @tparam T
 */
template <class T>
concept CBasis = requires(T t)
{
    requires std::is_integral_v<decltype(T::kDims)>;
    requires std::is_integral_v<decltype(T::kOrder)>;
    requires std::is_integral_v<decltype(T::kSize)>;
    {
        t.eval(Vector<T::kDims>{})
    } -> std::convertible_to<Vector<T::kSize>>;
    {
        t.derivatives(Vector<T::kDims>{})
    } -> std::convertible_to<Matrix<T::kDims, T::kSize>>;
    {
        t.antiderivatives(Vector<T::kDims>{})
    } -> std::convertible_to<Matrix<T::kSize, T::kDims>>;
};

/**
 * @brief
 *
 * @tparam T
 */
template <class T>
concept CVectorBasis = requires(T t)
{
    requires std::is_integral_v<decltype(T::kDims)>;
    requires std::is_integral_v<decltype(T::kOrder)>;
    requires std::is_integral_v<decltype(T::kSize)>;
    {
        t.eval(Vector<T::kDims>{})
    } -> std::convertible_to<Matrix<T::kSize, T::kDims>>;
};

} // namespace pbat::math::polynomial

#endif // PBAT_MATH_POLYNOMIAL_CONCEPTS_H
