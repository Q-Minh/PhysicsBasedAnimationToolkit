/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the concepts used in the math module.
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PBAT_MATH_CONCEPTS_H
#define PBAT_MATH_CONCEPTS_H

#include <concepts>
#include <pbat/Aliases.h>
#include <pbat/common/Concepts.h>

namespace pbat {
namespace math {

/**
 * @brief
 *
 * @tparam T
 */
template <class T>
concept CPolynomialBasis = requires(T t)
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
concept CVectorPolynomialBasis = requires(T t)
{
    requires std::is_integral_v<decltype(T::kDims)>;
    requires std::is_integral_v<decltype(T::kOrder)>;
    requires std::is_integral_v<decltype(T::kSize)>;
    {
        t.eval(Vector<T::kDims>{})
    } -> std::convertible_to<Matrix<T::kSize, T::kDims>>;
};

/**
 * @brief
 *
 * @tparam Q
 */
template <class Q>
concept CQuadratureRule = requires(Q q)
{
    requires std::integral<decltype(Q::kDims)>;
    requires common::CContiguousArithmeticRange<decltype(q.points)>;
    requires common::CContiguousArithmeticRange<decltype(q.weights)>;
    {
        q.points.size()
    } -> std::convertible_to<int>;
    {
        q.weights.size()
    } -> std::convertible_to<int>;
};

/**
 * @brief
 *
 * @tparam Q
 */
template <class Q>
concept CFixedPointQuadratureRule = requires(Q q)
{
    requires CQuadratureRule<Q>;
    requires std::is_integral_v<decltype(Q::kPoints)>;
    {q.points.size() / q.weights.size() == Q::kDims};
    {q.weights.size() == Q::kPoints};
};

/**
 * @brief
 *
 * @tparam Q
 */
template <class Q>
concept CPolynomialQuadratureRule = requires(Q q)
{
    requires CQuadratureRule<Q>;
    requires std::is_integral_v<decltype(Q::kOrder)>;
};

/**
 * @brief
 *
 * @tparam Q
 */
template <class Q>
concept CFixedPointPolynomialQuadratureRule =
    CFixedPointQuadratureRule<Q> and CPolynomialQuadratureRule<Q>;

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_CONCEPTS_H