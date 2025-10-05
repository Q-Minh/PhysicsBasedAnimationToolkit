/**
 * @file Concepts.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Concepts for common types
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_CONCEPTS_H
#define PBAT_COMMON_CONCEPTS_H

#include <concepts>
#include <iterator>
#include <pbat/Aliases.h>
#include <ranges>
#include <type_traits>

namespace pbat {
namespace common {

/**
 * @brief Concept for arithmetic types
 *
 * This concept checks if a type is an arithmetic type, which includes
 * integral types (like int, long, etc.) and floating-point types (like float, double).
 *
 * @tparam T
 */
template <class T>
concept CArithmetic = std::is_arithmetic_v<T>;

/**
 * @brief Concept for integral types
 *
 * This concept checks if a type is a non-bool integral type, which includes
 * types like int, long, short, etc.
 *
 * @todo Update all other concepts that manually check for integral types to use this concept
 * instead.
 *
 * @tparam T
 */
template <class T>
concept CIndex = std::is_integral_v<T> and not std::is_same_v<T, bool>;

/**
 * @brief Concept for floating-point types
 *
 * This concept checks if a type is a floating-point type, which includes
 * types like float, double, etc.
 *
 * @tparam T
 */
template <class T>
concept CFloatingPoint = std::is_floating_point_v<T>;

/**
 * @brief Range of integer types
 *
 * @tparam R
 */
template <class R>
concept CIndexRange =
    std::ranges::range<R> && std::is_convertible_v<std::ranges::range_value_t<R>, int>;

/**
 * @brief Contiguous range of integer types
 *
 * @tparam R
 */
template <class R>
concept CContiguousIndexRange =
    CIndexRange<R> && std::ranges::sized_range<R> && std::ranges::contiguous_range<R>;

/**
 * @brief Range of arithmetic types
 *
 * @tparam R
 */
template <class R>
concept CArithmeticRange =
    std::ranges::range<R> && std::is_arithmetic_v<std::ranges::range_value_t<R>>;

/**
 * @brief Contiguous range of arithmetic types
 *
 * @tparam R
 */
template <class R>
concept CContiguousArithmeticRange =
    CArithmeticRange<R> && std::ranges::sized_range<R> && std::ranges::contiguous_range<R>;

/**
 * @brief Range of Eigen fixed-size matrix types
 *
 * @tparam R
 */
template <class R>
concept CContiguousArithmeticMatrixRange = requires(R r)
{
    requires std::ranges::range<R>;
    requires std::ranges::sized_range<R>;
    requires std::ranges::contiguous_range<R>;
    {std::ranges::range_value_t<R>::RowsAtCompileTime}->std::convertible_to<int>;
    {std::ranges::range_value_t<R>::ColsAtCompileTime}->std::convertible_to<int>;
    requires std::is_arithmetic_v<typename std::ranges::range_value_t<R>::Scalar>;
    {std::ranges::range_value_t<R>::Flags};
};

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_CONCEPTS_H
