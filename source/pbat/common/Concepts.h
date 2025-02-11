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
    {
        std::ranges::range_value_t<R>::RowsAtCompileTime
    } -> std::convertible_to<int>;
    {
        std::ranges::range_value_t<R>::ColsAtCompileTime
    } -> std::convertible_to<int>;
    requires std::is_arithmetic_v<typename std::ranges::range_value_t<R>::Scalar>;
    {std::ranges::range_value_t<R>::Flags};
};

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_CONCEPTS_H
