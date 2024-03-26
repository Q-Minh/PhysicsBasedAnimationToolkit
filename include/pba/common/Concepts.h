#ifndef PBA_COMMON_CONCEPTS_H
#define PBA_COMMON_CONCEPTS_H

#include <concepts>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace pba {
namespace common {

template <class T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <class R>
using RangeIteratorType = decltype(std::ranges::begin(std::declval<R>()));

template <class R>
using RangeValueType = std::iterator_traits<RangeIteratorType<R>>::value_type;

template <class R>
concept IndexRange = std::ranges::range<R> && std::is_integral_v<RangeValueType<R>>;

template <class R>
concept ContiguousIndexRange =
    IndexRange<R> && std::ranges::sized_range<R> && std::ranges::contiguous_range<R>;

template <class R>
concept ArithmeticRange = std::ranges::range<R> && Arithmetic<RangeValueType<R>>;

template <class R>
concept ContiguousArithmeticRange =
    ArithmeticRange<R> && std::ranges::sized_range<R> && std::ranges::contiguous_range<R>;

} // namespace common
} // namespace pba

#endif // PBA_COMMON_CONCEPTS_H
