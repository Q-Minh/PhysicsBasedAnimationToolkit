/**
 * @file ConstexprFor.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Compile-time for loops
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_COMMON_CONSTEXPRFOR_H
#define PBAT_COMMON_CONSTEXPRFOR_H

#include <type_traits>
#include <utility>

namespace pbat {
namespace common {

/**
 * @brief Compile-time for loop over types
 *
 * @tparam Ts Types to loop over
 * @tparam F Callable with signature `void operator()<T>()`
 * @param f Function object to call
 */
template <class... Ts, class F>
constexpr void ForTypes(F&& f)
{
    (f.template operator()<Ts>(), ...);
}

/**
 * @brief Compile-time for loop over values
 *
 * @tparam Xs Values to loop over
 * @tparam F Callable with signature `void operator()<X>()`
 * @param f Function object to call
 */
template <auto... Xs, class F>
constexpr void ForValues(F&& f)
{
    (f.template operator()<Xs>(), ...);
}

/**
 * @brief Compile-time for loop over a range of values
 *
 * @tparam Begin Starting loop index
 * @tparam End Ending loop index (exclusive)
 * @tparam F Callable with signature `void operator()<decltype(Begin)>()`
 * @param f Function object to call
 */
template <auto Begin, auto End, typename F>
constexpr void ForRange(F&& f)
{
    using CounterType = std::common_type_t<decltype(Begin), decltype(End)>;

    [&f]<auto... Is>(std::integer_sequence<CounterType, Is...>) {
        ForValues<(Begin + Is)...>(f);
    }(std::make_integer_sequence<CounterType, End - Begin>{});
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_CONSTEXPRFOR_H
