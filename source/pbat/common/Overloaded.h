/**
 * @file Overloaded.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Helper class to allow multiple inheritance of operator() for lambdas.
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_COMMON_OVERLOADED_H
#define PBAT_COMMON_OVERLOADED_H

namespace pbat::common {

/**
 * @brief C++20 feature to allow multiple inheritance of operator() for lambdas.
 *
 * Particularly useful for `std::visit` with `std::variant`.
 * See [example](https://en.cppreference.com/w/cpp/utility/variant/visit2)
 *
 * @tparam ...Ts Variadic template parameter pack for the types to inherit from.
 */
template <class... Ts>
struct Overloaded : Ts...
{
    using Ts::operator()...;
};

} // namespace pbat::common

#endif // PBAT_COMMON_OVERLOADED_H
