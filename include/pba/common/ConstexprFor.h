#ifndef PBA_CORE_COMMON_CONSTEXPR_FOR_H
#define PBA_CORE_COMMON_CONSTEXPR_FOR_H

#include <type_traits>
#include <utility>

namespace pba {
namespace common {

template <class... Ts, class F>
constexpr void ForTypes(F&& f)
{
    (f.template operator()<Ts>(), ...);
}

template <auto... Xs, class F>
constexpr void ForValues(F&& f)
{
    (f.template operator()<Xs>(), ...);
}

template <auto Begin, auto End, typename F>
constexpr void ForRange(F&& f)
{
    using CounterType = std::common_type_t<decltype(Begin), decltype(End)>;

    [&f]<auto... Is>(std::integer_sequence<CounterType, Is...>) {
        ForValues<(Begin + Is)...>(f);
    }(std::make_integer_sequence<CounterType, End - Begin>{});
}

} // namespace common
} // namespace pba

#endif // PBA_CORE_COMMON_CONSTEXPR_FOR_H