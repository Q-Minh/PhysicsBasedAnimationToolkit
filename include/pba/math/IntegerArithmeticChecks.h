#ifndef PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H
#define PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H

#include <concepts>
#include <limits>

namespace pba {
namespace math {

template <std::integral Integer>
bool add_overflows(Integer a, Integer b)
{
    auto constexpr max = std::numeric_limits<Integer>::max();
    auto constexpr min = std::numeric_limits<Integer>::lowest();
    if (a < 0 && b < 0)
    {
        // adding negative numbers may underflow, i.e. a+b < min
        return a < (min - b);
    }
    if (a >= 0 && b >= 0)
    {
        // a+b > max <=> overflow
        return a > (max - b);
    }
    return false;
}

template <std::integral Integer>
bool multiply_overflows(Integer a, Integer b)
{
    auto constexpr max = std::numeric_limits<Integer>::max();
    auto constexpr min = std::numeric_limits<Integer>::lowest();
    if ((a >= 0 && b >= 0) or (a < 0 && b < 0))
    {
        // multiplying 2 same-sign numbers may overflow
        // a*b > max <=> overflow
        return a > (max / b);
    }
    else
    {
        // multiplying different sign numbers may underflow
        // a*b < min <=> underflow
        return a < (min / b);
    }
}

template <std::integral Integer>
bool negation_overflows(Integer a)
{
    auto constexpr max = std::numeric_limits<Integer>::max();
    return a == max;
}

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H