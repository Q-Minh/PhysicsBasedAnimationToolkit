#ifndef PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H
#define PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H

#include <concepts>
#include <limits>
#include <stdexcept>

namespace pba {
namespace math {

/**
 * @brief Checks if the operation a+b is not in the range of values representable by the type of
 * Integer.
 * @tparam Integer
 * @param a
 * @param b
 * @return
 */
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

/**
 * @brief Checks if the operation a*b is not in the range of values representable by the type of
 * Integer.
 * @tparam Integer
 * @param a
 * @param b
 * @return
 */
template <std::integral Integer>
bool multiply_overflows(Integer a, Integer b)
{
    if (a == 0 or b == 0)
        return false;

    auto constexpr max  = std::numeric_limits<Integer>::max();
    auto constexpr min  = std::numeric_limits<Integer>::lowest();
    bool const sameSign = (a > 0 && b > 0) or (a < 0 && b < 0);
    if (sameSign)
    {
        // multiplying 2 same-sign numbers may overflow
        // |a|*|b| > max <=> overflow
        return std::abs(a) > std::abs(max / b);
    }
    else
    {
        // multiplying different sign numbers may underflow
        // -|a|*|b| < min <=> underflow
        return -std::abs(a) < (min / std::abs(b));
    }
}

/**
 * @brief Checks if the operation -a is not in the range of values representable by the type of
 * Integer.
 * @tparam Integer
 * @param a
 * @return
 */
template <std::integral Integer>
bool negation_overflows(Integer a)
{
    auto constexpr max = std::numeric_limits<Integer>::max();
    if constexpr (std::is_signed_v<Integer>)
    {
        // Signed integer values are in the range [-2^{n-1}+1, 2^{n-1}],
        // hence only 2^{n-1} cannot be negated and held in the signed integer.
        return a == max;
    }
    else
    {
        // All unsigned integers are positive except 0, hence negating anything > 0 is negative and
        // cannot be represented by an unsigned type.
        return a > 0;
    }
}

/**
 * @brief Wrapper around integer types that throws when integer overflow is detected
 * @tparam Integer
 */
template <std::integral Integer>
struct OverflowChecked
{
    using SelfType = OverflowChecked<Integer>;

    Integer& operator*() { return value; }
    Integer const& operator*() const { return value; }
    SelfType operator-() const
    {
        if (negation_overflows(value))
            throw std::overflow_error("Negation overflow");
        return SelfType{-value};
    }
    SelfType operator+(SelfType rhs) const
    {
        if (add_overflows(value, rhs.value))
            throw std::overflow_error("Addition overflow");
        return SelfType{value + rhs.value};
    }
    SelfType operator*(SelfType rhs) const
    {
        if (multiply_overflows(value, rhs.value))
            throw std::overflow_error("Multiplication overflow");
        return SelfType{value * rhs.value};
    }
    SelfType operator-(SelfType rhs) const { return (*this) + -rhs; }
    SelfType operator/(SelfType rhs) const { return SelfType{value / rhs.value}; }
    template <std::integral OtherInteger>
    SelfType operator/(OtherInteger value) const
    {
        return SelfType{this->value / value};
    }
    operator Integer() const { return value; }

    Integer value;
};

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_INTEGER_ARITHMETIC_CHECKS_H