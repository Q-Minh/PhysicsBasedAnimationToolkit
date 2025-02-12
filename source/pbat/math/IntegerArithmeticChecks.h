/**
 * @file IntegerArithmeticChecks.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file provides functions to check for integer arithmetic overflow.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_INTEGER_ARITHMETIC_CHECKS_H
#define PBAT_MATH_INTEGER_ARITHMETIC_CHECKS_H

#include <concepts>
#include <limits>
#include <stdexcept>

namespace pbat {
namespace math {

/**
 * @brief Checks if the operation \f$ a+b \f$ is not in the range of values representable by the
 * type Integer.
 * @tparam Integer The type of the integers to check for overflow.
 * @param a Left operand
 * @param b Right operand
 * @return True if the operation overflows, false otherwise.
 */
template <std::integral Integer>
bool AddOverflows(Integer a, Integer b)
{
    if (a == 0 || b == 0)
        return false;

    auto constexpr max = std::numeric_limits<Integer>::max();
    auto constexpr min = std::numeric_limits<Integer>::lowest();
    if (a < 0 && b < 0)
    {
        // adding negative numbers may underflow, i.e. a+b < min
        return a < (min - b);
    }
    if (a > 0 && b > 0)
    {
        // a+b > max <=> overflow
        return a > (max - b);
    }
    return false;
}

/**
 * @brief Checks if the operation \f$ ab \f$ is not in the range of values representable by the type
 * of Integer.
 * @tparam Integer The type of the integers to check for overflow.
 * @param a Left operand
 * @param b Right operand
 * @return True if the operation overflows, false otherwise.
 */
template <std::integral Integer>
bool MultiplyOverflows(Integer a, Integer b)
{
    if (a == 0 or b == 0)
        return false;

    auto constexpr max   = std::numeric_limits<Integer>::max();
    auto constexpr min   = std::numeric_limits<Integer>::lowest();
    bool const bSameSign = (a > 0 && b > 0) or (a < 0 && b < 0);
    if (bSameSign)
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
 * @brief Checks if the operation \f$ -a \f$ is not in the range of values representable by the type
 * of Integer.
 * @tparam Integer The type of the integers to check for overflow.
 * @param a Operand
 * @return True if the operation overflows, false otherwise.
 */
template <std::integral Integer>
bool NegationOverflows(Integer a)
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
 * @tparam Integer The type of the integer to wrap
 */
template <std::integral Integer>
struct OverflowChecked
{
    using SelfType = OverflowChecked<Integer>; ///< Instance type

    /**
     * @brief Dereference operator
     *
     * @return Underlying integer value
     */
    Integer& operator*() { return value; }
    /**
     * @brief Const dereference operator
     *
     * @return Underlying integer value
     */
    Integer const& operator*() const { return value; }
    /**
     * @brief Negation operator
     *
     * @return Negated value
     */
    SelfType operator-() const
    {
        if (NegationOverflows(value))
            throw std::overflow_error("Negation overflow");
        return SelfType{-value};
    }
    /**
     * @brief Addition operator
     *
     * @param rhs Right-hand side operand
     * @return Sum of the two operands
     */
    SelfType operator+(SelfType rhs) const
    {
        if (AddOverflows(value, rhs.value))
            throw std::overflow_error("Addition overflow");
        return SelfType{value + rhs.value};
    }
    /**
     * @brief Multiplication operator
     *
     * @param rhs Right-hand side operand
     * @return Product of the two operands
     */
    SelfType operator*(SelfType rhs) const
    {
        if (MultiplyOverflows(value, rhs.value))
            throw std::overflow_error("Multiplication overflow");
        return SelfType{value * rhs.value};
    }
    /**
     * @brief Subtraction operator
     *
     * @param rhs Right-hand side operand
     * @return Difference of the two operands
     */
    SelfType operator-(SelfType rhs) const { return (*this) + -rhs; }
    /**
     * @brief Division operator
     *
     * @param rhs Right-hand side operand
     * @return Quotient of the two operands
     */
    SelfType operator/(SelfType rhs) const { return SelfType{value / rhs.value}; }
    /**
     * @brief Division operator
     *
     * @param rhs Right-hand side operand
     * @return Quotient of the two operands
     */
    template <std::integral OtherInteger>
    SelfType operator/(OtherInteger rhs) const
    {
        return SelfType{this->value / rhs};
    }
    /**
     * @brief Cast operator to underlying type
     */
    operator Integer() const { return value; }

    Integer value; ///< Underlying integer value
};

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_INTEGER_ARITHMETIC_CHECKS_H