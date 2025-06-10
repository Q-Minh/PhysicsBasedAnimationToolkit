/**
 * @file Rational.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Fixed size rational number representation using std::int64_t as numerator and denominator.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_MATH_RATIONAL_H
#define PBAT_MATH_RATIONAL_H

#include "PhysicsBasedAnimationToolkitExport.h"

#include <cstdint>
#include <pbat/Aliases.h>
#include <tuple>

namespace pbat {
namespace math {

/**
 * @brief Fixed size rational number \f$ \frac{a}{b} \f$ using std::int64_t for numerator and
 * denominator.
 *
 */
struct PBAT_API Rational
{
    /**
     * @brief Construct a new Rational object
     *
     */
    Rational();
    /**
     * @brief Construct a new Rational object
     *
     * @tparam Integer Integral type
     * @param a Numerator
     * @param b Denominator
     */
    template <std::integral Integer>
    Rational(Integer a, Integer b)
        : a(static_cast<std::int64_t>(a)), b(static_cast<std::int64_t>(b))
    {
    }
    /**
     * @brief Construct a new Rational object
     *
     * @tparam Integer Integral type
     * @param value Numerator
     * @post Denominator is set to 1
     */
    template <std::integral Integer>
    Rational(Integer value) : a(static_cast<std::int64_t>(value)), b(1)
    {
    }
    /**
     * @brief Addition operation
     * @param other Right-hand side operand
     * @return Result of addition
     */
    Rational operator+(Rational const& other) const;
    /**
     * @brief Subtraction operation
     * @param other Right-hand side operand
     * @return Result of subtraction
     */
    Rational operator-(Rational const& other) const;
    /**
     * @brief Negation operation
     * @return Result of negation
     */
    Rational operator-() const;
    /**
     * @brief Multiplication operation
     * @param other Right-hand side operand
     * @return Result of multiplication
     */
    Rational operator*(Rational const& other) const;
    /**
     * @brief Division operation
     * @param other Right-hand side operand
     * @return Result of division
     */
    Rational operator/(Rational const& other) const;
    /**
     * @brief Equality operation
     * @param other Right-hand side operand
     * @return true if equal
     */
    bool operator==(Rational const& other) const;
    /**
     * @brief Less-than operation
     * @param other Right-hand side operand
     * @return true if not equal
     */
    bool operator<(Rational const& other) const;
    /**
     * @brief Change internal rational representation to have %denominator denominator.
     *
     * @param denominator New denominator
     * @return true if successful
     */
    bool Rebase(std::int64_t denominator);
    /**
     * @brief Cast to double
     *
     * @return 
     */
    explicit operator double() const;
    /**
     * @brief Cast to float
     *
     * @return 
     */
    explicit operator float() const;
    /**
     * @brief Attempts to reduce magnitude of \f$ a,b \f$ by eliminating common divisor
     */
    void simplify();

    std::int64_t a; ///< Numerator
    std::int64_t b; ///< Denominator
};

/**
 * @brief Subtraction operation between Rational and integral type
 *
 * @tparam Integer Integral type
 * @param a Left operand
 * @param b Right operand
 * @return Result of subtraction
 */
template <std::integral Integer>
inline Rational operator-(Integer a, Rational const& b)
{
    return (-b) + a;
}

/**
 * @brief Addition operation between Rational and integral type
 *
 * @tparam Integer Integral type
 * @param a Left operand
 * @param b Right operand
 * @return Result of addition
 */
template <std::integral Integer>
inline Rational operator+(Integer a, Rational const& b)
{
    return b + a;
}

/**
 * @brief Multiplication operation between Rational and integral type
 *
 * @tparam Integer Integral type
 * @param a Left operand
 * @param b Right operand
 * @return Result of multiplication
 */
template <std::integral Integer>
inline Rational operator*(Integer a, Rational const& b)
{
    return b * a;
}

/**
 * @brief Division operation between Rational and integral type
 *
 * @tparam Integer Integral type
 * @param a Left operand
 * @param b Right operand
 * @return Result of division
 */
template <std::integral Integer>
inline Rational operator/(Integer a, Rational const& b)
{
    return Rational{a} / b;
}

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_RATIONAL_H