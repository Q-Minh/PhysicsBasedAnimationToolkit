#ifndef PBA_CORE_MATH_RATIONAL_H
#define PBA_CORE_MATH_RATIONAL_H

#include "pba/aliases.h"

#include <cstdint>
#include <tuple>

namespace pba {
namespace math {

/**
 * @brief
 *
 * WARNING: We are not checking for overflow/underflow in the implementation.
 *
 * TODO: Add checks for overflow/underflow in implementation and throw exceptions when checks fail.
 *
 */
struct Rational
{
    /**
     * @brief Construct a new Rational object
     *
     */
    Rational();
    /**
     * @brief Construct a new Rational object
     *
     * @param a
     * @param b
     */
    Rational(std::int64_t a, std::int64_t b);
    /**
     * @brief Construct a new Rational object
     *
     * @param value
     */
    Rational(std::int64_t value);
    /**
     * @brief
     * @param
     * @return
     */
    Rational operator+(Rational const&) const;
    /**
     * @brief
     * @param
     * @return
     */
    Rational operator-(Rational const&) const;
    /**
     * @brief
     * @return
     */
    Rational operator-() const;
    /**
     * @brief
     * @param
     * @return
     */
    Rational operator*(Rational const&) const;
    /**
     * @brief
     * @param
     * @return
     */
    Rational operator/(Rational const&) const;
    /**
     * @brief
     * @param
     * @return
     */
    bool operator==(Rational const&) const;
    /**
     * @brief
     * @param
     * @return
     */
    bool operator<(Rational const&) const;
    /**
     * @brief
     *
     * @param denominator
     * @return true
     * @return false
     */
    bool Rebase(std::int64_t denominator);
    /**
     * @brief
     *
     * @return Scalar
     */
    operator Scalar() const;

    std::int64_t a; ///< Numerator
    std::int64_t b; ///< Denominator
};

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_RATIONAL_H