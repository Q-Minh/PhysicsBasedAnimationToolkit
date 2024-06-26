#ifndef PBAT_MATH_RATIONAL_H
#define PBAT_MATH_RATIONAL_H

#include "PhysicsBasedAnimationToolkitExport.h"

#include <cstdint>
#include <pbat/Aliases.h>
#include <tuple>

namespace pbat {
namespace math {

/**
 * @brief Fixed size rational number representation using std::int64_t as numerator and denominator.
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
     * @param a
     * @param b
     */
    template <std::integral Integer>
    Rational(Integer a, Integer b)
        : a(static_cast<std::int64_t>(a)), b(static_cast<std::int64_t>(b))
    {
    }
    /**
     * @brief Construct a new Rational object
     *
     * @param value
     */
    template <std::integral Integer>
    Rational(Integer value) : a(static_cast<std::int64_t>(value)), b(1)
    {
    }
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
    explicit operator Scalar() const;
    /**
     * @brief Attempts to reduce magnitude of a,b by eliminating common divisor
     */
    void simplify();

    std::int64_t a; ///< Numerator
    std::int64_t b; ///< Denominator
};

template <std::integral Integer>
inline Rational operator-(Integer a, Rational const& b)
{
    return (-b) + a;
}

template <std::integral Integer>
inline Rational operator+(Integer a, Rational const& b)
{
    return b + a;
}

template <std::integral Integer>
inline Rational operator*(Integer a, Rational const& b)
{
    return b * a;
}

template <std::integral Integer>
inline Rational operator/(Integer a, Rational const& b)
{
    return Rational{a} / b;
}

} // namespace math
} // namespace pbat

#endif // PBAT_MATH_RATIONAL_H