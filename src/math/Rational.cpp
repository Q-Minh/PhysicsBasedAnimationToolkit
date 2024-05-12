#include "pba/math/Rational.h"

#include "pba/math/IntegerArithmeticChecks.h"

#include <numeric>

namespace pba {
namespace math {

Rational::Rational() : a(0), b(1) {}

Rational::Rational(std::int64_t a, std::int64_t b) : a(a), b(b)
{
    if (this->a >= 0 && this->b < 0)
    {
        this->a = -a;
        this->b = -b;
    }
    auto const gcd = std::gcd(a, b);
    this->a /= gcd;
    this->b /= gcd;
}

Rational::Rational(std::int64_t value) : a(value), b(1) {}

Rational Rational::operator+(Rational const& rhs) const
{
    // a1/b1 + a2/b2 = a1*b2/b1*b2 + a2*b1/b1*b2
    auto const gcd         = std::gcd(b, rhs.b);
    auto const b1          = b / gcd;
    auto const b2          = rhs.b / gcd;
    auto const denominator = b1 * b2 * gcd;
    auto const numerator   = a * b2 + rhs.a * b1;
    return Rational(numerator, denominator);
}

Rational Rational::operator-(Rational const& rhs) const
{
    return (*this) + (-rhs);
}

Rational Rational::operator-() const
{
    return Rational(-a, b);
}

Rational Rational::operator*(Rational const& rhs) const
{
    // a1/b1 * a2/b2 = a1*a2 / b1*b2
    return Rational(a * rhs.a, b * rhs.b);
}

Rational Rational::operator/(Rational const& rhs) const
{
    // (a1/b1) / (a2/b2) = a1*b2 / b1*a2
    return Rational(a * rhs.b, b * rhs.a);
}

bool Rational::operator==(Rational const& rhs) const
{
    auto const gcd        = std::gcd(b, rhs.b);
    auto const b1         = b / gcd;
    auto const b2         = rhs.b / gcd;
    auto const numerator1 = a * b2;
    auto const numerator2 = rhs.a * b1;
    return numerator1 == numerator2;
}

bool Rational::operator<(Rational const& rhs) const
{
    auto const gcd        = std::gcd(b, rhs.b);
    auto const b1         = b / gcd;
    auto const b2         = rhs.b / gcd;
    auto const numerator1 = a * b2;
    auto const numerator2 = rhs.a * b1;
    return numerator1 < numerator2;
}

bool Rational::Rebase(std::int64_t denominator)
{
    auto const prod       = a * denominator;
    bool const can_divide = (prod % b) == 0;
    if (!can_divide)
        return false;
    a = prod / b;
    b = denominator;
    return true;
}

Rational::operator Scalar() const
{
    return static_cast<Scalar>(a) / static_cast<Scalar>(b);
}

} // namespace math
} // namespace pba