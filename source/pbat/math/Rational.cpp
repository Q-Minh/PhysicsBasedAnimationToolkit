#include "Rational.h"

#include "IntegerArithmeticChecks.h"

#include <numeric>
#include <type_traits>

namespace pbat {
namespace math {

Rational::Rational() : a(0), b(1) {}

Rational Rational::operator+(Rational const& rhs) const
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    // a1/b1 + a2/b2 = a1*b2/b1*b2 + a2*b1/b1*b2
    auto const gcd         = std::gcd(b, rhs.b);
    auto const b1          = b / gcd;
    auto const b2          = rhs.b / gcd;
    auto const denominator = oc(b1) * oc(b2 * gcd);
    auto const numerator   = oc(a) * oc(b2) + oc(rhs.a) * oc(b1);
    return Rational(*numerator, *denominator);
}

Rational Rational::operator-(Rational const& rhs) const
{
    return (*this) + (-rhs);
}

Rational Rational::operator-() const
{
    using IntegerType = std::remove_cvref_t<decltype(a)>;
    auto const na = -OverflowChecked<IntegerType>{a};
    return Rational(*na, b);
}

Rational Rational::operator*(Rational const& rhs) const
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    // a1/b1 * a2/b2 = a1*a2 / b1*b2
    auto const num = oc(a) * oc(rhs.a);
    auto const den = oc(b) * oc(rhs.b);
    return Rational(*num, *den);
}

Rational Rational::operator/(Rational const& rhs) const
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    // (a1/b1) / (a2/b2) = a1*b2 / b1*a2
    auto const num = oc(a) * oc(rhs.b);
    auto const den = oc(b) * oc(rhs.a);
    return Rational(*num, *den);
}

bool Rational::operator==(Rational const& rhs) const
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    auto const gcd        = std::gcd(b, rhs.b);
    auto const b1         = oc(b) / gcd;
    auto const b2         = oc(rhs.b) / gcd;
    auto const numerator1 = oc(a) * b2;
    auto const numerator2 = oc(rhs.a) * b1;
    return numerator1 == numerator2;
}

bool Rational::operator<(Rational const& rhs) const
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    auto const gcd        = std::gcd(b, rhs.b);
    auto const b1         = oc(b) / gcd;
    auto const b2         = oc(rhs.b) / gcd;
    auto const numerator1 = oc(a) * b2;
    auto const numerator2 = oc(rhs.a) * b1;
    return numerator1 < numerator2;
}

bool Rational::Rebase(std::int64_t denominator)
{
    auto const oc = [](auto value) {
        using IntegerType = std::remove_cvref_t<decltype(value)>;
        return OverflowChecked<IntegerType>{value};
    };
    auto const prod       = oc(a) * oc(denominator);
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

void Rational::simplify()
{
    if (a >= 0 && b < 0)
    {
        using IntegerType = std::remove_cvref_t<decltype(a)>;
        a = -OverflowChecked<IntegerType>{a};
        b = -OverflowChecked<IntegerType>{b};
    }
    auto const gcd = std::gcd(a, b);
    a /= gcd;
    b /= gcd;
}

} // namespace math
} // namespace pbat