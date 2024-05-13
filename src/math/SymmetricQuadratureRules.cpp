#include "pba/math/SymmetricQuadratureRules.h"

#include "pba/common/ConstexprFor.h"
#include "pba/math/Concepts.h"

#include <doctest/doctest.h>

TEST_CASE("[math] SymmetricQuadratureRules")
{
    using namespace pba::common;
    namespace pm = pba::math;
    ForRange<1, 22>([]<auto Order>() {
        if constexpr (Order % 2 == 1)
        {
            CHECK(pm::CPolynomialQuadratureRule<pm::SymmetricPolynomialQuadratureRule<1, Order>>);
            CHECK(pm::CFixedPointQuadratureRule<pm::SymmetricPolynomialQuadratureRule<1, Order>>);
        }
    });
    ForRange<1, 30>([]<auto Order>() {
        CHECK(pm::CPolynomialQuadratureRule<pm::SymmetricPolynomialQuadratureRule<2, Order>>);
        CHECK(pm::CFixedPointQuadratureRule<pm::SymmetricPolynomialQuadratureRule<2, Order>>);
    });
    ForRange<1, 21>([]<auto Order>() {
        CHECK(pm::CPolynomialQuadratureRule<pm::SymmetricPolynomialQuadratureRule<3, Order>>);
        CHECK(pm::CFixedPointQuadratureRule<pm::SymmetricPolynomialQuadratureRule<3, Order>>);
    });
}