#include "GaussQuadrature.h"

#include "Concepts.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>

TEST_CASE("[math] GaussQuadrature")
{
    using namespace pbat::common;
    namespace pm = pbat::math;
    ForRange<1, 3>([]<auto Dims>() {
        ForRange<1, 10>([]<auto Order> {
            CHECK(pm::CPolynomialQuadratureRule<pm::GaussLegendreQuadrature<Dims, Order>>);
            CHECK(pm::CFixedPointQuadratureRule<pm::GaussLegendreQuadrature<Dims, Order>>);
        });
    });
}