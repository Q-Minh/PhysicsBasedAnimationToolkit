#include "pbat/math/GaussQuadrature.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/math/Concepts.h"

#include <doctest/doctest.h>

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