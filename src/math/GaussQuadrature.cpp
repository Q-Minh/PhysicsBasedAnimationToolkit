#include "pba/math/GaussQuadrature.h"

#include "pba/common/ConstexprFor.h"
#include "pba/math/Concepts.h"

#include <doctest/doctest.h>

TEST_CASE("[math] GaussQuadrature")
{
    using namespace pba::common;
    namespace pm = pba::math;
    ForRange<1, 3>([]<auto Dims>() {
        ForRange<1, 10>([]<auto Order> {
            CHECK(pm::CPolynomialQuadratureRule<pm::GaussLegendreQuadrature<Dims, Order>>);
            CHECK(pm::CFixedPointQuadratureRule<pm::GaussLegendreQuadrature<Dims, Order>>);
        });
    });
}