#include "pbat/math/PolynomialBasis.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/math/Concepts.h"

#include <doctest/doctest.h>

using namespace pbat;

TEST_CASE("[math] PolynomialBasis")
{
    common::ForRange<1, 3>([]<auto Dims>() {
        common::ForRange<1, 4>([]<auto Order>() {
            CHECK(math::CPolynomialBasis<math::MonomialBasis<Dims, Order>>);
            CHECK(math::CPolynomialBasis<math::OrthonormalPolynomialBasis<Dims, Order>>);
            CHECK(math::CVectorPolynomialBasis<math::DivergenceFreePolynomialBasis<Dims, Order>>);
        });
    });
}