#include "pba/math/PolynomialBasis.h"

#include "pba/common/ConstexprFor.h"
#include "pba/math/Concepts.h"

#include <doctest/doctest.h>

using namespace pba;

TEST_CASE("[math] PolynomialBasis")
{
    common::ForRange<1, 3>([]<auto Dims>() {
        common::ForRange<1, 4>([]<auto Order>() {
            CHECK(math::PolynomialBasis<math::MonomialBasis<Dims, Order>>);
            CHECK(math::PolynomialBasis<math::OrthonormalPolynomialBasis<Dims, Order>>);
            CHECK(math::VectorPolynomialBasis<math::DivergenceFreePolynomialBasis<Dims, Order>>);
        });
    });
}