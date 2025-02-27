#include "Basis.h"

#include "Concepts.h"
#include "pbat/common/ConstexprFor.h"

#include <doctest/doctest.h>

TEST_CASE("[math][polynomial] Basis")
{
    using namespace pbat::math::polynomial;
    using pbat::common::ForRange;
    ForRange<1, 3>([]<auto Dims>() {
        ForRange<1, 4>([]<auto Order>() {
            CHECK(CBasis<MonomialBasis<Dims, Order>>);
            CHECK(CBasis<OrthonormalBasis<Dims, Order>>);
            CHECK(CVectorBasis<DivergenceFreeBasis<Dims, Order>>);
        });
    });
}