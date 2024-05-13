#include "pba/math/SymmetricQuadratureRules.h"

#include "pba/common/ConstexprFor.h"
#include "pba/math/Concepts.h"

#include <doctest/doctest.h>

TEST_CASE("[math] SymmetricQuadratureRules")
{
    using namespace pba::common;
    namespace pm = pba::math;
    ForTypes<
        pm::Quad1DP1,
        pm::Quad1DP3,
        pm::Quad1DP5,
        pm::Quad1DP7,
        pm::Quad1DP9,
        pm::Quad1DP11,
        pm::Quad1DP13,
        pm::Quad1DP15,
        pm::Quad1DP17,
        pm::Quad1DP19,
        pm::Quad1DP21,
        pm::Quad2DP1,
        pm::Quad2DP2,
        pm::Quad2DP3,
        pm::Quad2DP4,
        pm::Quad2DP5,
        pm::Quad2DP6,
        pm::Quad2DP7,
        pm::Quad2DP8,
        pm::Quad2DP9,
        pm::Quad2DP10,
        pm::Quad2DP11,
        pm::Quad2DP12,
        pm::Quad2DP13,
        pm::Quad2DP14,
        pm::Quad2DP15,
        pm::Quad2DP16,
        pm::Quad2DP17,
        pm::Quad2DP18,
        pm::Quad2DP19,
        pm::Quad2DP20,
        pm::Quad2DP21,
        pm::Quad2DP22,
        pm::Quad2DP23,
        pm::Quad2DP24,
        pm::Quad2DP25,
        pm::Quad2DP26,
        pm::Quad2DP27,
        pm::Quad2DP28,
        pm::Quad2DP29,
        pm::Quad3DP1,
        pm::Quad3DP2,
        pm::Quad3DP3,
        pm::Quad3DP4,
        pm::Quad3DP5,
        pm::Quad3DP6,
        pm::Quad3DP7,
        pm::Quad3DP8,
        pm::Quad3DP9,
        pm::Quad3DP10,
        pm::Quad3DP11,
        pm::Quad3DP10,
        pm::Quad3DP11,
        pm::Quad3DP12,
        pm::Quad3DP13,
        pm::Quad3DP14,
        pm::Quad3DP15,
        pm::Quad3DP16,
        pm::Quad3DP17,
        pm::Quad3DP18,
        pm::Quad3DP19,
        pm::Quad3DP20>([]<class T>() {
        CHECK(pm::CPolynomialQuadratureRule<T>);
        CHECK(pm::CFixedPointQuadratureRule<T>);
    });
}