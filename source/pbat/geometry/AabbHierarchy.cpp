#include "AabbHierarchy.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] AabbHierarchy")
{
    using namespace pbat;
    using namespace pbat::geometry;

    static auto constexpr kDims = 3;
    AabbHierarchy<kDims> aabbHierarchy{};
}