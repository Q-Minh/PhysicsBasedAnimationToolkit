#include "AabbKdTreeHierarchy.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] AabbKdTreeHierarchy")
{
    using namespace pbat;
    using namespace pbat::geometry;

    static auto constexpr kDims = 3;
    AabbKdTreeHierarchy<kDims> aabbHierarchy{};
}