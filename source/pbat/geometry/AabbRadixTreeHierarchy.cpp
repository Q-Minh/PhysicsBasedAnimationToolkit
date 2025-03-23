#include "AabbRadixTreeHierarchy.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] AabbRadixTreeHierarchy")
{
    std::uint32_t* begin=nullptr;
    std::uint32_t* end=nullptr;
    cppsort::ska_sort(begin, end, [](std::uint32_t v) { return v; });
}