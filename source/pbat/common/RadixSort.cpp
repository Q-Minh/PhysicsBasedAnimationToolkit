#include "RadixSort.h"

namespace pbat::common {
} // namespace pbat::common

#include <cstdint>
#include <doctest/doctest.h>
#include <random>
#include <vector>

TEST_CASE("[common] RadixSort")
{
    using namespace pbat::common;
    SUBCASE("Sort an empty list")
    {
        RadixSortCountArray work;
        std::vector<std::uint32_t> sorted;
        std::vector<std::uint32_t> cpy;
        RadixSort(sorted, cpy, work);
        CHECK(sorted.empty());
        CHECK(cpy.empty());
    }
    SUBCASE("Sort a list of unsigned integers")
    {
        RadixSortCountArray work;
        std::vector<std::uint32_t> sorted = {5, 3, 8, 1, 2};
        std::vector<std::uint32_t> cpy    = sorted;
        RadixSort(sorted, cpy, work);
        CHECK(sorted == std::vector<std::uint32_t>{1, 2, 3, 5, 8});
    }
    SUBCASE("Sort a list of signed integers")
    {
        RadixSortCountArray work;
        std::vector<std::int32_t> sorted = {-5, 3, -8, 1, 2};
        std::vector<std::int32_t> cpy    = sorted;
        auto const fProject              = [](std::int32_t x) {
            return x + 8;
        };
        RadixSort(sorted, cpy, work, fProject);
        CHECK(sorted == std::vector<std::int32_t>{-8, -5, 1, 2, 3});
    }
    SUBCASE("Sort a list of objects by integer keys")
    {
        struct Object
        {
            int key;
            std::string value;
            bool operator<(Object const& other) const { return key < other.key; }
        };
        RadixSortCountArray work;
        std::vector<Object> sorted =
            {{5, "five"}, {3, "three"}, {8, "eight"}, {1, "one"}, {2, "two"}};
        std::vector<Object> cpy = sorted;
        auto const fProject     = [](Object const& obj) {
            return obj.key;
        };
        RadixSort(sorted, cpy, work, fProject);
        CHECK(std::is_sorted(sorted.begin(), sorted.end()));
    }
    SUBCASE("Sort random initialized unsigned integer array")
    {
        std::mt19937 rng(123);
        std::uniform_int_distribution<std::uint32_t> dist(0, 1000);
        RadixSortCountArray work;
        std::vector<std::uint32_t> sorted(100);
        for (auto& x : sorted)
            x = dist(rng);
        std::vector<std::uint32_t> cpy = sorted;
        RadixSort(sorted, cpy, work);
        CHECK(std::is_sorted(cpy.begin(), cpy.end()));
    }
}