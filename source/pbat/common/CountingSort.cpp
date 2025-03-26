#include "CountingSort.h"

#include <algorithm>
#include <doctest/doctest.h>
#include <string>
#include <vector>

TEST_CASE("[common] CountingSort")
{
    struct Object
    {
        int key;
        std::string value;
        bool operator<(Object const& other) const { return key < other.key; }
    };
    auto const fKey = [](Object const& obj) {
        return obj.key;
    };
    std::vector<int> workspace{};
    auto const fAllocateWorkspace = [&](std::vector<Object> const& objects) {
        auto const [min, max] = std::minmax_element(
            objects.begin(),
            objects.end(),
            [](Object const& o1, Object const& o2) { return o1.key < o2.key; });
        workspace.resize(static_cast<std::size_t>(max->key - min->key + 1));
        return min->key;
    };

    SUBCASE("Sort an empty list")
    {
        std::vector<Object> objects;
        pbat::common::CountingSort(
            workspace.begin(),
            workspace.end(),
            objects.begin(),
            objects.end(),
            0,
            fKey);
        CHECK(objects.empty());
    }

    SUBCASE("Sort a single element list")
    {
        std::vector<Object> objects = {{5, "single"}};
        auto min                    = fAllocateWorkspace(objects);
        pbat::common::CountingSort(
            workspace.begin(),
            workspace.end(),
            objects.begin(),
            objects.end(),
            min,
            fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Sort a list with multiple elements")
    {
        std::vector<Object> objects =
            {{3, "three"}, {1, "one"}, {4, "four"}, {2, "two"}, {5, "five"}};
        auto min = fAllocateWorkspace(objects);
        std::vector<Object> expected =
            {{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}};
        pbat::common::CountingSort(
            workspace.begin(),
            workspace.end(),
            objects.begin(),
            objects.end(),
            min,
            fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Sort a list with duplicate keys")
    {
        std::vector<Object> objects =
            {{3, "three"}, {1, "one"}, {3, "three-again"}, {2, "two"}, {1, "one-again"}};
        std::vector<Object> expected =
            {{1, "one"}, {1, "one-again"}, {2, "two"}, {3, "three"}, {3, "three-again"}};
        auto min = fAllocateWorkspace(objects);
        pbat::common::CountingSort(
            workspace.begin(),
            workspace.end(),
            objects.begin(),
            objects.end(),
            min,
            fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));

        // Compute prefix sum from sorted keys
        pbat::common::PrefixSumFromSortedKeys(
            objects.begin(),
            objects.end(),
            workspace.begin(),
            workspace.end(),
            fKey);
        CHECK_EQ(workspace[0], 2);
        CHECK_EQ(workspace[1], 3);
        CHECK_EQ(workspace[2], 5);
    }

    SUBCASE("Sort a list with all elements having the same key")
    {
        std::vector<Object> objects  = {{1, "one"}, {1, "one-again"}, {1, "one-more"}};
        std::vector<Object> expected = {{1, "one"}, {1, "one-again"}, {1, "one-more"}};
        auto min                     = fAllocateWorkspace(objects);
        pbat::common::CountingSort(
            workspace.begin(),
            workspace.end(),
            objects.begin(),
            objects.end(),
            min,
            fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }
}