#include "CountingSort.h"

#include <algorithm>
#include <doctest/doctest.h>
#include <random>
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
        pbat::common::CountingSort(objects, workspace, fKey);
        CHECK(objects.empty());
    }

    SUBCASE("Sort a single element list")
    {
        std::vector<Object> objects = {{5, "single"}};
        auto min                    = fAllocateWorkspace(objects);
        pbat::common::CountingSort(objects, workspace, fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Sort a list with multiple elements")
    {
        std::vector<Object> objects =
            {{3, "three"}, {1, "one"}, {4, "four"}, {2, "two"}, {5, "five"}};
        auto min = fAllocateWorkspace(objects);
        std::vector<Object> expected =
            {{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}};
        pbat::common::CountingSort(objects, workspace, fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Sort a list with duplicate keys")
    {
        std::vector<Object> objects =
            {{3, "three"}, {1, "one"}, {3, "three-again"}, {2, "two"}, {1, "one-again"}};
        std::vector<Object> expected =
            {{1, "one"}, {1, "one-again"}, {2, "two"}, {3, "three"}, {3, "three-again"}};
        auto min = fAllocateWorkspace(objects);
        pbat::common::CountingSort(objects, workspace, fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Sort a list with all elements having the same key")
    {
        std::vector<Object> objects  = {{1, "one"}, {1, "one-again"}, {1, "one-more"}};
        std::vector<Object> expected = {{1, "one"}, {1, "one-again"}, {1, "one-more"}};
        auto min                     = fAllocateWorkspace(objects);
        pbat::common::CountingSort(objects, workspace, fKey);
        CHECK(std::is_sorted(objects.begin(), objects.end()));
    }

    SUBCASE("Random vector")
    {
        SUBCASE("signed integer")
        {
            std::vector<int> objects{};
            std::mt19937 rng(123);
            std::uniform_int_distribution<int> dist(-100, 100);
            for (int i = 0; i < 1000; ++i)
                objects.push_back(dist(rng));
            std::vector<int> workspace{};
            workspace.resize(
                static_cast<std::size_t>(
                    *std::ranges::max_element(objects) - *std::ranges::min_element(objects) + 1));
            pbat::common::CountingSort(objects, workspace);
            CHECK(std::is_sorted(objects.begin(), objects.end()));
        }
        SUBCASE("unsigned integer")
        {
            std::vector<unsigned int> objects{};
            std::mt19937 rng(123);
            std::uniform_int_distribution<unsigned int> dist(0, 200);
            for (int i = 0; i < 1000; ++i)
                objects.push_back(dist(rng));
            std::vector<unsigned int> workspace{};
            workspace.resize(
                static_cast<std::size_t>(
                    *std::ranges::max_element(objects) - *std::ranges::min_element(objects) + 1));
            pbat::common::CountingSort(objects, workspace);
            CHECK(std::is_sorted(objects.begin(), objects.end()));
        }
    }
}