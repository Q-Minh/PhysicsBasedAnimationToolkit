#include "BruteSet.h"

#include <doctest/doctest.h>

TEST_CASE("[common] BruteSet")
{
    pbat::common::BruteSet<int> s;

    SUBCASE("Empty set")
    {
        CHECK_EQ(s.Size(), 0);
        CHECK(s.IsEmpty());
        CHECK(!s.Contains(1));
    }
    SUBCASE("Insert elements")
    {
        CHECK(s.Insert(1));
        CHECK(s.Insert(2));
        CHECK_FALSE(s.Insert(1)); // duplicate
        CHECK_EQ(s.Size(), 2);
        CHECK(s.Contains(1));
        CHECK(s.Contains(2));
        CHECK_FALSE(s.Contains(3));
    }
    SUBCASE("Erase elements")
    {
        s.Insert(1);
        s.Insert(2);
        CHECK(s.Remove(1));
        CHECK_FALSE(s.Contains(1));
        CHECK(s.Contains(2));
        CHECK_FALSE(s.Remove(3)); // not present
        CHECK_EQ(s.Size(), 1);
    }
    SUBCASE("Clear set")
    {
        s.Insert(1);
        s.Insert(2);
        s.Clear();
        CHECK(s.IsEmpty());
        CHECK_EQ(s.Size(), 0);
    }
}