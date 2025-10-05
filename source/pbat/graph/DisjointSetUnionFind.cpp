#include "DisjointSetUnionFind.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <doctest/doctest.h>

TEST_CASE("[graph] DisjointSetUnionFind")
{
    SUBCASE("Basic operations")
    {
        pbat::graph::DisjointSetUnionFind<int> dsu(5);

        // Initially, each node is its own parent
        for (int i = 0; i < 5; ++i)
            CHECK(dsu.Find(i) == i);

        // Union some sets
        dsu.Union(0, 1);
        CHECK(dsu.Find(0) == dsu.Find(1));
        dsu.Union(2, 3);
        CHECK(dsu.Find(2) == dsu.Find(3));
        dsu.Union(1, 2);
        CHECK(dsu.Find(0) == dsu.Find(3));
        CHECK(dsu.Find(1) == dsu.Find(2));

        // Check sizes
        CHECK(dsu.Size(0) == 4);
        CHECK(dsu.Size(1) == 4);
        CHECK(dsu.Size(2) == 4);
        CHECK(dsu.Size(3) == 4);
        CHECK(dsu.Size(4) == 1);

        // Check NumVertices
        CHECK(dsu.NumVertices() == 5);
    }
    SUBCASE("All merged")
    {
        pbat::graph::DisjointSetUnionFind<int> dsu(4);

        dsu.Union(0, 1);
        dsu.Union(1, 2);
        dsu.Union(2, 3);

        int root = dsu.Find(0);
        for (int i = 0; i < 4; ++i)
            CHECK(dsu.Find(i) == root);

        CHECK(dsu.Size(0) == 4);
        CHECK(dsu.Size(3) == 4);
    }
    SUBCASE("Singletons")
    {
        pbat::graph::DisjointSetUnionFind<int> dsu(3);

        // No unions, all sets are singletons
        for (int i = 0; i < 3; ++i)
        {
            CHECK(dsu.Find(i) == i);
            CHECK(dsu.Size(i) == 1);
        }
        CHECK(dsu.NumVertices() == 3);
    }
    SUBCASE("Prepare and reuse")
    {
        pbat::graph::DisjointSetUnionFind<int> dsu;
        dsu.Prepare(2);
        CHECK(dsu.NumVertices() == 2);
        dsu.Union(0, 1);
        CHECK(dsu.Size(0) == 2);

        dsu.Prepare(3);
        for (int i = 0; i < 3; ++i)
            CHECK(dsu.Size(i) == 1);
    }
}