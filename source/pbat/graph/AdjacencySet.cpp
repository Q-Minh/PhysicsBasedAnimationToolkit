#include "AdjacencySet.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <cstdint>
#include <doctest/doctest.h>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

TEST_CASE("[graph] AdjacencySet")
{
    using namespace pbat::graph;

    struct EdgeData
    {
        float weight{0.f};
        int tag{-1};
    };

    SUBCASE("Default-constructed set is empty")
    {
        AdjacencySet<EdgeData> adj;
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumVertices() == 0u);
        CHECK(adj.Data().empty());
    }

    SUBCASE("Add and Update create adjacencies")
    {
        AdjacencySet<EdgeData> adj;

        // Add some edges
        adj.Add(0u, 1u);
        adj.Add(1u, 3u);
        adj.Add(2u, 4u);

        int addedCount   = 0;
        int removedCount = 0;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                ++addedCount;
                return EdgeData{static_cast<float>(u + v), static_cast<int>(u * 10 + v)};
            },
            [&](std::uint32_t /*u*/, std::uint32_t /*v*/, EdgeData& /*w*/) { ++removedCount; });

        CHECK(addedCount == 3);
        CHECK(removedCount == 0);
        CHECK(adj.Size() == 3u);
        CHECK(adj.Data().size() == 3u);
    }

    SUBCASE("Add stores directed edges as-is")
    {
        AdjacencySet<EdgeData> adj;

        adj.Add(3u, 1u); // stored as (3,1)
        adj.Add(1u, 3u); // stored as (1,3) — distinct from (3,1)

        int addedCount = 0;
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                ++addedCount;
                edges.emplace(u, v);
                return {};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        CHECK(addedCount == 2);
        CHECK(adj.Size() == 2u);
        CHECK(edges.count({1u, 3u}) == 1u);
        CHECK(edges.count({3u, 1u}) == 1u);
    }

    SUBCASE("Update preserves existing adjacencies and detects additions/removals")
    {
        AdjacencySet<EdgeData> adj;

        // First update: add edges (0,1) and (1,2)
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 2u);

        // Second update: keep (0,1), drop (1,2), add (2,3)
        adj.Add(0u, 1u);
        adj.Add(2u, 3u);

        std::vector<std::pair<std::uint32_t, std::uint32_t>> added;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> removed;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                added.emplace_back(u, v);
                return {static_cast<float>(u + v), 1};
            },
            [&](std::uint32_t u, std::uint32_t v, EdgeData& w) {
                removed.emplace_back(u, v);
                // The data should still be valid at this point
                CHECK(w.weight == doctest::Approx(static_cast<float>(u + v)));
            });

        CHECK(adj.Size() == 2u);
        // (2,3) was added
        REQUIRE(added.size() == 1u);
        CHECK(added[0].first == 2u);
        CHECK(added[0].second == 3u);
        // (1,2) was removed
        REQUIRE(removed.size() == 1u);
        CHECK(removed[0].first == 1u);
        CHECK(removed[0].second == 2u);
    }

    SUBCASE("AdjacenciesOf iterates over neighbours of u")
    {
        AdjacencySet<EdgeData> adj;

        adj.Add(1u, 0u);
        adj.Add(1u, 2u);
        adj.Add(1u, 4u);
        adj.Add(3u, 4u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 10 + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        adj.Finalize();

        // Has() interface
        {
            CHECK(adj.Has(1u, 0u));
            CHECK(adj.Has(1u, 2u));
            CHECK(adj.Has(1u, 4u));
            CHECK(adj.Has(3u, 4u));
            CHECK_FALSE(adj.Has(2u, 1u));
            CHECK_FALSE(adj.Has(0u, 1u));
        }

        // Edges are directed: Add(1,0) stores (1,0).
        // Adjacencies with first endpoint 0: none
        {
            int count = 0;
            adj.AdjacenciesOf(0u, [&](std::uint32_t, std::uint32_t, EdgeData&) { ++count; });
            CHECK(count == 0);
        }

        // Adjacencies with first endpoint 1: {(1,0), (1,2), (1,4)}
        {
            std::vector<std::uint32_t> neighbours;
            adj.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v, EdgeData& w) {
                CHECK(u == 1u);
                neighbours.push_back(v);
                CHECK(w.weight == doctest::Approx(static_cast<float>(u * 10 + v)));
            });
            REQUIRE(neighbours.size() == 3u);
            CHECK(neighbours[0] == 0u);
            CHECK(neighbours[1] == 2u);
            CHECK(neighbours[2] == 4u);
        }

        // Adjacencies with first endpoint 3: {(3,4)}
        {
            std::vector<std::uint32_t> neighbours;
            adj.AdjacenciesOf(3u, [&](std::uint32_t u, std::uint32_t v, EdgeData& w) {
                CHECK(u == 3u);
                neighbours.push_back(v);
            });
            REQUIRE(neighbours.size() == 1u);
            CHECK(neighbours[0] == 4u);
        }

        // Adjacencies with first endpoint 2: empty
        {
            int count = 0;
            adj.AdjacenciesOf(2u, [&](std::uint32_t, std::uint32_t, EdgeData&) { ++count; });
            CHECK(count == 0);
        }
    }

    SUBCASE("Id recycling reuses data slots")
    {
        AdjacencySet<EdgeData> adj;

        // Add 3 edges
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(2u, 3u);
        adj.Update(
            [](std::uint32_t, std::uint32_t) -> EdgeData { return {1.f, 0}; },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Data().size() == 3u);

        // Remove all 3, add 2 new ones → should reuse slots
        adj.Add(0u, 4u);
        adj.Add(3u, 4u);
        adj.Update(
            [](std::uint32_t, std::uint32_t) -> EdgeData { return {2.f, 1}; },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 2u);
        CHECK(adj.Data().size() == 2u);
        // Check that the new data was actually written
        for (auto const& d : adj.Data())
        {
            CHECK(d.weight == doctest::Approx(2.f));
            CHECK(d.tag == 1);
        }
    }

    SUBCASE("Multiple updates converge correctly")
    {
        AdjacencySet<EdgeData> adj;

        // Build a triangle (0,1), (1,2), (0,2)
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(0u, 2u);
        adj.Update(
            [](std::uint32_t, std::uint32_t) -> EdgeData { return {}; },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 3u);

        // Rebuild the exact same set → no additions, no removals
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(0u, 2u);
        int addCnt = 0, rmCnt = 0;
        adj.Update(
            [&](std::uint32_t, std::uint32_t) -> EdgeData {
                ++addCnt;
                return {};
            },
            [&](std::uint32_t, std::uint32_t, EdgeData&) { ++rmCnt; });
        CHECK(addCnt == 0);
        CHECK(rmCnt == 0);
        CHECK(adj.Size() == 3u);

        // Empty update → remove everything
        adj.Update(
            [](std::uint32_t, std::uint32_t) -> EdgeData { return {}; },
            [&](std::uint32_t, std::uint32_t, EdgeData&) { ++rmCnt; });
        CHECK(rmCnt == 3);
        CHECK(adj.Size() == 0u);
        CHECK(adj.Data().empty());
    }

    SUBCASE("AppendOnly mode never removes adjacencies")
    {
        using namespace pbat::graph;
        using Options = AdjacencySetUpdateOptions;

        AdjacencySet<EdgeData> adj;

        Options appendOnly;
        appendOnly.eUpdatePolicy = Options::EUpdatePolicy::AppendOnly;

        // First update (AppendOnly): add (0,1), (1,2)
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        int addedCount   = 0;
        int removedCount = 0;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                ++addedCount;
                return {static_cast<float>(u + v), 0};
            },
            [&](std::uint32_t, std::uint32_t, EdgeData&) { ++removedCount; },
            appendOnly);
        CHECK(addedCount == 2);
        CHECK(removedCount == 0);
        CHECK(adj.Size() == 2u);

        // Second update (AppendOnly): only add (2,3), don't re-Add (0,1) or (1,2)
        // In AppendOnly mode, existing adjacencies are never removed.
        adj.Add(2u, 3u);
        addedCount   = 0;
        removedCount = 0;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                ++addedCount;
                return {static_cast<float>(u + v), 1};
            },
            [&](std::uint32_t, std::uint32_t, EdgeData&) { ++removedCount; },
            appendOnly);
        CHECK(addedCount == 1);
        CHECK(removedCount == 0);
        CHECK(adj.Size() == 3u);

        // Verify all three edges are present with correct data
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        adj.ForAll([&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            edges.emplace(u, v);
            CHECK(w.weight == doctest::Approx(static_cast<float>(u + v)));
        });
        CHECK(edges.count({0u, 1u}) == 1u);
        CHECK(edges.count({1u, 2u}) == 1u);
        CHECK(edges.count({2u, 3u}) == 1u);

        // Empty update (AppendOnly): nothing is removed
        addedCount   = 0;
        removedCount = 0;
        adj.Update(
            [&](std::uint32_t, std::uint32_t) -> EdgeData {
                ++addedCount;
                return {};
            },
            [&](std::uint32_t, std::uint32_t, EdgeData&) { ++removedCount; },
            appendOnly);
        CHECK(addedCount == 0);
        CHECK(removedCount == 0);
        CHECK(adj.Size() == 3u);

        // Duplicate Add in AppendOnly: re-adding existing edge does not create a new entry
        adj.Add(0u, 1u);
        addedCount = 0;
        adj.Update(
            [&](std::uint32_t, std::uint32_t) -> EdgeData {
                ++addedCount;
                return {};
            },
            [&](std::uint32_t, std::uint32_t, EdgeData&) {},
            appendOnly);
        CHECK(addedCount == 0);
        CHECK(adj.Size() == 3u);
    }

    SUBCASE("CompactIds shrinks indirection tables")
    {
        AdjacencySet<EdgeData> adj;

        // Round 1: add 4 edges
        adj.Add(0u, 1u);
        adj.Add(0u, 2u);
        adj.Add(1u, 3u);
        adj.Add(2u, 4u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), static_cast<int>(u * 10 + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 4u);

        // Round 2: keep only 2 of the 4 edges — this triggers 2 removals,
        // so 2 ids are "wasted" (indirection tables grow but mData stays at 2).
        adj.Add(0u, 1u);
        adj.Add(2u, 4u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), static_cast<int>(u * 10 + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 2u);
        CHECK(adj.Data().size() == 2u);

        // Round 3: add a new edge — allocates yet another fresh id
        adj.Add(0u, 1u);
        adj.Add(2u, 4u);
        adj.Add(3u, 4u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), static_cast<int>(u * 10 + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 3u);
        CHECK(adj.Data().size() == 3u);
        adj.Finalize();

        // Before compaction: indirection tables are larger than Size() because of prior
        // allocations.
        // Capture data values before compaction for later verification.
        std::vector<std::tuple<std::uint32_t, std::uint32_t, float, int>> before;
        adj.ForAll([&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            before.emplace_back(u, v, w.weight, w.tag);
        });

        // Compact
        adj.CompactIds();

        // After compaction: indirection tables have exactly Size() entries.
        // Verify that queries still return the same data.
        std::vector<std::tuple<std::uint32_t, std::uint32_t, float, int>> after;
        adj.ForAll([&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            after.emplace_back(u, v, w.weight, w.tag);
        });
        REQUIRE(before.size() == after.size());
        for (std::size_t i = 0u; i < before.size(); ++i)
        {
            CHECK(std::get<0>(before[i]) == std::get<0>(after[i]));
            CHECK(std::get<1>(before[i]) == std::get<1>(after[i]));
            CHECK(std::get<2>(before[i]) == doctest::Approx(std::get<2>(after[i])));
            CHECK(std::get<3>(before[i]) == std::get<3>(after[i]));
        }

        // AdjacenciesOf still works correctly
        adj.AdjacenciesOf(0u, [&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            CHECK(u == 0u);
            CHECK(v == 1u);
            CHECK(w.weight == doctest::Approx(1.f));
            CHECK(w.tag == 1);
        });

        // Further updates still work after compaction
        adj.Add(0u, 1u);
        adj.Add(2u, 4u);
        adj.Add(3u, 4u);
        adj.Add(1u, 2u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 100 + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        CHECK(adj.Size() == 4u);
        adj.Finalize();

        // Verify the new edge got proper data
        adj.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            if (v == 2u)
                CHECK(w.weight == doctest::Approx(102.f));
        });
    }
}

TEST_CASE("[graph] AdjacencySet<void>")
{
    using namespace pbat::graph;

    SUBCASE("Default-constructed void set is empty")
    {
        AdjacencySet<void> adj;
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumVertices() == 0u);
    }

    SUBCASE("Add and Update create adjacencies")
    {
        AdjacencySet<void> adj;

        adj.Add(0u, 1u);
        adj.Add(1u, 3u);
        adj.Add(2u, 4u);

        int addedCount   = 0;
        int removedCount = 0;
        adj.Update(
            [&](std::uint32_t, std::uint32_t) { ++addedCount; },
            [&](std::uint32_t, std::uint32_t) { ++removedCount; });

        CHECK(addedCount == 3);
        CHECK(removedCount == 0);
        CHECK(adj.Size() == 3u);
    }

    SUBCASE("Overwrite mode removes un-re-added adjacencies")
    {
        AdjacencySet<void> adj;

        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        CHECK(adj.Size() == 2u);

        // Keep (0,1), drop (1,2), add (2,3)
        adj.Add(0u, 1u);
        adj.Add(2u, 3u);

        std::vector<std::pair<std::uint32_t, std::uint32_t>> added, removed;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) { added.emplace_back(u, v); },
            [&](std::uint32_t u, std::uint32_t v) { removed.emplace_back(u, v); });

        CHECK(adj.Size() == 2u);
        REQUIRE(added.size() == 1u);
        CHECK(added[0] == std::pair{2u, 3u});
        REQUIRE(removed.size() == 1u);
        CHECK(removed[0] == std::pair{1u, 2u});
    }

    SUBCASE("AdjacenciesOf iterates over neighbours")
    {
        AdjacencySet<void> adj;

        adj.Add(1u, 0u);
        adj.Add(1u, 2u);
        adj.Add(1u, 4u);
        adj.Add(3u, 4u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        std::vector<std::uint32_t> neighbours;
        adj.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v) {
            CHECK(u == 1u);
            neighbours.push_back(v);
        });
        REQUIRE(neighbours.size() == 3u);
        CHECK(neighbours[0] == 0u);
        CHECK(neighbours[1] == 2u);
        CHECK(neighbours[2] == 4u);
    }

    SUBCASE("Merge void into void")
    {
        AdjacencySet<void> a;
        a.Add(0u, 1u);
        a.Add(1u, 2u);
        a.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        AdjacencySet<void> b;
        b.Add(1u, 2u);
        b.Add(2u, 3u);
        b.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        a.Merge(b);
        CHECK(a.Size() == 3u); // (0,1), (1,2), (2,3)
        CHECK(b.Size() == 0u);

        a.Finalize();
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        a.ForAll([&](std::uint32_t u, std::uint32_t v) { edges.emplace(u, v); });
        CHECK(edges.count({0u, 1u}) == 1u);
        CHECK(edges.count({1u, 2u}) == 1u);
        CHECK(edges.count({2u, 3u}) == 1u);
    }

    SUBCASE("Merge data-carrying set into void set (discard data)")
    {
        AdjacencySet<void> a;
        a.Add(0u, 1u);
        a.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        struct EdgeData
        {
            float weight{0.f};
        };
        AdjacencySet<EdgeData> b;
        b.Add(1u, 2u);
        b.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        a.Merge(b);
        CHECK(a.Size() == 2u);
        CHECK(b.Size() == 0u);

        a.Finalize();
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        a.ForAll([&](std::uint32_t u, std::uint32_t v) { edges.emplace(u, v); });
        CHECK(edges.count({0u, 1u}) == 1u);
        CHECK(edges.count({1u, 2u}) == 1u);
    }

    SUBCASE("Merge void set into data-carrying set (default-construct data)")
    {
        struct EdgeData
        {
            float weight{0.f};
            int tag{-1};
        };
        AdjacencySet<EdgeData> a;
        a.Add(0u, 1u);
        a.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), 42};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        AdjacencySet<void> b;
        b.Add(1u, 2u);
        b.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        a.Merge(b);
        CHECK(a.Size() == 2u);
        CHECK(b.Size() == 0u);

        a.Finalize();
        // The original edge should keep its data
        a.AdjacenciesOf(0u, [](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            CHECK(u == 0u);
            CHECK(v == 1u);
            CHECK(w.weight == doctest::Approx(1.f));
            CHECK(w.tag == 42);
        });
        // The edge from the void set gets default-constructed data
        a.AdjacenciesOf(1u, [](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            CHECK(u == 1u);
            CHECK(v == 2u);
            CHECK(w.weight == doctest::Approx(0.f));
            CHECK(w.tag == -1);
        });
    }

    SUBCASE("Reduce void sets")
    {
        std::vector<AdjacencySet<void>> sets(4);
        // Thread 0: (0,1)
        sets[0].Add(0u, 1u);
        sets[0].Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        // Thread 1: (1,2)
        sets[1].Add(1u, 2u);
        sets[1].Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        // Thread 2: (2,3)
        sets[2].Add(2u, 3u);
        sets[2].Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        // Thread 3: (3,4), (0,1) duplicate
        sets[3].Add(3u, 4u);
        sets[3].Add(0u, 1u);
        sets[3].Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        AdjacencySet<void> result;
        result.Reduce(sets.begin(), sets.end());
        result.Finalize();

        CHECK(result.Size() == 4u); // (0,1), (1,2), (2,3), (3,4)
        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        result.ForAll([&](std::uint32_t u, std::uint32_t v) { edges.emplace(u, v); });
        CHECK(edges.count({0u, 1u}) == 1u);
        CHECK(edges.count({1u, 2u}) == 1u);
        CHECK(edges.count({2u, 3u}) == 1u);
        CHECK(edges.count({3u, 4u}) == 1u);

        // All input sets should be empty
        for (auto const& s : sets)
            CHECK(s.Size() == 0u);
    }
}
