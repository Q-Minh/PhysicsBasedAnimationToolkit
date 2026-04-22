#include "DenseAdjacencySet.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <cstdint>
#include <doctest/doctest.h>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

TEST_CASE("[graph] DenseAdjacencySet")
{
    using namespace pbat::graph;

    struct EdgeData
    {
        float weight{0.f};
        int tag{-1};
    };

    SUBCASE("Default-constructed set is empty")
    {
        DenseAdjacencySet<EdgeData> adj;
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumVertices() == 0u);
        CHECK(adj.Data().empty());
    }

    SUBCASE("Add and Update create adjacencies")
    {
        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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

        DenseAdjacencySet<EdgeData> adj;

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
        DenseAdjacencySet<EdgeData> adj;

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

TEST_CASE("[graph] DenseAdjacencySet<void>")
{
    using namespace pbat::graph;

    SUBCASE("Default-constructed void set is empty")
    {
        DenseAdjacencySet<void> adj;
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumVertices() == 0u);
    }

    SUBCASE("Add and Update create adjacencies")
    {
        DenseAdjacencySet<void> adj;

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
        DenseAdjacencySet<void> adj;

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
        DenseAdjacencySet<void> adj;

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
        DenseAdjacencySet<void> a;
        a.Add(0u, 1u);
        a.Add(1u, 2u);
        a.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        DenseAdjacencySet<void> b;
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
        DenseAdjacencySet<void> a;
        a.Add(0u, 1u);
        a.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});

        struct EdgeData
        {
            float weight{0.f};
        };
        DenseAdjacencySet<EdgeData> b;
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
        DenseAdjacencySet<EdgeData> a;
        a.Add(0u, 1u);
        a.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), 42};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        DenseAdjacencySet<void> b;
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
        std::vector<DenseAdjacencySet<void>> sets(4);
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

        DenseAdjacencySet<void> result;
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

    SUBCASE("Counting sort in Update")
    {
        using Options = AdjacencySetUpdateOptions;

        DenseAdjacencySet<void> adj;

        // Add edges in reverse order (worst case for comparison sort, but counting sort is O(n))
        adj.Add(3u, 2u);
        adj.Add(3u, 1u);
        adj.Add(2u, 3u);
        adj.Add(1u, 0u);
        adj.Add(0u, 2u);
        adj.Add(0u, 1u);
        adj.Add(2u, 0u);

        // Use counting sort by specifying nSourceVertices and nTargetVertices
        Options opts;
        opts.nSourceVertices = 4;
        opts.nTargetVertices = 4;

        int addedCount = 0;
        adj.Update(
            [&](std::uint32_t, std::uint32_t) { ++addedCount; },
            [](std::uint32_t, std::uint32_t) {},
            opts);

        CHECK(addedCount == 7);
        CHECK(adj.Size() == 7u);
        adj.Finalize();

        // Verify adjacencies are sorted lexicographically by (u, v)
        std::vector<std::pair<std::uint32_t, std::uint32_t>> sorted;
        adj.ForAll([&](std::uint32_t u, std::uint32_t v) { sorted.emplace_back(u, v); });

        REQUIRE(sorted.size() == 7u);
        // Expected order: (0,1), (0,2), (1,0), (2,0), (2,3), (3,1), (3,2)
        CHECK(sorted[0] == std::pair{0u, 1u});
        CHECK(sorted[1] == std::pair{0u, 2u});
        CHECK(sorted[2] == std::pair{1u, 0u});
        CHECK(sorted[3] == std::pair{2u, 0u});
        CHECK(sorted[4] == std::pair{2u, 3u});
        CHECK(sorted[5] == std::pair{3u, 1u});
        CHECK(sorted[6] == std::pair{3u, 2u});

        // Verify AdjacenciesOf works correctly
        std::vector<std::uint32_t> neighbours;
        adj.AdjacenciesOf(0u, [&](std::uint32_t, std::uint32_t v) { neighbours.push_back(v); });
        REQUIRE(neighbours.size() == 2u);
        CHECK(neighbours[0] == 1u);
        CHECK(neighbours[1] == 2u);

        neighbours.clear();
        adj.AdjacenciesOf(3u, [&](std::uint32_t, std::uint32_t v) { neighbours.push_back(v); });
        REQUIRE(neighbours.size() == 2u);
        CHECK(neighbours[0] == 1u);
        CHECK(neighbours[1] == 2u);
    }
}

TEST_CASE("[graph] DenseReverseAdjacencySetView")
{
    using namespace pbat::graph;

    SUBCASE("Default-constructed view is empty")
    {
        DenseReverseAdjacencySetView<void> view;
        CHECK(view.Size() == 0u);
        CHECK(view.NumVertices() == 0u);
        CHECK(view.Empty());
    }

    SUBCASE("Update from void DenseAdjacencySet")
    {
        DenseAdjacencySet<void> adj;
        adj.Add(0u, 1u);
        adj.Add(0u, 2u);
        adj.Add(1u, 2u);
        adj.Add(2u, 0u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<void> view;
        view.Update(adj);

        CHECK(view.Size() == adj.Size());
        CHECK(view.NumVertices() == 3u);

        // Degree checks
        CHECK(view.Degree(0u) == 1u); // (2,0) → one source pointing to 0
        CHECK(view.Degree(1u) == 1u); // (0,1) → one source pointing to 1
        CHECK(view.Degree(2u) == 2u); // (0,2), (1,2) → two sources pointing to 2
    }

    SUBCASE("AdjacenciesOf iterates reverse neighbours")
    {
        DenseAdjacencySet<void> adj;
        // Edges: 0→1, 0→2, 1→2, 2→0, 3→2
        adj.Add(0u, 1u);
        adj.Add(0u, 2u);
        adj.Add(1u, 2u);
        adj.Add(2u, 0u);
        adj.Add(3u, 2u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<void> view;
        view.Update(adj);

        // Sources pointing to target 2: 0, 1, 3
        std::vector<std::uint32_t> sources;
        view.AdjacenciesOf(2u, [&](std::uint32_t u, std::uint32_t v) {
            CHECK(v == 2u);
            sources.push_back(u);
        });
        REQUIRE(sources.size() == 3u);
        // Should be sorted by u within each target bucket
        CHECK(sources[0] == 0u);
        CHECK(sources[1] == 1u);
        CHECK(sources[2] == 3u);

        // Sources pointing to target 0: 2
        sources.clear();
        view.AdjacenciesOf(0u, [&](std::uint32_t u, std::uint32_t v) {
            CHECK(v == 0u);
            sources.push_back(u);
        });
        REQUIRE(sources.size() == 1u);
        CHECK(sources[0] == 2u);

        // Sources pointing to target 1: 0
        sources.clear();
        view.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v) {
            CHECK(v == 1u);
            sources.push_back(u);
        });
        REQUIRE(sources.size() == 1u);
        CHECK(sources[0] == 0u);
    }

    SUBCASE("Has checks existence")
    {
        DenseAdjacencySet<void> adj;
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(2u, 0u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<void> view;
        view.Update(adj);

        // Original edges exist in the view
        CHECK(view.Has(0u, 1u));
        CHECK(view.Has(1u, 2u));
        CHECK(view.Has(2u, 0u));

        // Non-existent edges
        CHECK_FALSE(view.Has(1u, 0u)); // reverse of (0,1) doesn't exist
        CHECK_FALSE(view.Has(0u, 2u));
        CHECK_FALSE(view.Has(3u, 0u)); // vertex 3 doesn't exist
    }

    SUBCASE("ForAll iterates all adjacencies")
    {
        DenseAdjacencySet<void> adj;
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(2u, 0u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<void> view;
        view.Update(adj);

        std::set<std::pair<std::uint32_t, std::uint32_t>> edges;
        view.ForAll([&](std::uint32_t u, std::uint32_t v) { edges.emplace(u, v); });

        CHECK(edges.size() == 3u);
        CHECK(edges.count({0u, 1u}) == 1u);
        CHECK(edges.count({1u, 2u}) == 1u);
        CHECK(edges.count({2u, 0u}) == 1u);
    }

    SUBCASE("Clear resets the view")
    {
        DenseAdjacencySet<void> adj;
        adj.Add(0u, 1u);
        adj.Update([](std::uint32_t, std::uint32_t) {}, [](std::uint32_t, std::uint32_t) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<void> view;
        view.Update(adj);
        CHECK(view.Size() == 1u);

        view.Clear();
        CHECK(view.Size() == 0u);
        CHECK(view.NumVertices() == 0u);
        CHECK(view.Empty());
    }
}

TEST_CASE("[graph] DenseReverseAdjacencySetView with data")
{
    using namespace pbat::graph;

    struct EdgeData
    {
        float weight{0.f};
        int tag{-1};
    };

    SUBCASE("Update from data-carrying DenseAdjacencySet")
    {
        DenseAdjacencySet<EdgeData> adj;
        adj.Add(0u, 1u);
        adj.Add(0u, 2u);
        adj.Add(1u, 2u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 10 + v), static_cast<int>(u + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<EdgeData> view;
        view.Update(adj);

        CHECK(view.Size() == 3u);
        CHECK(view.NumVertices() == 3u);

        // Verify data indices are correct
        view.AdjacenciesOf(2u, [&](std::uint32_t u, std::uint32_t v, std::uint32_t c) {
            CHECK(v == 2u);
            EdgeData const& data = adj.Data()[c];
            CHECK(data.weight == doctest::Approx(static_cast<float>(u * 10 + v)));
            CHECK(data.tag == static_cast<int>(u + v));
        });
    }

    SUBCASE("Has with data index")
    {
        DenseAdjacencySet<EdgeData> adj;
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<EdgeData> view;
        view.Update(adj);

        std::uint32_t c{};
        CHECK(view.Has(0u, 1u, &c));
        CHECK(adj.Data()[c].weight == doctest::Approx(1.f));

        CHECK(view.Has(1u, 2u, &c));
        CHECK(adj.Data()[c].weight == doctest::Approx(3.f));
    }

    SUBCASE("ForAll with data index")
    {
        DenseAdjacencySet<EdgeData> adj;
        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        adj.Add(2u, 0u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 100 + v), static_cast<int>(u + v)};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});
        adj.Finalize();

        DenseReverseAdjacencySetView<EdgeData> view;
        view.Update(adj);

        int count = 0;
        view.ForAll([&](std::uint32_t u, std::uint32_t v, std::uint32_t c) {
            EdgeData const& data = adj.Data()[c];
            CHECK(data.weight == doctest::Approx(static_cast<float>(u * 100 + v)));
            CHECK(data.tag == static_cast<int>(u + v));
            ++count;
        });
        CHECK(count == 3);
    }
}

