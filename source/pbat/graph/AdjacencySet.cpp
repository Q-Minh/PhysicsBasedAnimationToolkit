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

    SUBCASE("Construct initialises empty set")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(10u);
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumVertices() == 10u);
        CHECK(adj.Data().empty());
    }

    SUBCASE("Add and Update create adjacencies")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

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

    SUBCASE("Canonicalisation: Add(v,u) == Add(u,v) when u < v")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

        adj.Add(3u, 1u); // should become (1,3)
        adj.Add(1u, 3u); // duplicate after canonicalisation

        int addedCount = 0;
        adj.Update(
            [&](std::uint32_t u, std::uint32_t v) -> EdgeData {
                ++addedCount;
                CHECK(u == 1u);
                CHECK(v == 3u);
                return {};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        CHECK(addedCount == 1);
        CHECK(adj.Size() == 1u);
    }

    SUBCASE("Update preserves existing adjacencies and detects additions/removals")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

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
        adj.Construct(5u);

        adj.Add(1u, 0u);
        adj.Add(1u, 2u);
        adj.Add(1u, 4u);
        adj.Add(3u, 4u);
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 10 + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        // Since canonicalisation makes u < v, edge (1,0) becomes (0,1).
        // So adjacencies with first endpoint 0: {(0,1)}
        {
            std::vector<std::uint32_t> neighbours;
            adj.AdjacenciesOf(0u, [&](std::uint32_t u, std::uint32_t v, EdgeData& w) {
                CHECK(u == 0u);
                neighbours.push_back(v);
                CHECK(w.weight == doctest::Approx(static_cast<float>(u * 10 + v)));
            });
            REQUIRE(neighbours.size() == 1u);
            CHECK(neighbours[0] == 1u);
        }

        // Adjacencies with first endpoint 1: {(1,2), (1,4)}
        {
            std::vector<std::uint32_t> neighbours;
            adj.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v, EdgeData& w) {
                CHECK(u == 1u);
                neighbours.push_back(v);
            });
            REQUIRE(neighbours.size() == 2u);
            CHECK(neighbours[0] == 2u);
            CHECK(neighbours[1] == 4u);
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

        // Adjacencies with first endpoint 2: empty (all edges with 2 have 2 as the larger vertex)
        {
            int count = 0;
            adj.AdjacenciesOf(2u, [&](std::uint32_t, std::uint32_t, EdgeData&) { ++count; });
            CHECK(count == 0);
        }
    }

    SUBCASE("Id recycling reuses data slots")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

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
        adj.Construct(6u);

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
        adj.Construct(6u);

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

    SUBCASE("InverseAdjacenciesOf iterates neighbours keyed by v")
    {
        using namespace pbat::graph;
        using Options = AdjacencySetUpdateOptions;

        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

        // Edges: (0,1), (0,3), (1,2), (1,4), (2,4)
        // After canonicalisation, u < v always holds.
        adj.Add(0u, 1u);
        adj.Add(0u, 3u);
        adj.Add(1u, 2u);
        adj.Add(1u, 4u);
        adj.Add(2u, 4u);

        Options opts;
        opts.bBuildInverse = true;
        adj.Update(
            [](std::uint32_t u, std::uint32_t v) -> EdgeData {
                return {static_cast<float>(u * 10 + v), 0};
            },
            [](std::uint32_t, std::uint32_t, EdgeData&) {},
            opts);

        CHECK(adj.Size() == 5u);

        // AdjacenciesOf(0) = {(0,1), (0,3)}  — forward: u == 0
        {
            std::vector<std::uint32_t> neighbours;
            adj.AdjacenciesOf(0u, [&](std::uint32_t u, std::uint32_t v, EdgeData const&) {
                CHECK(u == 0u);
                neighbours.push_back(v);
            });
            REQUIRE(neighbours.size() == 2u);
            CHECK(neighbours[0] == 1u);
            CHECK(neighbours[1] == 3u);
        }

        // InverseAdjacenciesOf(4) = {(4,1), (4,2)}  — inverse: v == 4
        // These come from forward edges (1,4) and (2,4).
        {
            std::vector<std::uint32_t> sources;
            adj.InverseAdjacenciesOf(4u, [&](std::uint32_t v, std::uint32_t u, EdgeData const& w) {
                CHECK(v == 4u);
                sources.push_back(u);
                CHECK(w.weight == doctest::Approx(static_cast<float>(u * 10 + v)));
            });
            REQUIRE(sources.size() == 2u);
            // Within the v==4 bucket, sources should appear in u-order (1, 2)
            // because mAdjacencies is sorted by (u,v) and the counting sort is stable.
            CHECK(sources[0] == 1u);
            CHECK(sources[1] == 2u);
        }

        // InverseAdjacenciesOf(1) = {(1,0)}  — only edge with v==1 is (0,1)
        {
            std::vector<std::uint32_t> sources;
            adj.InverseAdjacenciesOf(1u, [&](std::uint32_t v, std::uint32_t u, EdgeData const& w) {
                CHECK(v == 1u);
                sources.push_back(u);
            });
            REQUIRE(sources.size() == 1u);
            CHECK(sources[0] == 0u);
        }

        // InverseAdjacenciesOf(0) = {}  — no edge has v==0 (since u < v, 0 is never a second
        // endpoint)
        {
            int count = 0;
            adj.InverseAdjacenciesOf(0u, [&](std::uint32_t, std::uint32_t, EdgeData const&) {
                ++count;
            });
            CHECK(count == 0);
        }

        // InverseAdjacenciesOf(2) = {(2,1)}  — edge (1,2) has v==2
        {
            std::vector<std::uint32_t> sources;
            adj.InverseAdjacenciesOf(2u, [&](std::uint32_t v, std::uint32_t u, EdgeData const& w) {
                CHECK(v == 2u);
                sources.push_back(u);
            });
            REQUIRE(sources.size() == 1u);
            CHECK(sources[0] == 1u);
        }

        // InverseAdjacenciesOf(3) = {(3,0)}  — edge (0,3) has v==3
        {
            std::vector<std::uint32_t> sources;
            adj.InverseAdjacenciesOf(3u, [&](std::uint32_t v, std::uint32_t u, EdgeData const& w) {
                CHECK(v == 3u);
                sources.push_back(u);
                CHECK(w.weight == doctest::Approx(static_cast<float>(u * 10 + v)));
            });
            REQUIRE(sources.size() == 1u);
            CHECK(sources[0] == 0u);
        }
    }

    SUBCASE("Inverse is not built when bBuildInverse is false")
    {
        using namespace pbat::graph;

        AdjacencySet<EdgeData> adj;
        adj.Construct(3u);

        adj.Add(0u, 1u);
        adj.Add(1u, 2u);
        // Default options: bBuildInverse == false
        adj.Update(
            [](std::uint32_t, std::uint32_t) -> EdgeData { return {}; },
            [](std::uint32_t, std::uint32_t, EdgeData&) {});

        CHECK(adj.Size() == 2u);
        // No inverse arrays should be allocated — we just verify the set works without inverse.
        // (InverseAdjacenciesOf would assert-fail here, so we don't call it.)
    }

    SUBCASE("CompactIds shrinks indirection tables")
    {
        AdjacencySet<EdgeData> adj;
        adj.Construct(5u);

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

        // Verify the new edge got proper data
        adj.AdjacenciesOf(1u, [&](std::uint32_t u, std::uint32_t v, EdgeData const& w) {
            if (v == 2u)
                CHECK(w.weight == doctest::Approx(102.f));
        });
    }
}
