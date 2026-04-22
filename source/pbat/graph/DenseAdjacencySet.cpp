#include "DenseAdjacencySet.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <cstdint>
#include <doctest/doctest.h>
#include <random>
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
        DenseAdjacencySet<int, EdgeData> adj;
        CHECK(adj.Size() == 0u);
        CHECK(adj.NumSourceVertices() == 0u);
        CHECK(adj.Data<EdgeData>().empty());
    }

    SUBCASE("Add and Update create adjacencies")
    {
        DenseAdjacencySet<int, EdgeData> adj;

        // Add some edges
        std::vector<std::pair<int, int>> edges{};
        edges.push_back({0u, 1u});
        edges.push_back({1u, 3u});
        edges.push_back({2u, 4u});
        adj.Assign(edges);
        CHECK(adj.Size() == 3u);
        CHECK(adj.Data<EdgeData>().size() == 3u);
    }

    SUBCASE("Add stores directed edges as-is")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{};
        edges.push_back({1u, 3u});
        edges.push_back({3u, 1u});
        adj.Assign(edges);
        adj.Finalize();
        CHECK(adj.Size() == 2u);
        CHECK(adj.NumSourceVertices() == 4u);
        CHECK(adj.Has(1u, 3u) >= 0);
        CHECK(adj.Has(3u, 1u) >= 0);
    }

    SUBCASE("Update preserves existing adjacencies and detects additions/removals")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        // First update: add edges (0,1) and (1,2)
        std::vector<std::pair<int, int>> edges{};
        edges.push_back({0, 1});
        edges.push_back({1, 2});
        adj.Assign(edges);
        CHECK(adj.Size() == 2u);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).weight = 1.f; });
        // Second update: keep (0,1), drop (1,2), add (2,3)
        edges.clear();
        edges.push_back({0, 1});
        edges.push_back({2, 3});
        adj.Assign(edges);
        adj.Finalize();
        // (0,1) stayed and was not modified
        CHECK(adj.Size() == 2u);
        int k01 = adj.Has(0, 1);
        CHECK(k01 >= 0);
        CHECK(adj.Data<EdgeData>(k01).weight == 1.f);
        // (1,2) was removed
        CHECK(adj.Has(1, 2) < 0);
        // (2,3) was added
        int k23 = adj.Has(2, 3);
        CHECK(k23 >= 0);
        CHECK(adj.Data<EdgeData>(k23).weight == 0.f);
    }

    SUBCASE("AdjacenciesOf iterates over neighbours of u")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{};
        edges.push_back({1, 0});
        edges.push_back({1, 2});
        edges.push_back({1, 4});
        edges.push_back({3, 4});
        adj.Assign(edges);
        adj.Finalize();

        // Has() interface
        {
            CHECK(adj.Has(1u, 0u) >= 0);
            CHECK(adj.Has(1u, 2u) >= 0);
            CHECK(adj.Has(1u, 4u) >= 0);
            CHECK(adj.Has(3u, 4u) >= 0);
            CHECK(adj.Has(2u, 1u) < 0);
            CHECK(adj.Has(0u, 1u) < 0);
        }

        // Edges are directed: Add(1,0) stores (1,0).
        // Adjacencies with first endpoint 0: none
        {
            int count = 0;
            adj.ForEach(0, [&](auto, auto, auto) { ++count; });
            CHECK(count == 0);
        }

        // Adjacencies with first endpoint 1: {(1,0), (1,2), (1,4)}
        {
            std::vector<int> neighbours;
            adj.ForEach(1, [&](int u, int v, int k) {
                CHECK(u == 1u);
                neighbours.push_back(v);
            });
            REQUIRE(neighbours.size() == 3u);
            CHECK(neighbours[0] == 0u);
            CHECK(neighbours[1] == 2u);
            CHECK(neighbours[2] == 4u);
        }

        // Adjacencies with first endpoint 3: {(3,4)}
        {
            std::vector<int> neighbours;
            adj.ForEach(3, [&](int u, int v, int k) {
                CHECK(u == 3);
                neighbours.push_back(v);
            });
            REQUIRE(neighbours.size() == 1u);
            CHECK(neighbours[0] == 4);
        }

        // Adjacencies with first endpoint 2: empty
        {
            int count = 0;
            adj.ForEach(2, [&](int, int) { ++count; });
            CHECK(count == 0);
        }
    }

    SUBCASE("Multiple larger updates converge correctly")
    {
        DenseAdjacencySet<int, EdgeData> adj;

        std::vector<std::pair<int, int>> edges;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);
        auto n = 20;
        for (int i = 0; i < n; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        n = static_cast<int>(edges.size());
        adj.Assign(edges);
        CHECK(adj.Size() == n);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u; });

        // Rebuild same set
        std::ranges::sort(edges);
        adj.Assign(edges);
        CHECK(adj.Size() == n);
        adj.ForEach([&](int u, int v, int k) { CHECK_EQ(adj.Data<EdgeData>(k).tag, u); });

        // Rebuild half the set
        edges.erase(edges.begin() + n / 2, edges.end());
        std::ranges::sort(edges);
        adj.Assign(edges);
        CHECK(adj.Size() == n / 2);
        adj.ForEach([&](int u, int v, int k) { CHECK_EQ(adj.Data<EdgeData>(k).tag, u); });

        // Remove and add from set
        edges.erase(edges.begin(), edges.begin() + n / 4);
        for (int i = 0; i < n / 4; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        adj.Assign(edges);
        CHECK(adj.Size() == edges.size());

        // Empty update → remove everything
        edges.clear();
        adj.Assign(edges);
        CHECK(adj.Size() == 0u);
        CHECK(adj.Data<EdgeData>().empty());
    }

    SUBCASE("Union into empty set adds all edges")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 3}, {2, 4}};
        auto nNew = adj.Union(edges);
        CHECK(nNew == 3);
        CHECK(adj.Size() == 3u);
        CHECK(adj.Data<EdgeData>().size() == 3u);
    }

    SUBCASE("Union with empty set is a no-op")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}};
        adj.Union(edges);
        CHECK(adj.Size() == 2u);
        // Union with empty B
        std::vector<std::pair<int, int>> empty{};
        auto nNew = adj.Union(empty);
        CHECK(nNew == 0);
        CHECK(adj.Size() == 2u);
    }

    SUBCASE("Union preserves existing adjacencies and data")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        // First, add (0,1) and (1,2)
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}};
        adj.Union(edges);
        CHECK(adj.Size() == 2u);
        // Tag existing edges
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).weight = 1.f; });
        // Union with (0,1), (2,3): (0,1) already exists, (2,3) is new
        edges.clear();
        edges.push_back({0, 1});
        edges.push_back({2, 3});
        auto nNew = adj.Union(edges);
        adj.Finalize();
        CHECK(nNew == 1); // only (2,3) is new
        CHECK(adj.Size() == 3u);
        // (0,1) kept its data
        int k01 = adj.Has(0, 1);
        CHECK(k01 >= 0);
        CHECK(adj.Data<EdgeData>(k01).weight == 1.f);
        // (1,2) still exists and kept its data
        int k12 = adj.Has(1, 2);
        CHECK(k12 >= 0);
        CHECK(adj.Data<EdgeData>(k12).weight == 1.f);
        // (2,3) was added with default data
        int k23 = adj.Has(2, 3);
        CHECK(k23 >= 0);
        CHECK(adj.Data<EdgeData>(k23).weight == 0.f);
    }

    SUBCASE("Union does not remove edges absent from B")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {3, 4}};
        adj.Union(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        // Union with only (1,2) — unlike Assign, (0,1) and (3,4) should NOT be removed
        edges.clear();
        edges.push_back({1, 2});
        auto nNew = adj.Union(edges);
        adj.Finalize();
        CHECK(nNew == 0);
        CHECK(adj.Size() == 3u);
        // All original edges still present with their data
        int k01 = adj.Has(0, 1);
        CHECK(k01 >= 0);
        CHECK(adj.Data<EdgeData>(k01).tag == 0 * 100 + 1);
        int k12 = adj.Has(1, 2);
        CHECK(k12 >= 0);
        CHECK(adj.Data<EdgeData>(k12).tag == 1 * 100 + 2);
        int k34 = adj.Has(3, 4);
        CHECK(k34 >= 0);
        CHECK(adj.Data<EdgeData>(k34).tag == 3 * 100 + 4);
    }

    SUBCASE("Union with identical set adds nothing")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 2}, {1, 3}, {4, 5}};
        adj.Union(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).weight = 42.f; });
        // Re-union with the same set
        std::ranges::sort(edges);
        auto nNew = adj.Union(edges);
        CHECK(nNew == 0);
        CHECK(adj.Size() == 3u);
        // Data unchanged
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).weight == 42.f); });
    }

    SUBCASE("Union with disjoint set adds all")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edgesA{{0, 1}, {2, 3}};
        adj.Union(edgesA);
        CHECK(adj.Size() == 2u);
        std::vector<std::pair<int, int>> edgesB{{4, 5}, {6, 7}};
        auto nNew = adj.Union(edgesB);
        CHECK(nNew == 2);
        CHECK(adj.Size() == 4u);
        adj.Finalize();
        CHECK(adj.Has(0, 1) >= 0);
        CHECK(adj.Has(2, 3) >= 0);
        CHECK(adj.Has(4, 5) >= 0);
        CHECK(adj.Has(6, 7) >= 0);
    }

    SUBCASE("Multiple larger Unions converge correctly")
    {
        DenseAdjacencySet<int, EdgeData> adj;

        std::set<std::pair<int, int>> expected;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);

        // Build up the set through multiple unions
        for (int round = 0; round < 5; ++round)
        {
            std::vector<std::pair<int, int>> edges;
            for (int i = 0; i < 10; ++i)
                edges.push_back({dis(gen), dis(gen)});
            std::ranges::sort(edges);
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            for (auto const& e : edges)
                expected.insert(e);
            adj.Union(edges);
            CHECK(adj.Size() == expected.size());
        }
        // Tag every edge
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        // Verify all expected edges exist
        adj.Finalize();
        for (auto const& [u, v] : expected)
        {
            int k = adj.Has(u, v);
            CHECK(k >= 0);
            CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v);
        }
    }

    SUBCASE("Subtract from empty set is a no-op")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}};
        auto nRemoved = adj.Subtract(edges);
        CHECK(nRemoved == 0);
        CHECK(adj.Size() == 0u);
    }

    SUBCASE("Subtract with empty B is a no-op")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}};
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        std::vector<std::pair<int, int>> empty{};
        auto nRemoved = adj.Subtract(empty);
        CHECK(nRemoved == 0);
        CHECK(adj.Size() == 3u);
        // Data unchanged
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v); });
    }

    SUBCASE("Subtract removes only overlapping edges")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).weight = 1.f; });
        // Subtract {(1,2), (3,4)}: removes those two, keeps (0,1) and (2,3)
        std::vector<std::pair<int, int>> toRemove{{1, 2}, {3, 4}};
        auto nRemoved = adj.Subtract(toRemove);
        adj.Finalize();
        CHECK(nRemoved == 2);
        CHECK(adj.Size() == 2u);
        // Remaining edges kept their data
        int k01 = adj.Has(0, 1);
        CHECK(k01 >= 0);
        CHECK(adj.Data<EdgeData>(k01).weight == 1.f);
        int k23 = adj.Has(2, 3);
        CHECK(k23 >= 0);
        CHECK(adj.Data<EdgeData>(k23).weight == 1.f);
        // Removed edges are gone
        CHECK(adj.Has(1, 2) < 0);
        CHECK(adj.Has(3, 4) < 0);
    }

    SUBCASE("Subtract with identical set removes everything")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {2, 3}, {4, 5}};
        adj.Assign(edges);
        auto nRemoved = adj.Subtract(edges);
        CHECK(nRemoved == 3);
        CHECK(adj.Size() == 0u);
        CHECK(adj.Data<EdgeData>().empty());
    }

    SUBCASE("Subtract with disjoint set removes nothing")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {2, 3}};
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        std::vector<std::pair<int, int>> disjoint{{4, 5}, {6, 7}};
        auto nRemoved = adj.Subtract(disjoint);
        CHECK(nRemoved == 0);
        CHECK(adj.Size() == 2u);
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v); });
    }

    SUBCASE("Subtract where B is a superset of A removes everything")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{1, 2}, {3, 4}};
        adj.Assign(edges);
        std::vector<std::pair<int, int>> superset{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {5, 6}};
        auto nRemoved = adj.Subtract(superset);
        CHECK(nRemoved == 2);
        CHECK(adj.Size() == 0u);
    }

    SUBCASE("Multiple larger Subtracts converge correctly")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::set<std::pair<int, int>> expected;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);

        // Build initial set
        std::vector<std::pair<int, int>> edges;
        for (int i = 0; i < 20; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        adj.Assign(edges);
        for (auto const& e : edges)
            expected.insert(e);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });

        // Subtract in rounds
        for (int round = 0; round < 3; ++round)
        {
            std::vector<std::pair<int, int>> toRemove;
            for (int i = 0; i < 5; ++i)
                toRemove.push_back({dis(gen), dis(gen)});
            std::ranges::sort(toRemove);
            toRemove.erase(std::unique(toRemove.begin(), toRemove.end()), toRemove.end());
            adj.Subtract(toRemove);
            for (auto const& e : toRemove)
                expected.erase(e);
            CHECK(adj.Size() == expected.size());
        }
        // Verify surviving edges kept their data
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v); });
        // Verify all expected edges exist
        adj.Finalize();
        for (auto const& [u, v] : expected)
        {
            int k = adj.Has(u, v);
            CHECK(k >= 0);
            CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v);
        }
    }

    SUBCASE("RemoveIf on empty set is a no-op")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        auto nRemoved = adj.RemoveIf([](int, int, int) { return true; });
        CHECK(nRemoved == 0);
        CHECK(adj.Size() == 0u);
    }

    SUBCASE("RemoveIf with always-false predicate removes nothing")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}};
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        auto nRemoved = adj.RemoveIf([](int, int, int) { return false; });
        CHECK(nRemoved == 0);
        CHECK(adj.Size() == 3u);
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v); });
    }

    SUBCASE("RemoveIf with always-true predicate removes everything")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}};
        adj.Assign(edges);
        auto nRemoved = adj.RemoveIf([](int, int, int) { return true; });
        CHECK(nRemoved == 3);
        CHECK(adj.Size() == 0u);
        CHECK(adj.Data<EdgeData>().empty());
    }

    SUBCASE("RemoveIf removes edges matching a predicate on endpoints")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).weight = 1.f; });
        // Remove edges whose source vertex is even
        auto nRemoved = adj.RemoveIf([](int u, int, int) { return u % 2 == 0; });
        adj.Finalize();
        CHECK(nRemoved == 3); // (0,1), (2,3), (4,5) removed
        CHECK(adj.Size() == 2u);
        // Remaining edges kept their data
        int k12 = adj.Has(1, 2);
        CHECK(k12 >= 0);
        CHECK(adj.Data<EdgeData>(k12).weight == 1.f);
        int k34 = adj.Has(3, 4);
        CHECK(k34 >= 0);
        CHECK(adj.Data<EdgeData>(k34).weight == 1.f);
        // Removed edges are gone
        CHECK(adj.Has(0, 1) < 0);
        CHECK(adj.Has(2, 3) < 0);
        CHECK(adj.Has(4, 5) < 0);
    }

    SUBCASE("RemoveIf removes edges matching a predicate on data")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        adj.Assign(edges);
        // Tag specific edges for removal
        adj.ForEach([&](int u, int v, int k) {
            adj.Data<EdgeData>(k).tag   = u * 100 + v;
            adj.Data<EdgeData>(k).weight = (u == 1 || u == 3) ? -1.f : 1.f;
        });
        // Remove edges with negative weight
        auto nRemoved =
            adj.RemoveIf([&](int, int, int k) { return adj.Data<EdgeData>(k).weight < 0.f; });
        adj.Finalize();
        CHECK(nRemoved == 2); // (1,2), (3,4) removed
        CHECK(adj.Size() == 2u);
        int k01 = adj.Has(0, 1);
        CHECK(k01 >= 0);
        CHECK(adj.Data<EdgeData>(k01).tag == 0 * 100 + 1);
        int k23 = adj.Has(2, 3);
        CHECK(k23 >= 0);
        CHECK(adj.Data<EdgeData>(k23).tag == 2 * 100 + 3);
    }

    SUBCASE("RemoveIf preserves sorted order")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges{
            {0, 1}, {0, 2}, {1, 0}, {1, 3}, {2, 1}, {2, 4}, {3, 0}, {3, 2}};
        adj.Assign(edges);
        // Remove every other edge by index
        int idx = 0;
        adj.RemoveIf([&](int, int, int) { return (idx++) % 2 == 0; });
        // Verify remaining edges are still sorted
        std::vector<std::pair<int, int>> remaining;
        adj.ForEach([&](int u, int v) { remaining.push_back({u, v}); });
        CHECK(std::ranges::is_sorted(remaining));
        CHECK(remaining.size() == 4u);
    }

    SUBCASE("Multiple RemoveIf operations converge correctly")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::set<std::pair<int, int>> expected;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 7);

        // Build initial set
        std::vector<std::pair<int, int>> edges;
        for (int i = 0; i < 30; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        adj.Assign(edges);
        for (auto const& e : edges)
            expected.insert(e);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });

        // Round 1: remove edges with source vertex == 0
        adj.RemoveIf([](int u, int, int) { return u == 0; });
        std::erase_if(expected, [](auto const& e) { return e.first == 0; });
        CHECK(adj.Size() == expected.size());

        // Round 2: remove edges with target vertex > 5
        adj.RemoveIf([](int, int v, int) { return v > 5; });
        std::erase_if(expected, [](auto const& e) { return e.second > 5; });
        CHECK(adj.Size() == expected.size());

        // Verify surviving edges kept their data
        adj.ForEach(
            [&](int u, int v, int k) { CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v); });
        // Verify all expected edges exist
        adj.Finalize();
        for (auto const& [u, v] : expected)
        {
            int k = adj.Has(u, v);
            CHECK(k >= 0);
            CHECK(adj.Data<EdgeData>(k).tag == u * 100 + v);
        }
    }

    SUBCASE("CompactIds shrinks indirection tables")
    {
        DenseAdjacencySet<int, EdgeData> adj;
        std::vector<std::pair<int, int>> edges;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 5);
        auto n = 20;
        for (int i = 0; i < n; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        adj.Assign(edges);
        // Remove edges then add some new ones so the indirection table should be larger than the
        // actual number of adjacencies
        edges.erase(edges.begin() + n / 2, edges.end());
        for (int i = 0; i < n / 2; ++i)
            edges.push_back({dis(gen), dis(gen)});
        std::ranges::sort(edges);
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        adj.Assign(edges);
        adj.ForEach([&](int u, int v, int k) { adj.Data<EdgeData>(k).tag = u * 100 + v; });
        // Act: Compact the indirection tables
        adj.CompactIds();
        // Assert: Check that all edges are still reachable
        std::vector<bool> edgeExists(adj.Size(), false);
        adj.ForEach([&](int u, int v, int k) {
            CHECK_EQ(adj.Data<EdgeData>(k).tag, u * 100 + v);
            edgeExists[k] = true;
        });
        CHECK(std::ranges::all_of(edgeExists, [](auto x) { return x; }));
    }
}
