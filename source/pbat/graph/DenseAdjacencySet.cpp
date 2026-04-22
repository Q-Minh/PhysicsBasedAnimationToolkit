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
        adj.Assign(edges);
        CHECK(adj.Size() == n);
        adj.ForEach([&](int u, int v, int k) { CHECK_EQ(adj.Data<EdgeData>(k).tag, u); });

        // Rebuild half the set
        edges.erase(edges.begin() + n / 2, edges.end());
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
