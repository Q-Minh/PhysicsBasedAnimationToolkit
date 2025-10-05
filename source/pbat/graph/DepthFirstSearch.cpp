#include "DepthFirstSearch.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <doctest/doctest.h>
#include <vector>

TEST_CASE("[graph] DepthFirstSearch")
{
    SUBCASE("Simple chain")
    {
        // Graph: 0 -> 1 -> 2 -> 3
        Eigen::VectorXi ptr(5); // 4 nodes + 1
        ptr << 0, 1, 2, 3, 3;
        Eigen::VectorXi adj(3);
        adj << 1, 2, 3;

        pbat::graph::DepthFirstSearch<int> dfs(ptr.size() - 1);

        std::vector<int> visited;
        dfs(ptr, adj, 0, [&](int v) { visited.push_back(v); });

        CHECK(visited == std::vector<int>({0, 1, 2, 3}));
    }
    SUBCASE("Tree")
    {
        // Graph:
        // 0 -> 1,2
        // 1 -> 3
        // 2 -> (none)
        // 3 -> (none)
        Eigen::VectorXi ptr(5);
        ptr << 0, 2, 3, 3, 3;
        Eigen::VectorXi adj(3);
        adj << 1, 2, 3;

        pbat::graph::DepthFirstSearch<int> dfs(ptr.size() - 1);

        std::vector<int> visited;
        dfs(ptr, adj, 0, [&](int v) { visited.push_back(v); });

        // DFS order: 0, 2, 1, 3 or 0, 1, 3, 2 (depends on stack push order)
        // Our implementation pushes 2 then 1, so 1 is visited after 2
        CHECK(visited == std::vector<int>({0, 2, 1, 3}));
    }
    SUBCASE("Disconnected graph")
    {
        // Graph: 0 -> 1, 2 (disconnected)
        Eigen::VectorXi ptr(4);
        ptr << 0, 1, 1, 1;
        Eigen::VectorXi adj(1);
        adj << 1;

        pbat::graph::DepthFirstSearch<int> dfs(ptr.size() - 1);

        std::vector<int> visited;
        dfs(ptr, adj, 0, [&](int v) { visited.push_back(v); });

        CHECK(visited == std::vector<int>({0, 1}));
    }
    SUBCASE("Cycle")
    {
        // Graph: 0 -> 1 -> 2 -> 0 (cycle)
        Eigen::VectorXi ptr(4);
        ptr << 0, 1, 2, 3;
        Eigen::VectorXi adj(3);
        adj << 1, 2, 0;

        pbat::graph::DepthFirstSearch<int> dfs(ptr.size() - 1);

        std::vector<int> visited;
        dfs(ptr, adj, 0, [&](int v) { visited.push_back(v); });

        // Should visit all nodes once
        CHECK(visited.size() == 3);
        CHECK(std::find(visited.begin(), visited.end(), 0) != visited.end());
        CHECK(std::find(visited.begin(), visited.end(), 1) != visited.end());
        CHECK(std::find(visited.begin(), visited.end(), 2) != visited.end());
    }
}