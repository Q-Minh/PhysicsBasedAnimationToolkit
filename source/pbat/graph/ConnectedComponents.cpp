#include "ConnectedComponents.h"

namespace pbat::graph {
} // namespace pbat::graph

#include <doctest/doctest.h>
#include <set>
#include <vector>

TEST_CASE("[graph] ConnectedComponents")
{
    SUBCASE("Single component")
    {
        // Graph: 0-1-2-3 (all connected)
        Eigen::VectorXi ptr(5);
        ptr << 0, 1, 2, 3, 3;
        Eigen::VectorXi adj(3);
        adj << 1, 2, 3;
        Eigen::VectorXi cc(4);
        cc.setConstant(-1);
        pbat::graph::BreadthFirstSearch<int> bfs(4);
        auto nComponents = pbat::graph::ConnectedComponents<int>(ptr, adj, cc, bfs);
        // All nodes should have the same label
        CHECK_EQ(nComponents, 1);
        for (int i = 1; i < cc.size(); ++i)
            CHECK_EQ(cc[i], cc[0]);
    }
    SUBCASE("Two components")
    {
        // Graph: 0-1  2-3 (two disconnected components)
        Eigen::VectorXi ptr(5);
        ptr << 0, 1, 1, 2, 2;
        Eigen::VectorXi adj(2);
        adj << 1, 3;
        Eigen::VectorXi cc(4);
        cc.setConstant(-1);
        pbat::graph::DepthFirstSearch<int> dfs(4);
        auto nComponents = pbat::graph::ConnectedComponents<int>(ptr, adj, cc, dfs);
        // 0 and 1 should have the same label, 2 and 3 should have the same label, but different
        // from 0/1
        CHECK_EQ(nComponents, 2);
        CHECK_EQ(cc[0], cc[1]);
        CHECK_EQ(cc[2], cc[3]);
        CHECK_NE(cc[0], cc[2]);
    }
    SUBCASE("All singletons")
    {
        // Graph: 0, 1, 2 (no edges)
        Eigen::VectorXi ptr(4);
        ptr << 0, 0, 0, 0;
        Eigen::VectorXi adj(0);
        Eigen::VectorXi cc(3);
        cc.setConstant(-1);
        pbat::graph::BreadthFirstSearch<int> bfs(3);
        auto nComponents = pbat::graph::ConnectedComponents<int>(ptr, adj, cc, bfs);
        // Each node should have a unique label
        CHECK_EQ(nComponents, 3);
        std::set<int> unique_labels(cc.data(), cc.data() + cc.size());
        CHECK_EQ(unique_labels.size(), 3);
    }
    SUBCASE("Cycle")
    {
        // Graph: 0-1-2-0 (cycle)
        Eigen::VectorXi ptr(4);
        ptr << 0, 1, 2, 3;
        Eigen::VectorXi adj(3);
        adj << 1, 2, 0;
        Eigen::VectorXi cc(3);
        cc.setConstant(-1);
        pbat::graph::DepthFirstSearch<int> dfs(3);
        auto nComponents = pbat::graph::ConnectedComponents<int>(ptr, adj, cc, dfs);
        // All nodes should have the same label
        CHECK_EQ(nComponents, 1);
        for (int i = 1; i < cc.size(); ++i)
            CHECK_EQ(cc[i], cc[0]);
    }
}
