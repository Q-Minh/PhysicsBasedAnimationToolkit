#include "NAryTreeTraversal.h"

#include <array>
#include <doctest/doctest.h>

TEST_CASE("[common] NAryTreeTraversal")
{
    using pbat::Index;
    // Arrange
    struct Node
    {
        Index l{-1}, r{-1};
    };
    std::array<Node, 5> nodes{Node{1, 4}, Node{2, 3}, Node{}, Node{}, Node{}};
    std::vector<Index> result{};
    result.reserve(nodes.size());
    auto const fChildren = [&nodes]<auto c>(Index node) {
        if constexpr (c == 0)
            return nodes[static_cast<std::size_t>(node)].l;
        else
            return nodes[static_cast<std::size_t>(node)].r;
    };
    SUBCASE("Pre-order")
    {
        auto const fVisit = [&result](Index node) {
            result.push_back(node);
            return true;
        };
        // Act
        pbat::common::TraverseNAryTreePseudoPreOrder(fVisit, fChildren);
        // Assert
        for (auto r = 0ULL; r < result.size(); ++r)
        {
            auto node = result[r];
            auto it1 =
                std::find(result.begin(), result.end(), nodes[static_cast<std::size_t>(node)].l);
            auto it2 =
                std::find(result.begin(), result.end(), nodes[static_cast<std::size_t>(node)].r);
            CHECK_LT(r, std::distance(result.begin(), it1));
            CHECK_LT(r, std::distance(result.begin(), it2));
        }
    }
    SUBCASE("Post-order")
    {
        auto const fVisit = [&result](Index node) {
            result.push_back(node);
        };
        // Act
        pbat::common::TraverseNAryTreePseudoPostOrder(fVisit, fChildren);
        // Assert
        for (auto r = 0ULL; r < result.size(); ++r)
        {
            auto node = result[r];
            auto lc   = nodes[static_cast<std::size_t>(node)].l;
            auto rc   = nodes[static_cast<std::size_t>(node)].r;
            if (lc >= 0)
            {
                auto it1 = std::find(result.begin(), result.end(), lc);
                CHECK_GT(r, std::distance(result.begin(), it1));
            }
            if (rc >= 0)
            {
                auto it2 = std::find(result.begin(), result.end(), rc);
                CHECK_GT(r, std::distance(result.begin(), it2));
            }
        }
    }
}