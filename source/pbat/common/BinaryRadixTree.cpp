#include "BinaryRadixTree.h"

#include <algorithm>
#include <doctest/doctest.h>

TEST_CASE("[common] BinaryRadixTree")
{
    using namespace pbat;
    // Arrange
    auto constexpr n            = 10;
    bool constexpr bStoreParent = true;
    using CodeType              = std::uint32_t;
    using EigenCodes            = Eigen::Vector<CodeType, Eigen::Dynamic>;

    auto const fAssert = [=](common::BinaryRadixTree<Index> const& tree) {
        auto const nLeaves   = tree.LeafCount();
        auto const nInternal = tree.InternalNodeCount();
        CHECK_EQ(nLeaves, n);
        CHECK_EQ(nInternal, n - 1);
        for (auto i = 0; i < nInternal; ++i)
        {
            auto lc = tree.Left(i);
            auto rc = tree.Right(i);
            CHECK_EQ(tree.Parent(lc), i);
            CHECK_EQ(tree.Parent(rc), i);
        }
        IndexVectorX nParentsFromChildren(n - 1);
        nParentsFromChildren.setZero();
        for (auto i = 0; i < nInternal + nLeaves; ++i)
        {
            auto const p = tree.Parent(i);
            CHECK_LT(p, nInternal);
            ++nParentsFromChildren(p);
        }
        // Root node has self-parent loop
        CHECK_EQ(nParentsFromChildren(0), 3);
        for (auto i = 1; i < nInternal; ++i)
        {
            CHECK_EQ(nParentsFromChildren(i), 2);
        }
    };
    SUBCASE("Random codes")
    {
        auto codes = EigenCodes::Random(n).eval();
        std::sort(codes.begin(), codes.end());
        // Act
        common::BinaryRadixTree tree{codes, bStoreParent};
        // Assert
        fAssert(tree);
    }
    SUBCASE("Duplicate codes")
    {
        auto codes = EigenCodes::Zero(n).eval();
        // Act
        common::BinaryRadixTree tree{codes, bStoreParent};
        // Assert
        fAssert(tree);
    }
}
