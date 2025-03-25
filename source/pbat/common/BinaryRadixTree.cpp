#include "BinaryRadixTree.h"

#include <algorithm>
#include <doctest/doctest.h>

TEST_CASE("[common] BinaryRadixTree")
{
    // Arrange
    auto constexpr n            = 10;
    bool constexpr bStoreParent = true;
    using CodeType              = std::uint32_t;
    using EigenCodes            = Eigen::Vector<CodeType, Eigen::Dynamic>;
    auto codes                  = EigenCodes::Random(n).eval();
    std::sort(codes.begin(), codes.end());
    // Act
    pbat::common::BinaryRadixTree tree{codes, bStoreParent};
    auto const nLeaves   = tree.LeafCount();
    auto const nInternal = tree.InternalNodeCount();
    // Assert
    CHECK_EQ(nLeaves, n);
    CHECK_EQ(nInternal, n - 1);
    for (auto i = 0; i < nInternal; ++i)
    {
        auto lc = tree.Left(i);
        auto rc = tree.Right(i);
        CHECK_EQ(tree.Parent(lc), i);
        CHECK_EQ(tree.Parent(rc), i);
    }
    pbat::IndexVectorX nParentsFromChildren(n - 1);
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
}
