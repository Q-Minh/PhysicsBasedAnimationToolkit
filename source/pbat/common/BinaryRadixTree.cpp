#include "BinaryRadixTree.h"

#include <algorithm>
#include <doctest/doctest.h>

TEST_CASE("[common] BinaryRadixTree")
{
    // Arrange
    auto constexpr n = 10;
    using CodeType   = std::uint32_t;
    using EigenCodes = Eigen::Vector<CodeType, Eigen::Dynamic>;
    auto codes       = EigenCodes::Random(n).eval();
    std::sort(codes.begin(), codes.end());
    // Act
    pbat::common::BinaryRadixTree tree{codes};
    // Assert
    CHECK_EQ(tree.LeafCount(), n);
    CHECK_EQ(tree.InternalNodeCount(), n - 1);
    for (auto i = 0; i < tree.InternalNodeCount(); ++i)
    {
        auto lc = tree.Left(i);
        auto rc = tree.Right(i);
        CHECK_EQ(tree.Parent(lc), i);
        CHECK_EQ(tree.Parent(rc), i);
    }
}
