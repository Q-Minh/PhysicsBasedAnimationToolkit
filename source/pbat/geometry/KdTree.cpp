#include "KdTree.h"

#include <algorithm>
#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <ranges>

TEST_CASE("[geometry] KdTree")
{
    using namespace pbat;
    common::ForValues<1, 2, 3>([]<auto Dims>() {
        auto constexpr N               = 10u;
        auto constexpr maxPointsInLeaf = 2;
        Matrix<Dims, N> const P        = Matrix<Dims, N>::Random();
        geometry::KdTree<Dims> const kdTree(P, maxPointsInLeaf);
        for (auto const& node : kdTree.Nodes())
        {
            if (node.IsLeafNode())
            {
                CHECK_LE(node.n, maxPointsInLeaf);
            }
        }
        std::vector<Index> const& permutation = kdTree.Permutation();
        CHECK_EQ(permutation.size(), N);
        std::vector<std::size_t> counts(N, 0ULL);
        for (auto idx : permutation)
            ++counts[static_cast<std::size_t>(idx)];
        bool const bIndicesAreUnique =
            std::ranges::all_of(counts, [](std::size_t count) { return count == 1ULL; });
        CHECK(bIndicesAreUnique);
    });
}