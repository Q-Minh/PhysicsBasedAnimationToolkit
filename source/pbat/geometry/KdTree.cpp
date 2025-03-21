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
        IndexVectorX const& permutation = kdTree.Permutation();
        CHECK_EQ(permutation.size(), N);
        IndexVectorX counts(N);
        counts.setZero();
        for (auto idx : permutation)
            ++counts(idx);
        bool const bIndicesAreUnique = (counts.array() == 1).all();
        CHECK(bIndicesAreUnique);
    });
}