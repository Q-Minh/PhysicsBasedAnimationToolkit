#include "AabbRadixTreeHierarchy.h"

#include "DistanceQueries.h"
#include "OverlapQueries.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] AabbRadixTreeHierarchy")
{
    using namespace pbat;
    // Arrange
    auto constexpr kDims = 3;
    auto constexpr n     = 10;
    Matrix<2 * kDims, Eigen::Dynamic> B(2 * kDims, n);
    B.topRows<kDims>().setRandom();
    B.bottomRows<kDims>() = B.topRows<kDims>();
    // Act
    geometry::AabbRadixTreeHierarchy<kDims> tree{B};
    // Assert
    using math::linalg::mini::FromEigen;
    for (auto i = 0; i < n; ++i)
    {
        auto P = B.col(i).head<kDims>();
        tree.Overlaps(
            [&]<class TL, class TU>(TL const& L, TU const& U) {
                return geometry::OverlapQueries::PointAxisAlignedBoundingBox(
                    FromEigen(P),
                    FromEigen(L),
                    FromEigen(U));
            },
            [&](Index j) { return P.isApprox(B.col(j).head<kDims>()); },
            [&](Index j, [[maybe_unused]] Index k) { CHECK_EQ(j, i); });
        tree.NearestNeighbour(
            [&]<class TL, class TU>(TL const& L, TU const& U) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(P),
                    FromEigen(L),
                    FromEigen(U));
            },
            [&](Index j) { return (P - B.col(j).head<kDims>()).squaredNorm(); },
            [&](Index j, Scalar d, Index k) {
                CHECK_EQ(j, i);
                CHECK_EQ(d, Scalar(0));
                CHECK_EQ(k, 0);
            });
    }
}