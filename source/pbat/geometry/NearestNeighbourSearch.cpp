#include "NearestNeighbourSearch.h"

#include "AxisAlignedBoundingBox.h"
#include "KdTree.h"
#include "pbat/common/Eigen.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <Eigen/Core>
#include <doctest/doctest.h>
#include <vector>

TEST_CASE("[geometry] NearestNeighbourSearch")
{
    using namespace pbat;
    // Arrange
    auto constexpr kDims = 3;
    MatrixX P            = MatrixX::Random(kDims, 100);
    using Aabb           = geometry::AxisAlignedBoundingBox<kDims>;
    using KdTree         = geometry::KdTree<kDims>;
    KdTree tree{P};
    auto const& nodes = tree.Nodes();
    auto const& inds  = tree.Permutation();

    std::vector<Aabb> aabbs(nodes.size());
    for (auto i = 0ULL; i < nodes.size(); ++i)
        for (auto p : tree.PointsInNode(static_cast<Index>(i)))
            aabbs[i].extend(P.col(p));

    Vector<3> const p = Vector<3>::Random();
    VectorX d2        = (P.colwise() - p).colwise().squaredNorm();

    using pbat::math::linalg::mini::FromEigen;
    SUBCASE("Nearest neighbour")
    {
        // Act
        Index nn{-1};
        Scalar dmin{std::numeric_limits<Scalar>::max()};
        geometry::NearestNeighbour(
            [&]<auto c>(Index node) {
                if constexpr (c == 0)
                    return nodes[static_cast<std::size_t>(node)].lc;
                else
                    return nodes[static_cast<std::size_t>(node)].rc;
            },
            [&](Index node) { return nodes[static_cast<std::size_t>(node)].IsLeafNode(); },
            [&](Index node) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(p),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].min()),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].max()));
            },
            [&](Index node) { return nodes[static_cast<std::size_t>(node)].n; },
            [&](Index node, Index i) {
                auto j = nodes[static_cast<std::size_t>(node)].begin + i;
                return inds(j);
            },
            [&](Index o) { return (p - P.col(o)).squaredNorm(); },
            [&](Index o, Scalar d, [[maybe_unused]] Index k) {
                nn   = o;
                dmin = d;
            });
        // Assert
        Index nnExpected;
        Scalar dminExpected = d2.minCoeff(&nnExpected);
        CHECK_EQ(nn, nnExpected);
        CHECK_EQ(dmin, dminExpected);
    }
    SUBCASE("K nearest neighbours")
    {
        // Act
        static Index constexpr K = 5;
        Index nn[K];
        Scalar dmin[K];
        geometry::KNearestNeighbours(
            [&]<auto c>(Index node) {
                if constexpr (c == 0)
                    return nodes[static_cast<std::size_t>(node)].lc;
                else
                    return nodes[static_cast<std::size_t>(node)].rc;
            },
            [&](Index node) { return nodes[static_cast<std::size_t>(node)].IsLeafNode(); },
            [&](Index node) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(p),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].min()),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].max()));
            },
            [&](Index node) { return nodes[static_cast<std::size_t>(node)].n; },
            [&](Index node, Index i) {
                auto j = nodes[static_cast<std::size_t>(node)].begin + i;
                return inds(j);
            },
            [&](Index o) { return (p - P.col(o)).squaredNorm(); },
            [&](Index o, Scalar d, Index k) {
                nn[k]   = o;
                dmin[k] = d;
            },
            K);
        // Assert
        for (auto k = 0; k < K; ++k)
        {
            Index nnExpected;
            Scalar dminExpected = d2.minCoeff(&nnExpected);
            d2(nnExpected)      = std::numeric_limits<Scalar>::max();
            CHECK_EQ(nn[k], nnExpected);
            CHECK_EQ(dmin[k], dminExpected);
        }
    }
}