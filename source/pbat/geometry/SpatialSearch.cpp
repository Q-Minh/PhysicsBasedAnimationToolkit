#include "SpatialSearch.h"

#include "AxisAlignedBoundingBox.h"
#include "KdTree.h"
#include "pbat/common/Eigen.h"
#include "pbat/common/Hash.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/math/linalg/mini/Matrix.h"

#include <Eigen/Core>
#include <algorithm>
#include <doctest/doctest.h>
#include <unordered_set>
#include <vector>

TEST_CASE("[geometry] SpatialSearch")
{
    using namespace pbat;
    // Arrange
    auto constexpr kDims   = 3;
    auto constexpr kPoints = 100;
    MatrixX P              = MatrixX::Random(kDims, kPoints);
    using Aabb             = geometry::AxisAlignedBoundingBox<kDims>;
    using KdTree           = geometry::KdTree<kDims>;
    KdTree tree{P};
    auto const nNodes = tree.Nodes().size();
    auto const* nodes = tree.Nodes().data();
    auto const& inds  = tree.Permutation();
    // Compute node AABBs in the O(n log n) way
    std::vector<Aabb> aabbs(nNodes);
    auto const fComputeAabbs = [&]() {
        for (auto i = 0ULL; i < nNodes; ++i)
        {
            aabbs[i].setEmpty();
            for (auto p : tree.PointsInNode(static_cast<Index>(i)))
                aabbs[i].extend(P.col(p));
        }
    };
    // Define branch and bound tree getters here
    auto const fChild = [&]<auto c>(Index node) {
        if constexpr (c == 0)
            return nodes[node].Left();
        else
            return nodes[node].Right();
    };
    auto const fIsLeaf = [&](Index node) {
        return nodes[node].IsLeaf();
    };
    auto const fLeafSize = [&](Index node) {
        return nodes[node].n;
    };
    auto const fLeafObject = [&](Index node, Index i) {
        auto j = nodes[node].begin + i;
        return inds(j);
    };
    fComputeAabbs();

    using pbat::math::linalg::mini::FromEigen;
    SUBCASE("Overlaps")
    {
        // Act
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const p = P.col(i).head<kDims>();
            Index overlapping{-1};
            geometry::Overlaps(
                fChild,
                fIsLeaf,
                fLeafSize,
                fLeafObject,
                [&](Index node) {
                    return geometry::OverlapQueries::PointAxisAlignedBoundingBox(
                        FromEigen(p),
                        FromEigen(aabbs[static_cast<std::size_t>(node)].min()),
                        FromEigen(aabbs[static_cast<std::size_t>(node)].max()));
                },
                [&](Index o) { return p.isApprox(P.col(o)); },
                [&](Index o, [[maybe_unused]] Index k) { overlapping = o; });
            // Assert
            CHECK_EQ(overlapping, i);
        }
    }
    SUBCASE("Self-Overlaps")
    {
        std::unordered_set<std::pair<Index, Index>> overlaps{};
        overlaps.reserve(kPoints * 2);
        Index nOverlaps{0};
        auto const fNodesOverlap = [&](Index n1, Index n2) {
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(
                FromEigen(aabbs[static_cast<std::size_t>(n1)].min()),
                FromEigen(aabbs[static_cast<std::size_t>(n1)].max()),
                FromEigen(aabbs[static_cast<std::size_t>(n2)].min()),
                FromEigen(aabbs[static_cast<std::size_t>(n2)].max()));
        };
        auto const fObjectsOverlap = [&](Index o1, Index o2) {
            return (P.col(o1).isApprox(P.col(o2)));
        };
        auto const fCheckOverlap = [&](Index o1, Index o2, [[maybe_unused]] Index k) {
            ++nOverlaps;
            overlaps.insert({std::min(o1, o2), std::max(o1, o2)});
        };
        // Act
        geometry::SelfOverlaps(
            fChild,
            fIsLeaf,
            fLeafSize,
            fLeafObject,
            fNodesOverlap,
            fObjectsOverlap,
            fCheckOverlap);
        // Assert
        CHECK_EQ(nOverlaps, 0);
        CHECK(overlaps.empty());
        // Arrange
        P.rightCols(kPoints / 2) = P.leftCols(kPoints / 2);
        fComputeAabbs();
        // Act
        bool const bTraversalCompleted = geometry::SelfOverlaps(
            fChild,
            fIsLeaf,
            fLeafSize,
            fLeafObject,
            fNodesOverlap,
            fObjectsOverlap,
            fCheckOverlap);
        // Assert
        CHECK(bTraversalCompleted);
        CHECK_EQ(nOverlaps, kPoints / 2);
        CHECK_EQ(overlaps.size(), kPoints / 2);
        for (auto [o1, o2] : overlaps)
            CHECK_NE(o1, o2);
    }
    // Arrange
    Vector<3> const p = Vector<3>::Random();
    VectorX d2        = (P.colwise() - p).colwise().squaredNorm();
    SUBCASE("Nearest neighbour")
    {
        // Act
        Index nn{-1};
        Scalar dmin{std::numeric_limits<Scalar>::max()};
        bool bUseBestFirstSearch{false};
        SUBCASE("Best-first order")
        {
            bUseBestFirstSearch = true;
        }
        geometry::NearestNeighbour(
            fChild,
            fIsLeaf,
            fLeafSize,
            fLeafObject,
            [&](Index node) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(p),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].min()),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].max()));
            },
            [&](Index o) { return (p - P.col(o)).squaredNorm(); },
            [&](Index o, Scalar d, [[maybe_unused]] Index k) {
                nn   = o;
                dmin = d;
            },
            bUseBestFirstSearch);
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
            fChild,
            fIsLeaf,
            fLeafSize,
            fLeafObject,
            [&](Index node) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(p),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].min()),
                    FromEigen(aabbs[static_cast<std::size_t>(node)].max()));
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