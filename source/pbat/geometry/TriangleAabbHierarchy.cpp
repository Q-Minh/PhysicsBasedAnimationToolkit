#include "TriangleAabbHierarchy.h"

#include "pbat/Aliases.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] TriangleAabbHierarchy")
{
    using namespace pbat;
    using pbat::math::linalg::mini::FromEigen;
    SUBCASE("2D")
    {
        // Beam mesh
        // clang-format off
        MatrixX V(2, 8);
        V << 0., 1., 2., 3., 0., 1., 2., 3.,
             0., 0., 0., 0., 1., 1., 1., 1.;
        IndexMatrixX C(3, 6);
        C << 0, 1, 1, 2, 2, 3,
             1, 5, 2, 6, 3, 7,
             4, 4, 5, 5, 6, 6;
        // clang-format on
        std::size_t constexpr kMaxPointsInLeaf = 1ULL;
        auto constexpr kDims                   = 2;
        geometry::TriangleAabbHierarchy<kDims> bvh(V, C, kMaxPointsInLeaf);
        CHECK_EQ(bvh.V.rows(), V.rows());
        CHECK_EQ(bvh.V.cols(), V.cols());
        CHECK_EQ(bvh.C.rows(), C.rows());
        CHECK_EQ(bvh.C.cols(), C.cols());
        auto constexpr kPoints          = 20;
        Matrix<kDims, Eigen::Dynamic> P = Matrix<kDims, Eigen::Dynamic>::Random(kDims, kPoints);
        std::vector<Index> const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
        geometry::AxisAlignedBoundingBox<kDims> const domain(
            V.rowwise().minCoeff(),
            V.rowwise().maxCoeff());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = primitivesContainingP[static_cast<std::size_t>(i)];
            if (not domain.contains(P.col(i)))
            {
                CHECK_EQ(primitiveIdx, Index{-1});
            }
            else
            {
                CHECK_GT(primitiveIdx, Index{-1});
            }
        }
        auto const [nearestPrimitivesToP, distancesToP] = bvh.NearestPrimitivesToPoints(P);
        CHECK_EQ(nearestPrimitivesToP.size(), P.cols());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = nearestPrimitivesToP[static_cast<std::size_t>(i)];
            CHECK_GT(primitiveIdx, Index{-1});
        }

        geometry::TriangleAabbHierarchy<kDims> const otherBvh{bvh};
        IndexMatrix<2, Eigen::Dynamic> const overlappingP = bvh.OverlappingPrimitives(otherBvh);
        auto const nPrimitives                            = static_cast<std::size_t>(C.cols());
        std::vector<std::size_t> overlapCounts(nPrimitives, 0ULL);
        for (auto p : overlappingP.row(0))
        {
            auto const pStl = static_cast<std::size_t>(p);
            ++overlapCounts[pStl];
        }
        bool const bAllPrimitivesHaveAnOverlap =
            std::ranges::all_of(overlapCounts, [](std::size_t c) { return c > 0; });
        CHECK(bAllPrimitivesHaveAnOverlap);

        // Adjacent primitives of the same mesh will touch each other, but they should not count as
        // overlapping.
        auto const nSelfOverlappingPrimitives = bvh.OverlappingPrimitives(bvh).size();
        CHECK_EQ(nSelfOverlappingPrimitives, 0);

        // If points haven't changed, update should preserve the same volumes.
        std::vector<geometry::TriangleAabbHierarchy<kDims>::BoundingVolumeType> const bvsExpected =
            bvh.GetBoundingVolumes();
        bvh.Update();
        auto const nBvs = bvh.GetBoundingVolumes().size();
        for (auto b = std::size_t{0}; b < nBvs; ++b)
        {
            auto const& bv         = bvh.GetBoundingVolumes().at(b);
            auto const& bvExpected = bvsExpected.at(b);
            CHECK(bv.min().isApprox(bvExpected.min()));
            CHECK(bv.max().isApprox(bvExpected.max()));
        }

        // The root volume should contain all the primitive locations
        auto const& rootBv = bvh.GetBoundingVolumes().front();
        for (auto p = 0; p < bvh.C.cols(); ++p)
        {
            auto const X = bvh.PrimitiveLocation(bvh.Primitive(p));
            CHECK(rootBv.contains(X));
        }
    }
    SUBCASE("3D")
    {
        // Beam mesh
        // clang-format off
        MatrixX V(3, 8);
        V << 0., 1., 2., 3., 0., 1., 2., 3.,
             0., 0., 0., 0., 1., 1., 1., 1.,
             0., 0., 0., 0., 1., 1., 1., 1.;
        IndexMatrixX C(3, 6);
        C << 0, 1, 1, 2, 2, 3,
             1, 5, 2, 6, 3, 7,
             4, 4, 5, 5, 6, 6;
        // clang-format on

        std::size_t constexpr kMaxPointsInLeaf = 1ULL;
        auto constexpr kDims                   = 3;
        geometry::TriangleAabbHierarchy<kDims> bvh(V, C, kMaxPointsInLeaf);
        CHECK_EQ(bvh.V.rows(), V.rows());
        CHECK_EQ(bvh.V.cols(), V.cols());
        CHECK_EQ(bvh.C.rows(), C.rows());
        CHECK_EQ(bvh.C.cols(), C.cols());
        auto constexpr kPoints          = 20;
        Matrix<kDims, Eigen::Dynamic> P = Matrix<kDims, Eigen::Dynamic>::Random(kDims, kPoints);
        std::vector<Index> const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
        geometry::AxisAlignedBoundingBox<kDims> const domain(
            V.rowwise().minCoeff(),
            V.rowwise().maxCoeff());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = primitivesContainingP[static_cast<std::size_t>(i)];
            if (primitiveIdx > -1)
            {
                IndexVector<3> const triangle = bvh.Primitive(primitiveIdx);
                Scalar const sd               = geometry::DistanceQueries::PointTriangle(
                    FromEigen(P.col(i).head<kDims>()),
                    FromEigen(V.col(triangle(0)).head<kDims>()),
                    FromEigen(V.col(triangle(1)).head<kDims>()),
                    FromEigen(V.col(triangle(2)).head<kDims>()));
                auto constexpr eps = 1e-14;
                CHECK_LE(sd, eps);
            }
        }
        auto const [nearestPrimitivesToP, distancesToP] = bvh.NearestPrimitivesToPoints(P);
        CHECK_EQ(nearestPrimitivesToP.size(), P.cols());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = nearestPrimitivesToP[static_cast<std::size_t>(i)];
            CHECK_GT(primitiveIdx, Index{-1});
        }

        // NOTE: We cannot check for geometrically overlapping primitives, because our
        // triangle-triangle overlap tests are not robust, i.e. we do not handle all degeneracies
        // (coplanar triangles, triangles sharing an edge, vertex, vertex barely touching other
        // triangle, etc.)

        // Adjacent primitives of the same mesh will touch each other, but they should not count as
        // overlapping.
        auto const nSelfOverlappingPrimitives = bvh.OverlappingPrimitives(bvh).size();
        CHECK_EQ(nSelfOverlappingPrimitives, 0);

        // If points haven't changed, update should preserve the same volumes.
        std::vector<geometry::TriangleAabbHierarchy<kDims>::BoundingVolumeType> const bvsExpected =
            bvh.GetBoundingVolumes();
        bvh.Update();
        auto const nBvs = bvh.GetBoundingVolumes().size();
        for (auto b = std::size_t{0}; b < nBvs; ++b)
        {
            auto const& bv         = bvh.GetBoundingVolumes().at(b);
            auto const& bvExpected = bvsExpected.at(b);
            CHECK(bv.min().isApprox(bvExpected.min()));
            CHECK(bv.max().isApprox(bvExpected.max()));
        }

        // The root volume should contain all the primitive locations
        auto const& rootBv = bvh.GetBoundingVolumes().front();
        for (auto p = 0; p < bvh.C.cols(); ++p)
        {
            auto const X = bvh.PrimitiveLocation(bvh.Primitive(p));
            CHECK(rootBv.contains(X));
        }
    }
}