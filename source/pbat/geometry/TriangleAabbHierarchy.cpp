#include "TriangleAabbHierarchy.h"

namespace pbat::geometry {

TriangleAabbHierarchy3D::TriangleAabbHierarchy3D(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C,
    std::size_t maxPointsInLeaf)
    : V(V), C(C)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.Construct");
    auto constexpr kRowsC = static_cast<int>(PrimitiveType::RowsAtCompileTime);
    if (V.rows() != kDims and C.rows() != kRowsC)
    {
        std::string const what = fmt::format(
            "Expected vertex positions V of dimensions {}x|#verts| and triangle vertex indices "
            "C of dimensions {}x|#triangles|, but got V={}x{} and C={}x{}.",
            kDims,
            kRowsC,
            V.rows(),
            V.cols(),
            C.rows(),
            C.cols());
        throw std::invalid_argument(what);
    }
    Construct(static_cast<std::size_t>(C.cols()), maxPointsInLeaf);
}

TriangleAabbHierarchy3D::PrimitiveType TriangleAabbHierarchy3D::Primitive(Index p) const
{
    PrimitiveType const inds = C.col(p);
    return inds;
}

Vector<TriangleAabbHierarchy3D::kDims>
TriangleAabbHierarchy3D::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::placeholders::all, primitive).rowwise().mean();
}

void TriangleAabbHierarchy3D::Update()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.Update");
    BaseType::Update();
}

IndexMatrixX
TriangleAabbHierarchy3D::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.OverlappingPrimitives");
    using math::linalg::mini::FromEigen;
    return this->OverlappingPrimitivesImpl<SelfType, BoundingVolumeType, PrimitiveType, kDims>(
        bvh,
        [](BoundingVolumeType const& bv1, BoundingVolumeType const& bv2) -> bool {
            return OverlapQueries::AxisAlignedBoundingBoxes(
                FromEigen(bv1.min()),
                FromEigen(bv1.max()),
                FromEigen(bv2.min()),
                FromEigen(bv2.max()));
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            auto const V1 = V(Eigen::placeholders::all, p1);
            auto const V2 = bvh.V(Eigen::placeholders::all, p2);
            return OverlapQueries::Triangles3D(
                FromEigen(V1.col(0).head<kDims>()),
                FromEigen(V1.col(1).head<kDims>()),
                FromEigen(V1.col(2).head<kDims>()),
                FromEigen(V2.col(0).head<kDims>()),
                FromEigen(V2.col(1).head<kDims>()),
                FromEigen(V2.col(2).head<kDims>()));
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            if (this == &bvh)
            {
                for (auto i : p1)
                    for (auto j : p2)
                        if (i == j)
                            return true;
            }
            return false;
        },
        reserve);
}

TriangleAabbHierarchy2D::TriangleAabbHierarchy2D(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C,
    std::size_t maxPointsInLeaf)
    : V(V), C(C)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.Construct");
    auto constexpr kRowsC = static_cast<int>(PrimitiveType::RowsAtCompileTime);
    if (V.rows() != kDims and C.rows() != kRowsC)
    {
        std::string const what = fmt::format(
            "Expected vertex positions V of dimensions {}x|#verts| and triangle vertex indices "
            "C of dimensions {}x|#triangles|, but got V={}x{} and C={}x{}.",
            kDims,
            kRowsC,
            V.rows(),
            V.cols(),
            C.rows(),
            C.cols());
        throw std::invalid_argument(what);
    }
    Construct(static_cast<std::size_t>(C.cols()), maxPointsInLeaf);
}

TriangleAabbHierarchy2D::PrimitiveType TriangleAabbHierarchy2D::Primitive(Index p) const
{
    PrimitiveType const inds = C.col(p);
    return inds;
}

Vector<TriangleAabbHierarchy2D::kDims>
TriangleAabbHierarchy2D::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::placeholders::all, primitive).rowwise().mean();
}

void TriangleAabbHierarchy2D::Update()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.Update");
    BaseType::Update();
}

IndexMatrixX
TriangleAabbHierarchy2D::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.OverlappingPrimitives");
    using math::linalg::mini::FromEigen;
    return this->OverlappingPrimitivesImpl<SelfType, BoundingVolumeType, PrimitiveType, kDims>(
        bvh,
        [](BoundingVolumeType const& bv1, BoundingVolumeType const& bv2) -> bool {
            return OverlapQueries::AxisAlignedBoundingBoxes(
                FromEigen(bv1.min()),
                FromEigen(bv1.max()),
                FromEigen(bv2.min()),
                FromEigen(bv2.max()));
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            auto const V1 = V(Eigen::placeholders::all, p1);
            auto const V2 = bvh.V(Eigen::placeholders::all, p2);
            return OverlapQueries::Triangles2D(
                FromEigen(V1.col(0).head<kDims>()),
                FromEigen(V1.col(1).head<kDims>()),
                FromEigen(V1.col(2).head<kDims>()),
                FromEigen(V2.col(0).head<kDims>()),
                FromEigen(V2.col(1).head<kDims>()),
                FromEigen(V2.col(2).head<kDims>()));
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            if (this == &bvh)
            {
                for (auto i : p1)
                    for (auto j : p2)
                        if (i == j)
                            return true;
            }
            return false;
        },
        reserve);
}

} // namespace pbat::geometry

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
        geometry::TriangleAabbHierarchy2D bvh(V, C, kMaxPointsInLeaf);
        CHECK_EQ(bvh.V.rows(), V.rows());
        CHECK_EQ(bvh.V.cols(), V.cols());
        CHECK_EQ(bvh.C.rows(), C.rows());
        CHECK_EQ(bvh.C.cols(), C.cols());
        auto constexpr kPoints          = 20;
        Matrix<kDims, Eigen::Dynamic> P = Matrix<kDims, Eigen::Dynamic>::Random(kDims, kPoints);
        IndexVectorX const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
        geometry::AxisAlignedBoundingBox<kDims> const domain(
            V.rowwise().minCoeff(),
            V.rowwise().maxCoeff());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = primitivesContainingP(i);
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
            auto const primitiveIdx = nearestPrimitivesToP(i);
            CHECK_GT(primitiveIdx, Index{-1});
        }

        geometry::TriangleAabbHierarchy2D const otherBvh{bvh};
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
        std::vector<geometry::TriangleAabbHierarchy2D::BoundingVolumeType> const bvsExpected =
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
        geometry::TriangleAabbHierarchy3D bvh(V, C, kMaxPointsInLeaf);
        CHECK_EQ(bvh.V.rows(), V.rows());
        CHECK_EQ(bvh.V.cols(), V.cols());
        CHECK_EQ(bvh.C.rows(), C.rows());
        CHECK_EQ(bvh.C.cols(), C.cols());
        auto constexpr kPoints          = 20;
        Matrix<kDims, Eigen::Dynamic> P = Matrix<kDims, Eigen::Dynamic>::Random(kDims, kPoints);
        IndexVectorX const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
        geometry::AxisAlignedBoundingBox<kDims> const domain(
            V.rowwise().minCoeff(),
            V.rowwise().maxCoeff());
        for (auto i = 0; i < P.cols(); ++i)
        {
            auto const primitiveIdx = primitivesContainingP(i);
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
            auto const primitiveIdx = nearestPrimitivesToP(i);
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
        std::vector<geometry::TriangleAabbHierarchy3D::BoundingVolumeType> const bvsExpected =
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