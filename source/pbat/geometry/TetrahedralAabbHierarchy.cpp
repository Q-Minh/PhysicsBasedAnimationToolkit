#include "TetrahedralAabbHierarchy.h"

#include "OverlapQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace geometry {

TetrahedralAabbHierarchy::TetrahedralAabbHierarchy(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C,
    std::size_t maxPointsInLeaf)
    : V(V), C(C)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TetrahedralAabbHierarchy.Construct");
    auto constexpr kRowsC = static_cast<int>(PrimitiveType::RowsAtCompileTime);
    if (V.rows() != kDims and C.rows() != kRowsC)
    {
        std::string const what = fmt::format(
            "Expected vertex positions V of dimensions {}x|#verts| and tetrahedral vertex indices "
            "T of dimensions {}x|#tets|, but got V={}x{} and T={}x{}.",
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

TetrahedralAabbHierarchy::PrimitiveType TetrahedralAabbHierarchy::Primitive(Index p) const
{
    PrimitiveType const inds = C.col(p);
    return inds;
}

Vector<TetrahedralAabbHierarchy::kDims>
TetrahedralAabbHierarchy::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::placeholders::all, primitive).rowwise().mean();
}

void TetrahedralAabbHierarchy::Update()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TetrahedralAabbHierarchy.Update");
    BaseType::Update();
}

IndexMatrixX TetrahedralAabbHierarchy::OverlappingPrimitives(
    TetrahedralAabbHierarchy const& bvh,
    std::size_t reserve) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TetrahedralAabbHierarchy.OverlappingPrimitives");
    using math::linalg::mini::FromEigen;
    return this->OverlappingPrimitivesImpl<
        TetrahedralAabbHierarchy,
        BoundingVolumeType,
        PrimitiveType,
        kDims>(
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
            return OverlapQueries::Tetrahedra(
                FromEigen(V1.col(0).head<kDims>()),
                FromEigen(V1.col(1).head<kDims>()),
                FromEigen(V1.col(2).head<kDims>()),
                FromEigen(V1.col(3).head<kDims>()),
                FromEigen(V2.col(0).head<kDims>()),
                FromEigen(V2.col(1).head<kDims>()),
                FromEigen(V2.col(2).head<kDims>()),
                FromEigen(V2.col(3).head<kDims>()));
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

} // namespace geometry
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[geometry] TetrahedralAabbHierarchy")
{
    using namespace pbat;

    // Cube tetrahedral mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on

    std::size_t constexpr kMaxPointsInLeaf = 1ULL;
    geometry::TetrahedralAabbHierarchy bvh(V, C, kMaxPointsInLeaf);
    CHECK_EQ(bvh.V.rows(), V.rows());
    CHECK_EQ(bvh.V.cols(), V.cols());
    CHECK_EQ(bvh.C.rows(), C.rows());
    CHECK_EQ(bvh.C.cols(), C.cols());
    auto constexpr kPoints                   = 20;
    Matrix<3, Eigen::Dynamic> P              = Matrix<3, Eigen::Dynamic>::Random(3, kPoints);
    IndexVectorX const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
    CHECK_EQ(primitivesContainingP.size(), P.cols());
    geometry::AxisAlignedBoundingBox<3> const domain(
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

    geometry::TetrahedralAabbHierarchy const otherBvh{bvh};
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
    std::vector<geometry::TetrahedralAabbHierarchy::BoundingVolumeType> const bvsExpected =
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