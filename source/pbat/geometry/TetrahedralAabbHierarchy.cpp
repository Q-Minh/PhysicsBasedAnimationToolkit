#include "TetrahedralAabbHierarchy.h"

#include "OverlapQueries.h"

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
    return V(Eigen::all, primitive).rowwise().mean();
}

IndexMatrixX TetrahedralAabbHierarchy::OverlappingPrimitives(
    TetrahedralAabbHierarchy const& bvh,
    std::size_t reserve) const
{
    return this->OverlappingPrimitivesImpl<
        TetrahedralAabbHierarchy,
        BoundingVolumeType,
        PrimitiveType,
        kDims>(
        bvh,
        [](BoundingVolumeType const& bv1, BoundingVolumeType const& bv2) -> bool {
            return OverlapQueries::AxisAlignedBoundingBoxes(
                bv1.min(),
                bv1.max(),
                bv2.min(),
                bv2.max());
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            auto const V1 = V(Eigen::all, p1);
            auto const V2 = bvh.V(Eigen::all, p2);
            return OverlapQueries::Tetrahedra(
                V1.col(0).head<kDims>(),
                V1.col(1).head<kDims>(),
                V1.col(2).head<kDims>(),
                V1.col(3).head<kDims>(),
                V2.col(0).head<kDims>(),
                V2.col(1).head<kDims>(),
                V2.col(2).head<kDims>(),
                V2.col(3).head<kDims>());
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
    geometry::TetrahedralAabbHierarchy const bvh(V, C, kMaxPointsInLeaf);
    CHECK_EQ(bvh.V.rows(), V.rows());
    CHECK_EQ(bvh.V.cols(), V.cols());
    CHECK_EQ(bvh.C.rows(), C.rows());
    CHECK_EQ(bvh.C.cols(), C.cols());
    auto constexpr kPoints                         = 20;
    Matrix<3, Eigen::Dynamic> P                    = Matrix<3, Eigen::Dynamic>::Random(3, kPoints);
    std::vector<Index> const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
    CHECK_EQ(primitivesContainingP.size(), P.cols());
    geometry::AxisAlignedBoundingBox<3> const domain(
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
    std::vector<Index> const nearestPrimitivesToP = bvh.NearestPrimitivesToPoints(P);
    CHECK_EQ(nearestPrimitivesToP.size(), P.cols());
    for (auto i = 0; i < P.cols(); ++i)
    {
        auto const primitiveIdx = nearestPrimitivesToP[static_cast<std::size_t>(i)];
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
}