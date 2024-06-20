#include "TetrahedralAabbHierarchy.h"

#include "OverlapQueries.h"

#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace geometry {

TetrahedralAabbHierarchy::TetrahedralAabbHierarchy(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& T,
    std::size_t maxPointsInLeaf)
    : V(V), T(T)
{
    auto constexpr kRowsT = static_cast<int>(PrimitiveType::RowsAtCompileTime);
    if (V.rows() != kDims and T.rows() != kRowsT)
    {
        std::string const what = fmt::format(
            "Expected vertex positions V of dimensions {}x|#verts| and tetrahedral vertex indices "
            "T of dimensions {}x|#tets|, but got V={}x{} and T={}x{}.",
            kDims,
            kRowsT,
            V.rows(),
            V.cols(),
            T.rows(),
            T.cols());
        throw std::invalid_argument(what);
    }
    Construct(static_cast<std::size_t>(T.cols()), maxPointsInLeaf);
}

TetrahedralAabbHierarchy::PrimitiveType TetrahedralAabbHierarchy::Primitive(Index p) const
{
    PrimitiveType const inds = T.col(p);
    return inds;
}

Vector<TetrahedralAabbHierarchy::kDims>
TetrahedralAabbHierarchy::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::all, primitive).rowwise().mean();
}

IndexMatrixX TetrahedralAabbHierarchy::OverlappingPrimitives(
    TetrahedralAabbHierarchy const& tetbbh,
    std::size_t reserve) const
{
    return this->OverlappingPrimitivesImpl<
        TetrahedralAabbHierarchy,
        BoundingVolumeType,
        PrimitiveType,
        kDims>(
        tetbbh,
        [](BoundingVolumeType const& bv1, BoundingVolumeType const& bv2) -> bool {
            return OverlapQueries::AxisAlignedBoundingBoxes(
                bv1.min(),
                bv1.max(),
                bv2.min(),
                bv2.max());
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            Matrix<kDims, 4> const V1 = V(Eigen::all, p1);
            Matrix<kDims, 4> const V2 = tetbbh.V(Eigen::all, p2);
            return OverlapQueries::Tetrahedra(
                V1.col(0),
                V1.col(1),
                V1.col(2),
                V1.col(3),
                V2.col(0),
                V2.col(1),
                V2.col(2),
                V2.col(3));
        },
        [&](PrimitiveType const& p1, PrimitiveType const& p2) -> bool {
            if (this == &tetbbh)
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
    CHECK_EQ(bvh.T.rows(), C.rows());
    CHECK_EQ(bvh.T.cols(), C.cols());
    auto constexpr kPoints                         = 20;
    Matrix<3, Eigen::Dynamic> P                    = Matrix<3, Eigen::Dynamic>::Random(3, kPoints);
    std::vector<Index> const primitivesContainingP = bvh.PrimitivesContainingPoints(P);
    CHECK_EQ(primitivesContainingP.size(), P.cols());
    geometry::AxisAlignedBoundingBox<3> const domain(Vector<3>::Zero(), Vector<3>::Ones());
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
}