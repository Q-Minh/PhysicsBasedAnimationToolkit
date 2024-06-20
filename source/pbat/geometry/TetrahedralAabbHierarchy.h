#ifndef PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
#define PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"
#include "DistanceQueries.h"
#include "OverlapQueries.h"

#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace geometry {

class TetrahedralAabbHierarchy : public BoundingVolumeHierarchy<
                                     TetrahedralAabbHierarchy,
                                     AxisAlignedBoundingBox<3>,
                                     IndexVector<4>,
                                     3>
{
  public:
    static auto constexpr kDims = 3;

    TetrahedralAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& T,
        std::size_t maxPointsInLeaf = 10ULL);

    PrimitiveType Primitive(Index p) const;

    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    IndexMatrixX OverlappingPrimitives(
        TetrahedralAabbHierarchy const& tetbbh,
        std::size_t reserve = 1000ULL) const;

    template <class TDerivedP>
    std::vector<Index> PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    template <class TDerivedP>
    std::vector<Index> NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> T;
};

template <class RPrimitiveIndices>
inline TetrahedralAabbHierarchy::BoundingVolumeType
TetrahedralAabbHierarchy::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = T(Eigen::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::all, vertices));
}

template <class TDerivedP>
inline std::vector<Index> TetrahedralAabbHierarchy::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    std::vector<Index> p(static_cast<std::size_t>(P.cols()), -1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                auto const VT = V(Eigen::all, T);
                return OverlapQueries::PointTetrahedron(
                    P.col(i).head<3>(),
                    VT.col(0).head<3>(),
                    VT.col(1).head<3>(),
                    VT.col(2).head<3>(),
                    VT.col(3).head<3>());
            });
        if (not intersectingPrimitives.empty())
            p[static_cast<std::size_t>(i)] = intersectingPrimitives.front();
    };
    if (bParallelize)
    {
        for (auto i = 0; i < P.cols(); ++i)
            FindContainingPrimitive(i);
    }
    else
    {
        tbb::parallel_for(Index{0}, Index{P.cols()}, FindContainingPrimitive);
    }
    return p;
}

template <class TDerivedP>
inline std::vector<Index> TetrahedralAabbHierarchy::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    std::vector<Index> p(static_cast<std::size_t>(P.cols()), -1);
    auto const FindNearestPrimitive = [&](Index i) {
        std::size_t constexpr K{1};
        std::vector<Index> const nearestPrimitives = this->NearestPrimitivesTo(
            [&](BoundingVolumeType const& bv) -> Scalar {
                return bv.squaredExteriorDistance(P.col(i));
            },
            [&](PrimitiveType const& T) -> Scalar {
                auto const VT = V(Eigen::all, T);
                return DistanceQueries::PointTetrahedron(
                    P.col(i).head<3>(),
                    VT.col(0).head<3>(),
                    VT.col(1).head<3>(),
                    VT.col(2).head<3>(),
                    VT.col(3).head<3>());
            },
            K);
        p[static_cast<std::size_t>(i)] = nearestPrimitives.front();
    };
    if (bParallelize)
    {
        for (auto i = 0; i < P.cols(); ++i)
            FindNearestPrimitive(i);
    }
    else
    {
        tbb::parallel_for(Index{0}, Index{P.cols()}, FindNearestPrimitive);
    }
    return p;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
