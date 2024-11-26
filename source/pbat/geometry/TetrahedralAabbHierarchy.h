#ifndef PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
#define PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"
#include "DistanceQueries.h"
#include "OverlapQueries.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <limits>
#include <tbb/parallel_for.h>
#include <utility>

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
    using SelfType              = TetrahedralAabbHierarchy;
    using BaseType =
        BoundingVolumeHierarchy<SelfType, AxisAlignedBoundingBox<kDims>, IndexVector<4>, kDims>;

    PBAT_API TetrahedralAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        std::size_t maxPointsInLeaf = 10ULL);

    PBAT_API PrimitiveType Primitive(Index p) const;

    PBAT_API Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    PBAT_API void Update();

    PBAT_API IndexMatrixX
    OverlappingPrimitives(TetrahedralAabbHierarchy const& bvh, std::size_t reserve = 1000ULL) const;

    template <class TDerivedP, class FCull>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        FCull fCull,
        bool bParallelize = true) const;

    template <class TDerivedP>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = true) const;

    template <class TDerivedP>
    std::pair<IndexVectorX, VectorX> NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = true) const;

    [[maybe_unused]] auto const& GetBoundingVolumes() const { return mBoundingVolumes; }

    template <class TDerivedP>
    void SetV(Eigen::MatrixBase<TDerivedP> const& P)
    {
        V = P;
    }

    [[maybe_unused]] auto GetV() const { return V; }

    [[maybe_unused]] auto GetC() const { return C; }

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> C;
};

template <class RPrimitiveIndices>
inline TetrahedralAabbHierarchy::BoundingVolumeType
TetrahedralAabbHierarchy::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::placeholders::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::placeholders::all, vertices));
}

template <class TDerivedP, class FCull>
inline IndexVectorX TetrahedralAabbHierarchy::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    FCull fCull,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TetrahedralAabbHierarchy.PrimitivesContainingPoints");
    using math::linalg::mini::FromEigen;
    IndexVectorX p(P.cols());
    p.setConstant(-1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                if (fCull(i, T))
                    return false;
                auto const VT = V(Eigen::placeholders::all, T);
                return OverlapQueries::PointTetrahedron3D(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()),
                    FromEigen(VT.col(3).head<kDims>()));
            });
        if (not intersectingPrimitives.empty())
        {
            p(i) = intersectingPrimitives.front();
        }
    };
    if (bParallelize)
    {
        tbb::parallel_for(Index{0}, Index{P.cols()}, FindContainingPrimitive);
    }
    else
    {
        for (auto i = 0; i < P.cols(); ++i)
            FindContainingPrimitive(i);
    }
    return p;
}

template <class TDerivedP>
inline IndexVectorX TetrahedralAabbHierarchy::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    return PrimitivesContainingPoints(P, [](auto, auto) { return false; }, bParallelize);
}

template <class TDerivedP>
inline std::pair<IndexVectorX, VectorX> TetrahedralAabbHierarchy::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TetrahedralAabbHierarchy.NearestPrimitivesToPoints");
    using math::linalg::mini::FromEigen;
    IndexVectorX p(P.cols());
    p.setConstant(-1);
    VectorX d(P.cols());
    d.setConstant(std::numeric_limits<Scalar>::max());
    auto const FindNearestPrimitive = [&](Index i) {
        std::size_t constexpr K{1};
        auto const [nearestPrimitives, distances] = this->NearestPrimitivesTo(
            [&](BoundingVolumeType const& bv) -> Scalar {
                return bv.squaredExteriorDistance(P.col(i));
            },
            [&](PrimitiveType const& T) -> Scalar {
                auto const VT = V(Eigen::placeholders::all, T);
                return DistanceQueries::PointTetrahedron(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()),
                    FromEigen(VT.col(3).head<kDims>()));
            },
            K);
        p(i) = nearestPrimitives.front();
        d(i) = distances.front();
    };
    if (bParallelize)
    {
        tbb::parallel_for(Index{0}, Index{P.cols()}, FindNearestPrimitive);
    }
    else
    {
        for (auto i = 0; i < P.cols(); ++i)
            FindNearestPrimitive(i);
    }
    return {p, d};
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_TETRAHEDRAL_AABB_HIERARCHY_H
