#ifndef PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
#define PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"
#include "DistanceQueries.h"
#include "OverlapQueries.h"
#include "PhysicsBasedAnimationToolkitExport.h"

#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace geometry {

template <int Dims>
class TriangleAabbHierarchy;

} // namespace geometry
} // namespace pbat

namespace pbat {
namespace geometry {

template <>
class TriangleAabbHierarchy<3> : public BoundingVolumeHierarchy<
                                     TriangleAabbHierarchy<3>,
                                     AxisAlignedBoundingBox<3>,
                                     IndexVector<3>,
                                     3>
{
  public:
    static auto constexpr kDims = 3;
    using SelfType              = TriangleAabbHierarchy<3>;
    using BaseType =
        BoundingVolumeHierarchy<SelfType, AxisAlignedBoundingBox<kDims>, IndexVector<3>, kDims>;

    TriangleAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        std::size_t maxPointsInLeaf = 10ULL);

    PrimitiveType Primitive(Index p) const;

    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    IndexMatrixX OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;

    template <class TDerivedP>
    std::vector<Index> PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    template <class TDerivedP>
    std::vector<Index> NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    [[maybe_unused]] auto GetBoundingVolumes() const { return mBoundingVolumes; }

    template <class TDerivedP>
    void SetV(Eigen::MatrixBase<TDerivedP> const& P)
    {
        V = P;
    }

    [[maybe_unused]] auto GetV() const { return V; }

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> C;
};

inline TriangleAabbHierarchy<3>::TriangleAabbHierarchy(
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

inline TriangleAabbHierarchy<3>::PrimitiveType TriangleAabbHierarchy<3>::Primitive(Index p) const
{
    PrimitiveType const inds = C.col(p);
    return inds;
}

inline Vector<TriangleAabbHierarchy<3>::kDims>
TriangleAabbHierarchy<3>::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::all, primitive).rowwise().mean();
}

inline IndexMatrixX
TriangleAabbHierarchy<3>::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
{
    return this->OverlappingPrimitivesImpl<SelfType, BoundingVolumeType, PrimitiveType, kDims>(
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
            return OverlapQueries::Triangles3D(
                V1.col(0).head<kDims>(),
                V1.col(1).head<kDims>(),
                V1.col(2).head<kDims>(),
                V2.col(0).head<kDims>(),
                V2.col(1).head<kDims>(),
                V2.col(2).head<kDims>());
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

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy<3>::BoundingVolumeType
TriangleAabbHierarchy<3>::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::all, vertices));
}

template <class TDerivedP>
inline std::vector<Index> TriangleAabbHierarchy<3>::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    std::vector<Index> p(static_cast<std::size_t>(P.cols()), -1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                auto const VT  = V(Eigen::all, T);
                Scalar const d = DistanceQueries::PointTriangle(
                    P.col(i).head<kDims>(),
                    VT.col(0).head<kDims>(),
                    VT.col(1).head<kDims>(),
                    VT.col(2).head<kDims>());
                auto constexpr eps = 1e-15;
                return d < eps;
            });
        if (not intersectingPrimitives.empty())
        {
            auto const iStl = static_cast<std::size_t>(i);
            p[iStl]         = intersectingPrimitives.front();
        }
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
inline std::vector<Index> TriangleAabbHierarchy<3>::NearestPrimitivesToPoints(
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
                return DistanceQueries::PointTriangle(
                    P.col(i).head<kDims>(),
                    VT.col(0).head<kDims>(),
                    VT.col(1).head<kDims>(),
                    VT.col(2).head<kDims>());
            },
            K);
        auto const iStl = static_cast<std::size_t>(i);
        p[iStl]         = nearestPrimitives.front();
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

namespace pbat {
namespace geometry {

template <>
class TriangleAabbHierarchy<2> : public BoundingVolumeHierarchy<
                                     TriangleAabbHierarchy<2>,
                                     AxisAlignedBoundingBox<2>,
                                     IndexVector<3>,
                                     2>
{
  public:
    static auto constexpr kDims = 2;
    using SelfType              = TriangleAabbHierarchy<kDims>;
    using BaseType =
        BoundingVolumeHierarchy<SelfType, AxisAlignedBoundingBox<kDims>, IndexVector<3>, kDims>;

    TriangleAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        std::size_t maxPointsInLeaf = 10ULL);

    PrimitiveType Primitive(Index p) const;

    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    IndexMatrixX OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;

    template <class TDerivedP>
    std::vector<Index> PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    template <class TDerivedP>
    std::vector<Index> NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = false) const;

    [[maybe_unused]] auto GetBoundingVolumes() const { return mBoundingVolumes; }

    template <class TDerivedP>
    void SetV(Eigen::MatrixBase<TDerivedP> const& P)
    {
        V = P;
    }

    [[maybe_unused]] auto GetV() const { return V; }

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> C;
};

inline TriangleAabbHierarchy<2>::TriangleAabbHierarchy(
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

inline TriangleAabbHierarchy<2>::PrimitiveType TriangleAabbHierarchy<2>::Primitive(Index p) const
{
    PrimitiveType const inds = C.col(p);
    return inds;
}

inline Vector<TriangleAabbHierarchy<2>::kDims>
TriangleAabbHierarchy<2>::PrimitiveLocation(PrimitiveType const& primitive) const
{
    return V(Eigen::all, primitive).rowwise().mean();
}

inline IndexMatrixX
TriangleAabbHierarchy<2>::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
{
    return this->OverlappingPrimitivesImpl<SelfType, BoundingVolumeType, PrimitiveType, kDims>(
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
            return OverlapQueries::Triangles2D(
                V1.col(0).head<kDims>(),
                V1.col(1).head<kDims>(),
                V1.col(2).head<kDims>(),
                V2.col(0).head<kDims>(),
                V2.col(1).head<kDims>(),
                V2.col(2).head<kDims>());
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

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy<2>::BoundingVolumeType
TriangleAabbHierarchy<2>::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::all, vertices));
}

template <class TDerivedP>
inline std::vector<Index> TriangleAabbHierarchy<2>::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    std::vector<Index> p(static_cast<std::size_t>(P.cols()), -1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                auto const VT = V(Eigen::all, T);
                return OverlapQueries::PointTriangle(
                    P.col(i).head<kDims>(),
                    VT.col(0).head<kDims>(),
                    VT.col(1).head<kDims>(),
                    VT.col(2).head<kDims>());
            });
        if (not intersectingPrimitives.empty())
        {
            auto const iStl = static_cast<std::size_t>(i);
            p[iStl]         = intersectingPrimitives.front();
        }
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
inline std::vector<Index> TriangleAabbHierarchy<2>::NearestPrimitivesToPoints(
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
                return DistanceQueries::PointTriangle(
                    P.col(i).head<kDims>(),
                    VT.col(0).head<kDims>(),
                    VT.col(1).head<kDims>(),
                    VT.col(2).head<kDims>());
            },
            K);
        auto const iStl = static_cast<std::size_t>(i);
        p[iStl]         = nearestPrimitives.front();
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

#endif // PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
