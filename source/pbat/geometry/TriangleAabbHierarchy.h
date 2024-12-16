#ifndef PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
#define PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"
#include "DistanceQueries.h"
#include "OverlapQueries.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <limits>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>
#include <utility>

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

    [[maybe_unused]] TriangleAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        std::size_t maxPointsInLeaf = 10ULL);

    PrimitiveType Primitive(Index p) const;

    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;

    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;

    [[maybe_unused]] void Update();

    [[maybe_unused]] IndexMatrixX
    OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;

    template <class TDerivedP>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = true) const;

    template <class TDerivedP, class FCull>
    std::pair<IndexVectorX, VectorX> NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        FCull fCull,
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

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> C;
};

inline TriangleAabbHierarchy<3>::TriangleAabbHierarchy(
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
    return V(Eigen::placeholders::all, primitive).rowwise().mean();
}

inline void TriangleAabbHierarchy<3>::Update()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.Update");
    BaseType::Update();
}

inline IndexMatrixX
TriangleAabbHierarchy<3>::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
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

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy<3>::BoundingVolumeType
TriangleAabbHierarchy<3>::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::placeholders::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::placeholders::all, vertices));
}

template <class TDerivedP>
inline IndexVectorX TriangleAabbHierarchy<3>::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.PrimitivesContainingPoints");
    using math::linalg::mini::FromEigen;
    IndexVectorX p(P.cols());
    p.setConstant(-1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                auto const VT  = V(Eigen::placeholders::all, T);
                Scalar const d = DistanceQueries::PointTriangle(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()));
                auto constexpr eps = 1e-15;
                return d < eps;
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

template <class TDerivedP, class FCull>
inline std::pair<IndexVectorX, VectorX> TriangleAabbHierarchy<3>::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    FCull fCull,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy3D.NearestPrimitivesToPoints");
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
                if (fCull(i, T))
                {
                    return std::numeric_limits<Scalar>::max();
                }
                auto const VT = V(Eigen::placeholders::all, T);
                return DistanceQueries::PointTriangle(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()));
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

template <class TDerivedP>
inline std::pair<IndexVectorX, VectorX> TriangleAabbHierarchy<3>::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    return NearestPrimitivesToPoints(P, [](auto, auto) { return false; }, bParallelize);
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

    [[maybe_unused]] void Update();

    [[maybe_unused]] IndexMatrixX
    OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;

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

    Eigen::Ref<MatrixX const> V;
    Eigen::Ref<IndexMatrixX const> C;
};

[[maybe_unused]] inline TriangleAabbHierarchy<2>::TriangleAabbHierarchy(
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
    return V(Eigen::placeholders::all, primitive).rowwise().mean();
}

inline void TriangleAabbHierarchy<2>::Update()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.Update");
    BaseType::Update();
}

inline IndexMatrixX
TriangleAabbHierarchy<2>::OverlappingPrimitives(SelfType const& bvh, std::size_t reserve) const
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

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy<2>::BoundingVolumeType
TriangleAabbHierarchy<2>::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::placeholders::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::placeholders::all, vertices));
}

template <class TDerivedP>
inline IndexVectorX TriangleAabbHierarchy<2>::PrimitivesContainingPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.PrimitivesContainingPoints");
    using math::linalg::mini::FromEigen;
    IndexVectorX p(P.cols());
    p.setConstant(-1);
    auto const FindContainingPrimitive = [&](Index i) {
        std::vector<Index> const intersectingPrimitives = this->PrimitivesIntersecting(
            [&](BoundingVolumeType const& bv) -> bool { return bv.contains(P.col(i)); },
            [&](PrimitiveType const& T) -> bool {
                auto const VT = V(Eigen::placeholders::all, T);
                return OverlapQueries::PointTriangle(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()));
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
inline std::pair<IndexVectorX, VectorX> TriangleAabbHierarchy<2>::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.TriangleAabbHierarchy2D.NearestPrimitivesToPoints");
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
                return DistanceQueries::PointTriangle(
                    FromEigen(P.col(i).template head<kDims>()),
                    FromEigen(VT.col(0).head<kDims>()),
                    FromEigen(VT.col(1).head<kDims>()),
                    FromEigen(VT.col(2).head<kDims>()));
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

#endif // PBAT_GEOMETRY_TRIANGLE_AABB_HIERARCHY_H
