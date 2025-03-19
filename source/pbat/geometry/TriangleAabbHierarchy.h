/**
 * @file TriangleAabbHierarchy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the TriangleAabbHierarchy classes for 2D and 3D.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_TRIANGLEAABBHIERARCHY_H
#define PBAT_GEOMETRY_TRIANGLEAABBHIERARCHY_H

#include "AxisAlignedBoundingBox.h"
#include "BoundingVolumeHierarchy.h"
#include "DistanceQueries.h"
#include "OverlapQueries.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Eigen.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <limits>
#include <tbb/parallel_for.h>
#include <utility>

namespace pbat::geometry {

/**
 * @brief Bounding volume hierarchy for triangles in 3D.
 */
class TriangleAabbHierarchy3D : public BoundingVolumeHierarchy<
                                    TriangleAabbHierarchy3D,
                                    AxisAlignedBoundingBox<3>,
                                    IndexVector<3>,
                                    3>
{
  public:
    static auto constexpr kDims = 3;                       ///< Number of dimensions
    using SelfType              = TriangleAabbHierarchy3D; ///< Self type
    using BaseType              = BoundingVolumeHierarchy<
                     SelfType,
                     AxisAlignedBoundingBox<kDims>,
                     IndexVector<3>,
                     kDims>; ///< Base type

    /**
     * @brief Construct a triangle Aabb BVH from an input mesh (V,C)
     *
     * @param V `3x|# verts|` matrix of vertex positions
     * @param C `3x|# triangles|` matrix of cell vertex indices
     * @param maxPointsInLeaf Maximum number of simplices in a leaf node
     */
    TriangleAabbHierarchy3D(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        Index maxPointsInLeaf = 10);
    /**
     * @brief Returns the primitive at index p
     * @param p Index of the primitive
     * @return The primitive at index p
     */
    PrimitiveType Primitive(Index p) const;
    /**
     * @brief Returns the location of the primitive
     * @param primitive The primitive
     * @return The location of the primitive
     */
    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;
    /**
     * @brief Returns the bounding volume of the primitive
     * @tparam RPrimitiveIndices Index range type
     * @param pinds Range of primitive indices
     * @return The bounding volume of the primitives pinds
     */
    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;
    /**
     * @brief Updates the AABBs
     */
    void Update();
    /**
     * @brief Returns the overlapping primitives of this BVH and another BVH
     * @param bvh The other BVH
     * @param reserve Estimated number of overlapping primitives to reserve memory for
     * @return `2x|# overlaps|` matrix `O` of overlapping primitive pairs s.t. primitives `O(0,o)`
     * in this bvh, and `O(1,o)` in the other bvh overlap.
     */
    IndexMatrixX OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;
    /**
     * @brief For each point in P, returns the index of the primitive containing it
     * @tparam TDerivedP Eigen matrix type
     * @param P `|kDims|x|# points|` matrix of points
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of primitive indices containing the points
     */
    template <class TDerivedP>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = true) const;
    /**
     * @brief For each point in P, returns the index of the nearest primitive to it
     * @tparam TDerivedP Eigen matrix type
     * @tparam FCull Culling function type
     * @param P `|kDims|x|# points|` matrix of points
     * @param fCull Culling function
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of nearest primitive indices to the points
     */
    template <class TDerivedP, class FCull>
    auto NearestPrimitivesToPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        FCull fCull,
        bool bParallelize = true) const -> std::pair<IndexVectorX, VectorX>;
    /**
     * @brief For each point in P, returns the index of the nearest primitive to it
     * @tparam TDerivedP Eigen matrix type
     * @param P `|kDims|x|# points|` matrix of points
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of nearest primitive indices to the points
     */
    template <class TDerivedP>
    auto
    NearestPrimitivesToPoints(Eigen::MatrixBase<TDerivedP> const& P, bool bParallelize = true) const
        -> std::pair<IndexVectorX, VectorX>;
    /**
     * @brief Returns this BVH's bounding volumes
     * @return This BVH's bounding volumes
     */
    [[maybe_unused]] auto const& GetBoundingVolumes() const { return mBoundingVolumes; }
    /**
     * @brief Updates this BVH's mesh's vertex positions
     * @tparam TDerivedP Eigen matrix type
     * @param P `|kDims|x|# verts|` matrix of vertex positions
     */
    template <class TDerivedP>
    void SetV(Eigen::MatrixBase<TDerivedP> const& P)
    {
        V = P;
    }
    /**
     * @brief Returns this BVH's mesh's vertex positions
     * @return This BVH's mesh's vertex positions
     */
    [[maybe_unused]] auto GetV() const { return V; }

    Eigen::Ref<MatrixX const> V;      ///< `|kDims|x|# verts|` vertex positions
    Eigen::Ref<IndexMatrixX const> C; ///< `3x|# triangles|` triangle vertex indices into V
};

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy3D::BoundingVolumeType
TriangleAabbHierarchy3D::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::placeholders::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::placeholders::all, vertices));
}

template <class TDerivedP>
inline IndexVectorX TriangleAabbHierarchy3D::PrimitivesContainingPoints(
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
inline auto TriangleAabbHierarchy3D::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    FCull fCull,
    bool bParallelize) const -> std::pair<IndexVectorX, VectorX>
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
inline auto TriangleAabbHierarchy3D::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const -> std::pair<IndexVectorX, VectorX>
{
    return NearestPrimitivesToPoints(P, [](auto, auto) { return false; }, bParallelize);
}

/**
 * @brief Bounding volume hierarchy for triangles in 2D.
 */
class TriangleAabbHierarchy2D : public BoundingVolumeHierarchy<
                                    TriangleAabbHierarchy2D,
                                    AxisAlignedBoundingBox<2>,
                                    IndexVector<3>,
                                    2>
{
  public:
    static auto constexpr kDims = 2;                       ///< Number of dimensions
    using SelfType              = TriangleAabbHierarchy2D; ///< Self type
    using BaseType              = BoundingVolumeHierarchy<
                     SelfType,
                     AxisAlignedBoundingBox<kDims>,
                     IndexVector<3>,
                     kDims>; ///< Base type

    /**
     * @brief Construct a triangle Aabb BVH from an input mesh (V,C)
     *
     * @param V `2x|# verts|` matrix of vertex positions
     * @param C `3x|# triangles|` matrix of cell vertex indices
     * @param maxPointsInLeaf Maximum number of simplices in a leaf node
     */
    TriangleAabbHierarchy2D(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        Index maxPointsInLeaf = 10);
    /**
     * @brief Returns the primitive at index p
     * @param p Index of the primitive
     * @return The primitive at index p
     */
    PrimitiveType Primitive(Index p) const;
    /**
     * @brief Returns the location of the primitive
     * @param primitive The primitive
     * @return The location of the primitive
     */
    Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;
    /**
     * @brief Returns the bounding volume of the primitive
     * @tparam RPrimitiveIndices Index range type
     * @param pinds Range of primitive indices
     * @return The bounding volume of the primitives pinds
     */
    template <class RPrimitiveIndices>
    BoundingVolumeType BoundingVolumeOf(RPrimitiveIndices&& pinds) const;
    /**
     * @brief Updates the AABBs
     */
    void Update();
    /**
     * @brief Returns the overlapping primitives of this BVH and another BVH
     * @param bvh The other BVH
     * @param reserve Estimated number of overlapping primitives to reserve memory for
     * @return `2x|# overlaps|` matrix `O` of overlapping primitive pairs s.t. primitives `O(0,o)`
     * in this bvh, and `O(1,o)` in the other bvh overlap.
     */
    IndexMatrixX OverlappingPrimitives(SelfType const& bvh, std::size_t reserve = 1000ULL) const;
    /**
     * @brief For each point in P, returns the index of the primitive containing it
     * @tparam TDerivedP Eigen matrix type
     * @param P `|kDims|x|# points|` matrix of points
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of primitive indices containing the points
     */
    template <class TDerivedP>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        bool bParallelize = true) const;
    /**
     * @brief For each point in P, returns the index of the nearest primitive to it
     * @tparam TDerivedP Eigen matrix type
     * @tparam FCull Culling function type
     * @param P `|kDims|x|# points|` matrix of points
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of nearest primitive indices to the points
     */
    template <class TDerivedP>
    auto
    NearestPrimitivesToPoints(Eigen::MatrixBase<TDerivedP> const& P, bool bParallelize = true) const
        -> std::pair<IndexVectorX, VectorX>;
    /**
     * @brief Returns this BVH's bounding volumes
     * @return This BVH's bounding volumes
     */
    [[maybe_unused]] auto const& GetBoundingVolumes() const { return mBoundingVolumes; }
    /**
     * @brief Updates this BVH's mesh's vertex positions
     * @tparam TDerivedP Eigen matrix type
     * @param P `|kDims|x|# verts|` matrix of vertex positions
     */
    template <class TDerivedP>
    void SetV(Eigen::MatrixBase<TDerivedP> const& P)
    {
        V = P;
    }
    /**
     * @brief Returns this BVH's mesh's vertex positions
     * @return This BVH's mesh's vertex positions
     */
    [[maybe_unused]] auto GetV() const { return V; }

    Eigen::Ref<MatrixX const> V;      ///< `|kDims|x|# verts|` vertex positions
    Eigen::Ref<IndexMatrixX const> C; ///< `3x|# triangles|` triangle vertex indices into V
};

template <class RPrimitiveIndices>
inline TriangleAabbHierarchy2D::BoundingVolumeType
TriangleAabbHierarchy2D::BoundingVolumeOf(RPrimitiveIndices&& pinds) const
{
    auto vertices = C(Eigen::placeholders::all, common::Slice(pinds)).reshaped();
    return BoundingVolumeType(V(Eigen::placeholders::all, vertices));
}

template <class TDerivedP>
inline IndexVectorX TriangleAabbHierarchy2D::PrimitivesContainingPoints(
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
inline std::pair<IndexVectorX, VectorX> TriangleAabbHierarchy2D::NearestPrimitivesToPoints(
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

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_TRIANGLEAABBHIERARCHY_H
