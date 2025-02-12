/**
 * @file TetrahedralAabbHierarchy.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the TetrahedralAabbHierarchy class.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_TETRAHEDRALAABBHIERARCHY_H
#define PBAT_GEOMETRY_TETRAHEDRALAABBHIERARCHY_H

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

/**
 * @brief Tetrahedral AABB hierarchy class.
 */
class TetrahedralAabbHierarchy : public BoundingVolumeHierarchy<
                                     TetrahedralAabbHierarchy,
                                     AxisAlignedBoundingBox<3>,
                                     IndexVector<4>,
                                     3>
{
  public:
    static auto constexpr kDims = 3;                        ///< Dimension of the space
    using SelfType              = TetrahedralAabbHierarchy; ///< Type of this class
    using BaseType              = BoundingVolumeHierarchy<
                     SelfType,
                     AxisAlignedBoundingBox<kDims>,
                     IndexVector<4>,
                     kDims>; ///< Base type

    /**
     * @brief Construct a TetrahedralAabbHierarchy from a tetrahedral mesh (V,C)
     * @param V `|kDims|x|# verts|` vertex positions
     * @param C `4x|# tetrahedra|` cell vertex indices into V
     * @param maxPointsInLeaf Maximum number of simplices in a leaf node
     */
    PBAT_API TetrahedralAabbHierarchy(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        std::size_t maxPointsInLeaf = 10ULL);

    /**
     * @brief Returns the primitive at index p
     * @param p Index of the primitive
     * @return The primitive at index p
     */
    PBAT_API PrimitiveType Primitive(Index p) const;
    /**
     * @brief Returns the location of the primitive
     * @param primitive The primitive
     * @return The location of the primitive
     */
    PBAT_API Vector<kDims> PrimitiveLocation(PrimitiveType const& primitive) const;
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
    PBAT_API void Update();
    /**
     * @brief Returns the overlapping primitives of this BVH and another BVH
     * @param bvh The other BVH
     * @param reserve Estimated number of overlapping primitives to reserve memory for
     * @return `2x|# overlaps|` matrix `O` of overlapping primitive pairs s.t. primitives `O(0,o)`
     * in this bvh, and `O(1,o)` in the other bvh overlap.
     */
    PBAT_API IndexMatrixX
    OverlappingPrimitives(TetrahedralAabbHierarchy const& bvh, std::size_t reserve = 1000ULL) const;
    /**
     * @brief For each point in P, returns the index of the primitive containing it
     * @tparam TDerivedP Eigen matrix type
     * @tparam FCull Culling function type
     * @param P `|kDims|x|# points|` matrix of points
     * @param fCull Culling function
     * @param bParallelize Whether to parallelize the computation
     * @return `|# points|` vector of primitive indices containing the points
     */
    template <class TDerivedP, class FCull>
    IndexVectorX PrimitivesContainingPoints(
        Eigen::MatrixBase<TDerivedP> const& P,
        FCull fCull,
        bool bParallelize = true) const;
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
    /**
     * @brief Returns this BVH's mesh's cell indices
     * @return This BVH's mesh's cell indices
     */
    [[maybe_unused]] auto GetC() const { return C; }

    Eigen::Ref<MatrixX const> V;      ///< `|kDims|x|# verts|` vertex positions
    Eigen::Ref<IndexMatrixX const> C; ///< `4x|# tetrahedra|` cell indices into V
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
inline auto TetrahedralAabbHierarchy::NearestPrimitivesToPoints(
    Eigen::MatrixBase<TDerivedP> const& P,
    bool bParallelize) const -> std::pair<IndexVectorX, VectorX>
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

#endif // PBAT_GEOMETRY_TETRAHEDRALAABBHIERARCHY_H
