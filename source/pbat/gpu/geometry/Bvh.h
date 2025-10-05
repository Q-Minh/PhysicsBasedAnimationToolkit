/**
 * @file Bvh.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Linear BVH GPU implementation
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_GEOMETRY_BVH_H
#define PBAT_GPU_GEOMETRY_BVH_H

#include "Aabb.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/geometry/Morton.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.h"

#include <Eigen/Core>
#include <cstddef>
#include <limits>

namespace pbat::gpu::impl::geometry {
class Bvh;
} // namespace pbat::gpu::impl::geometry

namespace pbat {
namespace gpu {
namespace geometry {

/**
 * @brief GPU linear BVH \cite karras2012maxpartree public API
 */
class Bvh
{
  public:
    using MortonCodeType = pbat::geometry::MortonCodeType; ///< Type of the Morton codes

    /**
     * @brief Construct a new Bvh object with space allocated for nBoxes and at most nOverlaps
     * @param nBoxes Number of boxes
     * @param nOverlaps Maximum number of overlaps
     */
    PBAT_API Bvh(GpuIndex nBoxes, GpuIndex nOverlaps);
    Bvh(Bvh const&)            = delete;
    Bvh& operator=(Bvh const&) = delete;
    /**
     * @brief Move constructor
     * @param other Bvh to move from
     */
    PBAT_API Bvh(Bvh&& other) noexcept;
    /**
     * @brief Move assignment
     * @param other Bvh to move from
     * @return Reference to this
     */
    PBAT_API Bvh& operator=(Bvh&& other) noexcept;
    /**
     * @brief Build the BVH from the given AABBs
     * @param aabbs Handle to the AABBs
     * @param min Minimum of the world's bounding box
     * @param max Maximum of the world's bounding box
     */
    PBAT_API void Build(
        Aabb& aabbs,
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief Detect overlaps between the AABBs
     * @param aabbs The AABBs which were used to build this BVH
     */
    PBAT_API GpuIndexMatrixX DetectOverlaps(Aabb const& aabbs);
    /**
     * @brief Detect overlaps between the AABBs of different sets
     * @param set |# aabbs| map of indices of aabbs to their corresponding set, i.e. set[i] = j
     * means that aabb i belongs to set j. Must be a 1D Buffer of type GpuIndex of the same size as
     * aabbs.
     * @param aabbs The AABBs which were used to build this BVH
     * @return 2x|# Foverlaps| matrix of overlap pairs between boxes of different sets
     */
    PBAT_API GpuIndexMatrixX DetectOverlaps(common::Buffer const& set, Aabb const& aabbs);
    /**
     * @brief Compute nearest triangles (V,F) to points X, given the triangle AABBs
     * @param aabbs The AABBs of the triangles
     * @param X `3x|# pts|` matrix of NN query points
     * @param V `3x|# verts|` matrix of vertices
     * @param F `3x|# triangles|` matrix of triangle vertex indices
     * @return `|# X|` matrix of nearest triangles to corresponding columns in `X`
     */
    PBAT_API GpuIndexMatrixX PointTriangleNearestNeighbors(
        Aabb const& aabbs,
        common::Buffer const& X,
        common::Buffer const& V,
        common::Buffer const& F);
    /**
     * @brief Compute nearest tets (V,T) to points X, given the tetrahedron AABBs
     * @param aabbs The AABBs of the tets
     * @param X `3x|# pts|` matrix of NN query points
     * @param V `3x|# verts|` matrix of vertices
     * @param T `4x|# tets|` matrix of tet vertex indices
     * @return `|# X|` matrix of nearest tets to corresponding columns in `X`
     */
    PBAT_API GpuIndexMatrixX PointTetrahedronNearestNeighbors(
        Aabb const& aabbs,
        common::Buffer const& X,
        common::Buffer const& V,
        common::Buffer const& T);
    /**
     * @brief BVH nodes' box minimums
     * @return `dims x |# aabbs|` array of box lower bounds
     */
    PBAT_API GpuMatrixX Min() const;
    /**
     * @brief BVH nodes' box maximums
     * @return `dims x |# aabbs|` array of box upper bounds
     */
    PBAT_API GpuMatrixX Max() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexVectorX LeafOrdering() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Vector<MortonCodeType, Eigen::Dynamic> MortonCodes() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexMatrixX Child() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexVectorX Parent() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexMatrixX Rightmost() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuIndexVectorX Visits() const;
    /**
     * @brief Handle to the implementation
     * @return Handle to the implementation
     */
    PBAT_API impl::geometry::Bvh* Impl();
    /**
     * @brief Handle to the implementation
     * @return Handle to the implementation
     */
    PBAT_API impl::geometry::Bvh const* Impl() const;
    /**
     * @brief Destructor
     */
    PBAT_API ~Bvh();

  private:
    void Deallocate();

    impl::geometry::Bvh* mImpl;
    void* mOverlaps; ///< gpu::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_H
