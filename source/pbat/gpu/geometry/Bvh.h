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
 * @brief Linear BVH GPU implementation
 */
class Bvh
{
  public:
    using MortonCodeType = pbat::geometry::MortonCodeType;

    /**
     * @brief
     * @param nBoxes
     * @param nOverlaps
     */
    PBAT_API Bvh(GpuIndex nBoxes, GpuIndex nOverlaps);

    Bvh(Bvh const&)            = delete;
    Bvh& operator=(Bvh const&) = delete;
    PBAT_API Bvh(Bvh&& other) noexcept;
    PBAT_API Bvh& operator=(Bvh&& other) noexcept;
    /**
     * @brief
     * @param P
     * @param S
     * @param expansion
     */
    PBAT_API void Build(
        Aabb& aabbs,
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief
     * @param S The simplices which were used to build this BVH
     */
    PBAT_API GpuIndexMatrixX DetectOverlaps(Aabb const& aabbs);
    /**
     * @brief
     *
     * @param set |#aabbs| map of indices of aabbs to their corresponding set, i.e. set[i] = j means
     * that aabb i belongs to set j. Must be a 1D Buffer of type GpuIndex of the same size as aabbs.
     * @param aabbs
     * @return 2x|#overlaps| matrix of overlap pairs between boxes of different sets
     */
    PBAT_API GpuIndexMatrixX DetectOverlaps(common::Buffer const& set, Aabb const& aabbs);
    /**
     * @brief
     *
     * @param aabbs
     * @param X 3x|#pts| matrix of NN query points
     * @param V 3x|#verts| matrix of vertices
     * @param F 3x|#triangles| matrix of triangle vertex indices
     * @return |#X| matrix of nearest triangles to corresponding columns in X
     */
    PBAT_API GpuIndexMatrixX PointTriangleNearestNeighbors(
        Aabb const& aabbs,
        common::Buffer const& X,
        common::Buffer const& V,
        common::Buffer const& F);
    /**
     * @brief
     *
     * @param aabbs
     * @param X 3x|#pts| matrix of NN query points
     * @param V 3x|#verts| matrix of vertices
     * @param t 4x|#tets| matrix of tet vertex indices
     * @return |#X| matrix of nearest tets to corresponding columns in X
     */
    PBAT_API GpuIndexMatrixX PointTetrahedronNearestNeighbors(
        Aabb const& aabbs,
        common::Buffer const& X,
        common::Buffer const& V,
        common::Buffer const& T);
    /**
     * @brief BVH nodes' box minimums
     * @return
     */
    PBAT_API GpuMatrixX Min() const;
    /**
     * @brief BVH nodes' box maximums
     * @return
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

    PBAT_API impl::geometry::Bvh* Impl();
    PBAT_API impl::geometry::Bvh const* Impl() const;
    /**
     * @brief
     *
     * @return
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
