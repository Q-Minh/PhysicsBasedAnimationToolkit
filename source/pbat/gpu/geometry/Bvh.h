#ifndef PBAT_GPU_GEOMETRY_BVH_H
#define PBAT_GPU_GEOMETRY_BVH_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "Primitives.h"
#include "pbat/gpu/Aliases.h"

#include <Eigen/Core>
#include <cstddef>
#include <limits>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
class Bvh;
} // namespace impl

/**
 * @brief Linear BVH GPU implementation
 */
class Bvh
{
  public:
    using MortonCodeType = std::uint32_t;

    /**
     * @brief
     * @param nPrimitives
     * @param nOverlaps
     */
    PBAT_API Bvh(GpuIndex nPrimitives, GpuIndex nOverlaps);

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
        Points const& P,
        Simplices const& S,
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max,
        GpuScalar expansion = std::numeric_limits<GpuScalar>::epsilon());

    PBAT_API impl::Bvh* Impl();
    PBAT_API impl::Bvh const* Impl() const;

    /**
     * @brief
     * @param S The simplices which were used to build this BVH
     */
    // PBAT_API GpuIndexMatrixX DetectSelfOverlaps(Simplices const& S);
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
    PBAT_API GpuIndexVectorX SimplexOrdering() const;
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

    PBAT_API ~Bvh();

  private:
    impl::Bvh* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_BVH_H
