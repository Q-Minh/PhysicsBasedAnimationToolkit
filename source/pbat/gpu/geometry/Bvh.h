#ifndef PBAT_GPU_BVH_CUH
#define PBAT_GPU_BVH_CUH

#include "Primitives.h"
#include "pbat/gpu/Aliases.h"
#include "PhysicsBasedAnimationToolkitExport.h"

#include <Eigen/Core>
#include <cstddef>
#include <limits>

namespace pbat {
namespace gpu {
namespace geometry {

class BvhImpl;

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
    PBAT_API Bvh(std::size_t nPrimitives, std::size_t nOverlaps);

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

    PBAT_API BvhImpl* Impl();
    PBAT_API BvhImpl const* Impl() const;

    /**
     * @brief
     * @param S The simplices which were used to build this BVH
     */
    PBAT_API GpuIndexMatrixX DetectSelfOverlaps(Simplices const& S);
    /**
     * @brief BVH nodes' box minimums
     * @return
     */
    PBAT_API Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Min() const;
    /**
     * @brief BVH nodes' box maximums
     * @return
     */
    PBAT_API Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Max() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Vector<GpuIndex, Eigen::Dynamic> SimplexOrdering() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Vector<MortonCodeType, Eigen::Dynamic> MortonCodes() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Child() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Vector<GpuIndex, Eigen::Dynamic> Parent() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Rightmost() const;
    /**
     * @brief
     * @return
     */
    PBAT_API Eigen::Vector<GpuIndex, Eigen::Dynamic> Visits() const;

    PBAT_API ~Bvh();

  private:
    BvhImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_CUH