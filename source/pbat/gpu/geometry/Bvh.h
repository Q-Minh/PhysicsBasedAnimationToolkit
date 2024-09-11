#ifndef PBAT_GPU_BVH_CUH
#define PBAT_GPU_BVH_CUH

#include "Primitives.h"
#include "pbat/gpu/Aliases.h"

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
    Bvh(std::size_t nPrimitives, std::size_t nOverlaps);

    Bvh(Bvh const&)            = delete;
    Bvh& operator=(Bvh const&) = delete;

    Bvh(Bvh&& other) noexcept;
    Bvh& operator=(Bvh&& other) noexcept;

    /**
     * @brief
     * @param P
     * @param S
     * @param expansion
     */
    void Build(
        Points const& P,
        Simplices const& S,
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max,
        GpuScalar expansion = std::numeric_limits<GpuScalar>::epsilon());

    BvhImpl* Impl();
    BvhImpl const* Impl() const;

    /**
     * @brief
     * @param S The simplices which were used to build this BVH
     */
    GpuIndexMatrixX DetectSelfOverlaps(Simplices const& S);
    /**
     * @brief BVH nodes' box minimums
     * @return
     */
    Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Min() const;
    /**
     * @brief BVH nodes' box maximums
     * @return
     */
    Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic> Max() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> SimplexOrdering() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<MortonCodeType, Eigen::Dynamic> MortonCodes() const;
    /**
     * @brief
     * @return
     */
    Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Child() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> Parent() const;
    /**
     * @brief
     * @return
     */
    Eigen::Matrix<GpuIndex, Eigen::Dynamic, 2> Rightmost() const;
    /**
     * @brief
     * @return
     */
    Eigen::Vector<GpuIndex, Eigen::Dynamic> Visits() const;

    ~Bvh();

  private:
    BvhImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_CUH