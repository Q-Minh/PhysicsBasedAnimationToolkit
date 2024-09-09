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

    /**
     * @brief
     * @param S The simplices which were used to build this BVH
     */
    GpuIndexMatrixX DetectSelfOverlaps(Simplices const& S);

    ~Bvh();

  private:
    BvhImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_CUH