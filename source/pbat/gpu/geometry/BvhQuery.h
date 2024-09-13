#ifndef PBAT_GPU_BVH_QUERY_CUH
#define PBAT_GPU_BVH_QUERY_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/geometry/Bvh.h"
#include "pbat/gpu/geometry/Primitives.h"

#include <cstddef>

namespace pbat {
namespace gpu {
namespace geometry {

class BvhQueryImpl;

class BvhQuery
{
  public:
    BvhQuery(std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours);

    BvhQuery(BvhQuery const&)            = delete;
    BvhQuery& operator=(BvhQuery const&) = delete;

    BvhQuery(BvhQuery&& other) noexcept;
    BvhQuery& operator=(BvhQuery&& other) noexcept;

    /**
     * @brief
     * @param P
     * @param S
     * @param min
     * @param max
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
     * @param P
     * @param S1
     * @param S2
     * @param bvh
     */
    GpuIndexMatrixX
    DetectOverlaps(Points const& P, Simplices const& S1, Simplices const& S2, Bvh const& bvh);

    ~BvhQuery();

  private:
    BvhQueryImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_QUERY_CUH