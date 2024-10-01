#ifndef PBAT_GPU_BVH_QUERY_CUH
#define PBAT_GPU_BVH_QUERY_CUH

#include "PhysicsBasedAnimationToolkitExport.h"
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
    PBAT_API
    BvhQuery(std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours);

    BvhQuery(BvhQuery const&)            = delete;
    BvhQuery& operator=(BvhQuery const&) = delete;

    PBAT_API BvhQuery(BvhQuery&& other) noexcept;
    PBAT_API BvhQuery& operator=(BvhQuery&& other) noexcept;

    /**
     * @brief
     * @param P
     * @param S
     * @param min
     * @param max
     * @param expansion
     */
    PBAT_API void Build(
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
     * @returns 2x|#overlaps| matrix O of overlapping simplices, such that (O(0,k), O(1,k)) yields
     * the k^{th} overlapping pair (si \in S1,sj \in S2).
     */
    PBAT_API GpuIndexMatrixX
    DetectOverlaps(Points const& P, Simplices const& S1, Simplices const& S2, Bvh const& bvh);

    /**
     * @brief
     * @param P Simplex primitive vertex positions
     * @param S1 Query primitives
     * @param S2 Target primitives
     * @param B1 Query simplex bodies
     * @param B2 Target simplex bodies
     * @param bvh Bounding volume hierarchy over S2
     * @param dhat Radius of nearest neighbour search space for each query primitive in S1
     * @param dzero Floating point error considered negligible when comparing "duplicate" nearest
     * neighbours
     * @returns 2x|#neighbour pairs| matrix N of nearest neighbour simplices, such that (N(0,k),
     * N(1,k)) yields the k^{th} nearest neighbour pair (si \in S1, sj \in S2), i.e. the simplices
     * sj in S2 nearest to si.
     */
    PBAT_API GpuIndexMatrixX DetectContactPairsFromOverlaps(
        Points const& P,
        Simplices const& S1,
        Simplices const& S2,
        Bodies const& B1,
        Bodies const& B2,
        Bvh const& bvh,
        GpuScalar dhat  = std::numeric_limits<GpuScalar>::max(),
        GpuScalar dzero = std::numeric_limits<GpuScalar>::epsilon());

    PBAT_API ~BvhQuery();

  private:
    BvhQueryImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_QUERY_CUH