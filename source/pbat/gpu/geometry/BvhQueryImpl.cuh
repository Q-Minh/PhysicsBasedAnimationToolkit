#ifndef PBAT_GPU_BVH_QUERY_IMPL_CUH
#define PBAT_GPU_BVH_QUERY_IMPL_CUH

#include "BvhImpl.cuh"
#include "PrimitivesImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <cuda/std/cmath>
#include <limits>

namespace pbat {
namespace gpu {
namespace geometry {

/**
 * @brief Query object reporting geometric test results between simplex sets
 */
class BvhQueryImpl
{
  public:
    using OverlapType    = typename BvhImpl::OverlapType;
    using MortonCodeType = typename BvhImpl::MortonCodeType;

    /**
     * @brief
     * @param nPrimitives
     * @param nOverlaps
     */
    BvhQueryImpl(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief
     * @param P
     * @param S
     * @param expansion
     */
    void Build(
        PointsImpl const& P,
        SimplicesImpl const& S,
        GpuScalar expansion = std::numeric_limits<GpuScalar>::min());

    /**
     * @brief
     * @param P
     * @param S1
     * @param S2
     * @param bvh
     */
    void DetectOverlaps(
        PointsImpl const& P,
        SimplicesImpl const& S1,
        SimplicesImpl const& S2,
        BvhImpl const& bvh);

    /**
     * @brief
     * @return
     */
    std::size_t NumberOfAllocatedBoxes() const;

  private:
    common::Buffer<GpuIndex> simplex;      ///< Box/Simplex indices
    common::Buffer<MortonCodeType> morton; ///< Morton codes of simplices
    common::Buffer<GpuScalar, 3> b,
        e; ///< Simplex and internal node bounding boxes.

  public:
    common::SynchronizedList<OverlapType> overlaps; ///< Detected overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_BVH_IMPL_CUH