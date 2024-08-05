#ifndef PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH
#define PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH

#include "PrimitivesImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <cuda/std/utility>
#include <limits>
#include <thrust/host_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndPruneImpl
{
  public:
    using OverlapType = cuda::std::pair<GpuIndex, GpuIndex>;

    /**
     * @brief Construct a new Sweep And Prune object
     *
     * @param nPrimitives
     * @param nOverlaps
     */
    SweepAndPruneImpl(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief Compute overlapping, topologically non-adjacent simplices between S1 and S2
     * @param P
     * @param S1
     * @param S2
     */
    void SortAndSweep(
        PointsImpl const& P,
        SimplicesImpl const& S1,
        SimplicesImpl const& S2,
        GpuScalar expansion = std::numeric_limits<GpuScalar>::epsilon());

    /**
     * @brief Obtains the maximum number of simplices that can be tested for overlap.
     * @return
     */
    std::size_t NumberOfAllocatedBoxes() const;
    /**
     * @brief Obtains the maximum number of overlaps that can be detected.
     * @return
     */
    std::size_t NumberOfAllocatedOverlaps() const;
    /**
     * @brief Obtains the CPU copy of detected overlaps in the last call to SortAndSweep
     * @return
     */
    thrust::host_vector<OverlapType> Overlaps() const;

  private:
    common::Buffer<GpuIndex> binds;      ///< Box indices
    common::Buffer<GpuIndex, 4> sinds;   ///< Simplex vertex indices
    common::Buffer<GpuScalar, 3> b, e;   ///< Box begin/end
    common::Buffer<GpuScalar> mu, sigma; ///< Box center mean and variance

  public:
    common::Var<GpuIndex> no;      ///< Number of overlaps
    common::Buffer<OverlapType> o; ///< Overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH