#ifndef PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH
#define PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH

#include "Primitives.cuh"
#include "pbat/gpu/Aliases.h"

#include <array>
#include <cuda/std/utility>
#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndPruneImpl
{
  public:
    using OverlapType = cuda::std::pair<GpuIndex, GpuIndex>;

    /**
     * @brief Construct a new Sweep And Tiniest Queue object
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
    void SortAndSweep(Points const& P, Simplices const& S1, Simplices const& S2);

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

  private:
    thrust::device_vector<GpuIndex> binds;                ///< Box indices
    std::array<thrust::device_vector<GpuScalar>, 3> b, e; ///< Box begin/end
    thrust::device_vector<GpuScalar> mu, sigma;           ///< Box center mean and variance
    thrust::device_vector<GpuIndex> no;                   ///< Number of overlaps
    thrust::device_vector<OverlapType> o;                 ///< Overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_PRUNE_IMPL_CUH