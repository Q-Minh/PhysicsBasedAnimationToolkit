#ifndef PBAT_GPU_GEOMETRY_IMPL_SWEEPANDPRUNE_H
#define PBAT_GPU_GEOMETRY_IMPL_SWEEPANDPRUNE_H

#include "Primitives.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/common/Var.cuh"

#include <cuda/std/utility>
#include <limits>
#include <vector>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {

class SweepAndPrune
{
  public:
    using OverlapType = cuda::std::pair<GpuIndex, GpuIndex>;

    /**
     * @brief Construct a new Sweep And Prune object
     *
     * @param nPrimitives
     * @param nOverlaps
     */
    SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief Compute overlapping, topologically non-adjacent simplices between S1 and S2
     * @param P
     * @param S1
     * @param S2
     */
    void SortAndSweep(
        Points const& P,
        Simplices const& S1,
        Simplices const& S2,
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
    std::vector<OverlapType> Overlaps() const;

  private:
    common::Buffer<GpuIndex> binds;      ///< Box indices
    common::Buffer<GpuIndex, 4> sinds;   ///< Simplex vertex indices
    common::Buffer<GpuScalar, 3> b, e;   ///< Box begin/end
    common::Buffer<GpuScalar> mu, sigma; ///< Box center mean and variance

  public:
    common::SynchronizedList<OverlapType> overlaps; ///< Simplex box overlaps
};

} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_SWEEPANDPRUNE_H
