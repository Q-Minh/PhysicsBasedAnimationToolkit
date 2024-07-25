#ifndef PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH
#define PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH

#include "Primitives.cuh"
#include "pbat/gpu/Aliases.h"

#include <cuda/atomic>
#include <cuda/std/utility>
#include <thrust/device_ptr.h>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndTiniestQueue
{
  public:
    using AtomicSizeType   = cuda::atomic<GpuIndex, cuda::thread_scope_device>;
    using AtomicScalarType = cuda::atomic<GpuScalar, cuda::thread_scope_device>;
    using OverlapType      = cuda::std::pair<GpuIndex, GpuIndex>;

    /**
     * @brief Construct a new Sweep And Tiniest Queue object
     *
     * @param nPrimitives
     * @param nOverlaps
     */
    SweepAndTiniestQueue(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief Compute self-overlapping non-adjacent simplices in S
     * @param P
     * @param S
     */
    void SortAndSweep(Points const& P, Simplices const& S);

    ~SweepAndTiniestQueue();

  private:
    thrust::device_ptr<GpuIndex> binds;                           ///< Box indices
    thrust::device_ptr<GpuScalar> bx, by, bz;                    ///< Box beginnings
    thrust::device_ptr<GpuScalar> ex, ey, ez;                    ///< Box ends
    thrust::device_ptr<AtomicScalarType> mux, muy, muz;          ///< Box center mean
    thrust::device_ptr<AtomicScalarType> sigmax, sigmay, sigmaz; ///< Box center variance
    thrust::device_ptr<GpuIndex> saxis;                          ///< Sort axis
    thrust::device_ptr<AtomicSizeType> no;                       ///< Number of overlaps
    thrust::device_ptr<OverlapType> o;                           ///< Overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH