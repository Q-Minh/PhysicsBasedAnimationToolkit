#ifndef PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH
#define PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH

#include "AxisAlignedBoundingBoxes.cuh"

#include <cuda/atomic>
#include <cuda/std/utility>
#include <thrust/device_ptr.h>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndTiniestQueue
{
  public:
    using ScalarType       = float;
    using IndexType        = int;
    using AtomicSizeType   = cuda::atomic<int, cuda::thread_scope_device>;
    using AtomicScalarType = cuda::atomic<ScalarType, cuda::thread_scope_device>;
    using OverlapType      = cuda::std::pair<IndexType, IndexType>;

    /**
     * @brief Construct a new Sweep And Tiniest Queue object
     *
     * @param nPrimitives
     * @param nOverlaps
     */
    SweepAndTiniestQueue(std::size_t nPrimitives, std::size_t nOverlaps);

    /**
     * @brief
     *
     * @param lb
     * @param rb
     */
    // void SortAndSweep(AxisAlignedBoundingBoxes const& lb, AxisAlignedBoundingBoxes const& rb);

    ~SweepAndTiniestQueue();

  private:
    thrust::device_ptr<IndexType> inds;                          ///< Box indices
    thrust::device_ptr<ScalarType> bx, by, bz;                   ///< Box beginnings
    thrust::device_ptr<ScalarType> ex, ey, ez;                   ///< Box ends
    thrust::device_ptr<AtomicScalarType> mux, muy, muz;          ///< Box center mean
    thrust::device_ptr<AtomicScalarType> sigmax, sigmay, sigmaz; ///< Box center variance
    thrust::device_ptr<AtomicSizeType> no;                       ///< Number of overlaps
    thrust::device_ptr<OverlapType> o;                           ///< Overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH