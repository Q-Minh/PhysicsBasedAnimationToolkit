#ifndef PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH
#define PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH

#include <cuda/atomic>
#include <cuda/std/utility>
#include <thrust/device_vector.h>

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

    // SweepAndTiniestQueue(std::size_t nPrimitives, std::size_t nOverlaps);

    // ~SweepAndTiniestQueue();

  private:
    thrust::device_vector<IndexType> inds;       ///< Box indices
    thrust::device_vector<ScalarType> bx;        ///< Boxes begin along x-axis
    thrust::device_vector<ScalarType> by;        ///< Boxes begin along y-axis
    thrust::device_vector<ScalarType> bz;        ///< Boxes begin along z-axis
    thrust::device_vector<ScalarType> ex;        ///< Boxes end along x-axis
    thrust::device_vector<ScalarType> ey;        ///< Boxes end along y-axis
    thrust::device_vector<ScalarType> ez;        ///< Boxes end along z-axis
    thrust::device_ptr<AtomicScalarType> mux;    ///< Box center mean in x-axis
    thrust::device_ptr<AtomicScalarType> muy;    ///< Box center mean in y-axis
    thrust::device_ptr<AtomicScalarType> muz;    ///< Box center mean in z-axis
    thrust::device_ptr<AtomicScalarType> sigmax; ///< Box center variance in x-axis
    thrust::device_ptr<AtomicScalarType> sigmay; ///< Box center variance in y-axis
    thrust::device_ptr<AtomicScalarType> sigmaz; ///< Box center variance in z-axis
    thrust::device_vector<cuda::std::pair<IndexType, IndexType>> o; ///< Overlaps
    thrust::device_ptr<AtomicSizeType> no;                          ///< Number of overlaps
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_CUH