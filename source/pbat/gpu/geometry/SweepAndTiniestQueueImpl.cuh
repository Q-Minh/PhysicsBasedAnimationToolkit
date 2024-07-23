#ifndef PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_IMPL_H
#define PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_IMPL_H

#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndTiniestQueueImpl
{
  public:
    using ScalarType = float;

  private:
    thrust::device_vector<ScalarType> b; ///< Interval begin
    thrust::device_vector<ScalarType> e; ///< Interval end
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_IMPL_H