#ifndef PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_H
#define PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_H

#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat {
namespace gpu {
namespace geometry {

class SweepAndTiniestQueueImpl;

class SweepAndTiniestQueue
{
  public:
    PBAT_API SweepAndTiniestQueue();

    SweepAndTiniestQueue(SweepAndTiniestQueue const&)            = delete;
    SweepAndTiniestQueue& operator=(SweepAndTiniestQueue const&) = delete;

    PBAT_API ~SweepAndTiniestQueue();

  private:
    SweepAndTiniestQueueImpl* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_SWEEP_AND_TINIEST_QUEUE_H