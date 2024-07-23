#include "SweepAndTiniestQueue.h"

#include "SweepAndTiniestQueueImpl.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndTiniestQueue::SweepAndTiniestQueue() : mImpl(nullptr)
{
    mImpl = new SweepAndTiniestQueueImpl();
}

SweepAndTiniestQueue::~SweepAndTiniestQueue()
{
    if (mImpl != nullptr)
    {
        delete mImpl;
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat