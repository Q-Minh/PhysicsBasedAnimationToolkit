#ifndef PBAT_GPU_COMMON_QUEUE_CUH
#define PBAT_GPU_COMMON_QUEUE_CUH

#include "pbat/HostDevice.h"
#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace common {

template <class T, auto kCapacity = 64>
class Queue
{
  public:
    PBAT_HOST_DEVICE Queue() : queue{}, begin{0}, end{0} {}
    PBAT_HOST_DEVICE void Push(T value)
    {
        queue[end] = value;
        end        = (end + 1) % kCapacity;
        ++n;
    }
    PBAT_HOST_DEVICE T const& Top() const { return queue[begin]; }
    PBAT_HOST_DEVICE void Pop()
    {
        begin = (begin + 1) % kCapacity;
        --n;
    }
    PBAT_HOST_DEVICE bool IsFull() const { return n == kCapacity; }
    PBAT_HOST_DEVICE bool IsEmpty() const { return n == 0; }
    PBAT_HOST_DEVICE GpuIndex Size() const { return n; }
    PBAT_HOST_DEVICE void Clear() { begin = end = n = 0; }

  private:
    T queue[kCapacity];
    GpuIndex begin, end, n;
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_QUEUE_CUH