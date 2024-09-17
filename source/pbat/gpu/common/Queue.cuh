#ifndef PBAT_GPU_COMMON_QUEUE_CUH
#define PBAT_GPU_COMMON_QUEUE_CUH

#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace common {

template <class T, auto kCapacity = 64>
class Queue
{
  public:
    __host__ __device__ Queue() : queue{}, begin{0}, end{0} {}
    __host__ __device__ void Push(T value)
    {
        queue[end] = value;
        end        = (end + 1) % kCapacity;
        ++n;
    }
    __host__ __device__ T const& Top() const { return queue[begin]; }
    __host__ __device__ void Pop()
    {
        begin = (begin + 1) % kCapacity;
        --n;
    }
    __host__ __device__ bool IsFull() const { return n == kCapacity; }
    __host__ __device__ bool IsEmpty() const { return n == 0; }
    __host__ __device__ GpuIndex Size() const { return n; }
    __host__ __device__ void Clear() { begin = end = n = 0; }

  private:
    T queue[kCapacity];
    GpuIndex begin, end, n;
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_QUEUE_CUH