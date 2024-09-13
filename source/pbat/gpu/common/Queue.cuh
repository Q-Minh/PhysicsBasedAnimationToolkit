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
        end        = (end + 1) % kCapacity;
        queue[end] = std::forward<T>(value);
    }
    __host__ __device__ T const& Top() const { return queue[begin]; }
    __host__ __device__ void Pop() const { begin = (begin + 1) % kCapacity; }
    __host__ __device__ bool IsFull() const { return (end + 1) % kCapacity == begin; }
    __host__ __device__ bool IsEmpty() const { return begin == end; }
    __host__ __device__ void Clear()
    {
        begin = 0;
        end   = 0;
    }

  private:
    T queue[kCapacity];
    GpuIndex begin, end;
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_QUEUE_CUH