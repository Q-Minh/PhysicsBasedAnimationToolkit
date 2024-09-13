#ifndef PBAT_GPU_COMMON_STACK_CUH
#define PBAT_GPU_COMMON_STACK_CUH

#include <pbat/gpu/Aliases.h>

namespace pbat {
namespace gpu {
namespace common {

template <class T, auto kCapacity = 64>
class Stack
{
  public:
    __host__ __device__ Stack() : stack{}, size{0} {}
    __host__ __device__ void Push(T value) { stack[size++] = value; }
    __host__ __device__ T Pop() { return stack[--size]; }
    __host__ __device__ T const& Top() const { return stack[size - 1]; }
    __host__ __device__ GpuIndex Size() const { return size; }
    __host__ __device__ bool IsEmpty() const { return size == 0; }
    __host__ __device__ bool IsFull() const { return size == kCapacity; }
    __host__ __device__ void Clear() { size = 0; }

  private:
    T stack[kCapacity];
    GpuIndex size; ///< Serves as both the stack pointer and the number of elements in the stack
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_STACK_CUH