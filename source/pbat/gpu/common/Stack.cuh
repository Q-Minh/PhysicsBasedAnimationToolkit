#ifndef PBAT_GPU_COMMON_STACK_CUH
#define PBAT_GPU_COMMON_STACK_CUH

#include "pbat/HostDevice.h"
#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace common {

template <class T, auto kCapacity = 64>
class Stack
{
  public:
    PBAT_HOST_DEVICE Stack() : stack{}, size{0} {}
    PBAT_HOST_DEVICE void Push(T value) { stack[size++] = value; }
    PBAT_HOST_DEVICE T Pop() { return stack[--size]; }
    PBAT_HOST_DEVICE T const& Top() const { return stack[size - 1]; }
    PBAT_HOST_DEVICE GpuIndex Size() const { return size; }
    PBAT_HOST_DEVICE bool IsEmpty() const { return size == 0; }
    PBAT_HOST_DEVICE bool IsFull() const { return size == kCapacity; }
    PBAT_HOST_DEVICE void Clear() { size = 0; }

  private:
    T stack[kCapacity];
    GpuIndex size; ///< Serves as both the stack pointer and the number of elements in the stack
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_STACK_CUH