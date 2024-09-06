#ifndef PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH
#define PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH

#include "Buffer.cuh"
#include "Var.cuh"
#include "pbat/gpu/Aliases.h"

#include <cuda/atomic>
#include <vector>

namespace pbat {
namespace gpu {
namespace common {

template <class T>
class DeviceSynchronizedList
{
  public:
    __host__ DeviceSynchronizedList(Buffer<T>& buffer, Var<GpuIndex>& size)
        : mBuffer(buffer.Raw()), mSize(size.Raw()), mCapacity(buffer.Size())
    {
    }

    __device__ bool Append(T&& value)
    {
        cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> aSize{*mSize};
        GpuIndex k = aSize++;
        if (k >= mCapacity)
        {
            aSize.store(mCapacity);
            return false;
        }
        mBuffer[k] = std::forward<T>(value);
        return true;
    }

  private:
    T* mBuffer;
    GpuIndex* mSize;
    std::size_t mCapacity;
};

template <class T>
class SynchronizedList
{
  public:
    SynchronizedList(std::size_t capacity) : mBuffer(capacity), mSize{0} {}
    DeviceSynchronizedList<T> Raw() { return DeviceSynchronizedList<T>(mBuffer, mSize); }
    std::vector<T> Get() const { return mBuffer.Get(mSize.Get()); }

  private:
    Buffer<T> mBuffer;
    Var<GpuIndex> mSize;
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH