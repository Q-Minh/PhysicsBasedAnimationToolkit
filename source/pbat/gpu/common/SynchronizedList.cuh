#ifndef PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH
#define PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH

#include "pbat/HostDevice.h"
#include "Buffer.cuh"
#include "Var.cuh"
#include "pbat/gpu/Aliases.h"

#include <cuda/atomic>
#include <exception>
#include <string>
#include <vector>

namespace pbat {
namespace gpu {
namespace common {

template <class T>
class DeviceSynchronizedList
{
  public:
    PBAT_HOST DeviceSynchronizedList(Buffer<T>& buffer, Var<GpuIndex>& size)
        : mBuffer(buffer.Raw()), mSize(size.Raw()), mCapacity(buffer.Size())
    {
    }

    PBAT_DEVICE bool Append(T const& value)
    {
        cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> aSize{*mSize};
        GpuIndex k = aSize++;
        if (k >= mCapacity)
        {
            aSize.store(mCapacity);
            return false;
        }
        mBuffer[k] = value;
        return true;
    }

    PBAT_DEVICE T& operator[](auto i)
    {
        assert(i >= 0);
        assert(i < (*mSize));
        return mBuffer[i];
    }

    PBAT_DEVICE T const& operator[](auto i) const
    {
        assert(i >= 0);
        assert(i < (*mSize));
        return mBuffer[i];
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
    void Clear() { mSize = 0; }
    std::size_t Capacity() const { return mBuffer.Size(); }
    Buffer<T>& Memory() { return mBuffer; }
    Buffer<T> const& Memory() const { return mBuffer; }
    GpuIndex Size() const { return mSize.Get(); }
    void Resize(GpuIndex size)
    {
        if ((size < 0) or (size > mBuffer.Size()))
        {
            std::string const what = "Resize called with size outside of range [0,buffer capacity]";
            throw std::invalid_argument(what);
        }
        mSize = size;
    }
    auto Begin() { return mBuffer.Data(); }
    auto End() { return mBuffer.Data() + mSize.Get(); }

  private:
    Buffer<T> mBuffer;
    Var<GpuIndex> mSize;
};

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_SYNCHRONIZED_LIST_CUH