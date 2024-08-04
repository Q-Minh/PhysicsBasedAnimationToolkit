#ifndef PBAT_GPU_COMMON_VAR_CUH
#define PBAT_GPU_COMMON_VAR_CUH

#include <cstddef>
#include <cuda/api.hpp>
#include <memory>

namespace pbat {
namespace gpu {
namespace common {

template <class T>
class Var
{
  public:
    Var(T const& value = T{},
        std::shared_ptr<cuda::stream_t> stream =
            std::make_shared<cuda::stream_t>(cuda::device::current::get().default_stream()));

    Var(Var const&)            = delete;
    Var& operator=(Var const&) = delete;
    Var& operator=(T const& value);

    Var(Var&&)           = delete;
    Var operator=(Var&&) = delete;

    T Get() const;
    operator T() const;

    T* Raw();
    T const* Raw() const;

    ~Var();

  private:
    cuda::memory::region_t mDeviceMemory;
    std::shared_ptr<cuda::stream_t> mStream;
};

template <class T>
Var<T>::Var(T const& value, std::shared_ptr<cuda::stream_t> stream)
    : mDeviceMemory(), mStream(stream)
{
    mStream->device().make_current();
    mDeviceMemory = cuda::memory::device::async::allocate(*mStream, sizeof(T));
}

template <class T>
Var<T>& Var<T>::operator=(T const& value)
{
    mStream->device().make_current();
    cuda::memory::async::copy(
        mDeviceMemory,
        reinterpret_cast<void*>(const_cast<T*>(&value)),
        *mStream);
    return *this;
}

template <class T>
T Var<T>::Get() const
{
    std::byte memory[sizeof(T)];
    mStream->device().make_current();
    cuda::memory::async::copy(reinterpret_cast<void*>(&memory), mDeviceMemory, *mStream);
    mStream->synchronize();
    return *reinterpret_cast<T*>(&memory);
}

template <class T>
Var<T>::operator T() const
{
    return Get();
}

template <class T>
T* Var<T>::Raw()
{
    return static_cast<T*>(mDeviceMemory.data());
}

template <class T>
T const* Var<T>::Raw() const
{
    return static_cast<T const*>(mDeviceMemory.data());
}

template <class T>
Var<T>::~Var()
{
    mStream->device().make_current();
    cuda::memory::device::free(mDeviceMemory);
}

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_VAR_CUH