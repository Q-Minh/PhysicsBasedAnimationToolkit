#ifndef PBAT_GPU_COMMON_BUFFER_CUH
#define PBAT_GPU_COMMON_BUFFER_CUH

#include <concepts>
#include <cstddef>
#include <cuda/api.hpp>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pbat {
namespace gpu {
namespace common {

template <class T>
class Buffer
{
  public:
    static_assert(std::is_trivial_v<T>, "Buffer only supports trivial type template argument");

    Buffer(
        std::size_t count = 0ULL,
        std::shared_ptr<cuda::stream_t> stream =
            std::make_shared<cuda::stream_t>(cuda::device::current::get().default_stream()));

    Buffer(Buffer const&) noexcept            = delete;
    Buffer& operator=(Buffer const&) noexcept = delete;

    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    Buffer& operator=(std::span<T> value);
    Buffer& operator=(std::span<T const> value);

    std::size_t Size() const;
    bool Empty() const;
    std::vector<T> Get() const;

    Buffer<T> View(std::size_t start = 0ULL);
    Buffer<T> const View(std::size_t start = 0ULL) const;

    Buffer<T> View(std::size_t start, std::size_t end);
    Buffer<T> const View(std::size_t start, std::size_t end) const;

    T* Raw();
    T const* Raw() const;

    ~Buffer();

  private:
    Buffer(cuda::memory::region_t subregion, std::shared_ptr<cuda::stream_t> stream);

    cuda::memory::region_t mDeviceMemory;
    std::shared_ptr<cuda::stream_t> mStream;
    bool mbOwnsMemory;
};

template <class T>
Buffer<T>::Buffer(std::size_t count, std::shared_ptr<cuda::stream_t> stream)
    : mDeviceMemory(), mStream(stream), mbOwnsMemory(true)
{
    if (count > 0ULL)
    {
        mStream->device().make_current();
        mDeviceMemory = cuda::memory::device::async::allocate(*mStream, count * sizeof(T));
        cuda::memory::device::async::zero(Raw(), *mStream);
    }
}

template <class T>
Buffer<T>::Buffer(Buffer&& other) noexcept
    : mDeviceMemory(other.mDeviceMemory), mStream(other.mStream), mbOwnsMemory(other.mbOwnsMemory)
{
    other.mbOwnsMemory = false;
}

template <class T>
Buffer<T>& Buffer<T>::operator=(Buffer&& other) noexcept
{
    mDeviceMemory      = other.mDeviceMemory;
    mStream            = other.mStream;
    mbOwnsMemory       = other.mbOwnsMemory;
    other.mbOwnsMemory = false;
    return *this;
}

template <class T>
Buffer<T>& Buffer<T>::operator=(std::span<T> value)
{
    return (*this) = std::span<T const>(value);
}

template <class T>
Buffer<T>& Buffer<T>::operator=(std::span<T const> value)
{
    if (value.size_bytes() > mDeviceMemory.size())
    {
        std::string const what = "Expected argument of size " +
                                 std::to_string(mDeviceMemory.size()) + " bytes, but got " +
                                 std::to_string(value.size_bytes()) + " instead";
        throw std::invalid_argument(what);
    }
    auto subregion = mDeviceMemory.subregion(0ULL, value.size_bytes());
    mStream->device().make_current();
    cuda::memory::async::copy(
        subregion,
        reinterpret_cast<void*>(const_cast<T*>(value.data())),
        *mStream);
    return *this;
}

template <class T>
std::size_t Buffer<T>::Size() const
{
    return mDeviceMemory.size() / sizeof(T);
}

template <class T>
bool Buffer<T>::Empty() const
{
    return Size() == 0ULL;
}

template <class T>
std::vector<T> Buffer<T>::Get() const
{
    std::vector<T> memory(Size());
    mStream->device().make_current();
    cuda::memory::async::copy(reinterpret_cast<void*>(memory.data()), mDeviceMemory, *mStream);
    mStream->synchronize();
    return memory;
}

template <class T>
Buffer<T> Buffer<T>::View(std::size_t start)
{
    return View(start, Size());
}

template <class T>
Buffer<T> const Buffer<T>::View(std::size_t start) const
{
    return View(start, Size());
}

template <class T>
Buffer<T> Buffer<T>::View(std::size_t start, std::size_t end)
{
    auto subregion = mDeviceMemory.subregion(start * sizeof(T), end * sizeof(T));
    return Buffer<T>(subregion, mStream);
}

template <class T>
Buffer<T> const Buffer<T>::View(std::size_t start, std::size_t end) const
{
    auto subregion = mDeviceMemory.subregion(start * sizeof(T), end * sizeof(T));
    return Buffer<T>(subregion, mStream);
}

template <class T>
T* Buffer<T>::Raw()
{
    return static_cast<T*>(mDeviceMemory.data());
}

template <class T>
T const* Buffer<T>::Raw() const
{
    return static_cast<T const*>(mDeviceMemory.data());
}

template <class T>
Buffer<T>::~Buffer()
{
    if (mbOwnsMemory and !Empty())
    {
        mStream->device().make_current();
        cuda::memory::device::free(mDeviceMemory);
    }
}

template <class T>
Buffer<T>::Buffer(cuda::memory::region_t subregion, std::shared_ptr<cuda::stream_t> stream)
    : mDeviceMemory(subregion), mStream(stream), mbOwnsMemory(false)
{
}

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_BUFFER_CUH