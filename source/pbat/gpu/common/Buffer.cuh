#ifndef PBAT_GPU_COMMON_BUFFER_CUH
#define PBAT_GPU_COMMON_BUFFER_CUH

#include <cstddef>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <vector>

namespace pbat {
namespace gpu {
namespace common {

template <class T, int D = 1>
class Buffer
{
  public:
    Buffer(std::size_t count = 0ULL);

    thrust::device_vector<T>& operator[](auto d);
    thrust::device_vector<T> const& operator[](auto d) const;

    std::size_t Size() const;
    bool Empty() const;
    std::vector<T> Get() const;

    std::conditional_t<(D > 1), std::array<thrust::device_ptr<T>, D>, thrust::device_ptr<T>> Data();
    std::conditional_t<
        (D > 1),
        std::array<thrust::device_ptr<T const>, D>,
        thrust::device_ptr<T const>>
    Data() const;

    std::conditional_t<(D > 1), std::array<T*, D>, T*> Raw();
    std::conditional_t<(D > 1), std::array<T const*, D>, T const*> Raw() const;

    constexpr int Dimensions() const { return D; }

  private:
    std::array<thrust::device_vector<T>, D> mBuffers;
};

template <class T, int D>
Buffer<T, D>::Buffer(std::size_t count) : mBuffers()
{
    for (auto d = 0; d < D; ++d)
        mBuffers[d].resize(count);
}

template <class T, int D>
thrust::device_vector<T>& Buffer<T, D>::operator[](auto d)
{
    return mBuffers[d];
}

template <class T, int D>
thrust::device_vector<T> const& Buffer<T, D>::operator[](auto d) const
{
    return mBuffers[d];
}

template <class T, int D>
std::size_t Buffer<T, D>::Size() const
{
    return mBuffers[0].size();
}

template <class T, int D>
bool Buffer<T, D>::Empty() const
{
    return Size() == 0ULL;
}

template <class T, int D>
std::vector<T> Buffer<T, D>::Get() const
{
    std::vector<T> buffer(Size() * D);
    for (auto d = 0; d < D; ++d)
    {
        thrust::copy(mBuffers[d].begin(), mBuffers[d].end(), buffer.begin() + d * Size());
    }
    return buffer;
}

template <class T, int D>
std::conditional_t<(D > 1), std::array<thrust::device_ptr<T>, D>, thrust::device_ptr<T>>
Buffer<T, D>::Data()
{
    std::array<thrust::device_ptr<T>, D> data{};
    for (auto d = 0; d < D; ++d)
        data[d] = mBuffers[d].data();
    if constexpr (D > 1)
    {
        return data;
    }
    else
    {
        return data[0];
    }
}

template <class T, int D>
std::conditional_t<(D > 1), std::array<thrust::device_ptr<T const>, D>, thrust::device_ptr<T const>>
Buffer<T, D>::Data() const
{
    std::array<thrust::device_ptr<T const>, D> data{};
    for (auto d = 0; d < D; ++d)
        data[d] = mBuffers[d].data();
    if constexpr (D > 1)
    {
        return data;
    }
    else
    {
        return data[0];
    }
}

template <class T, int D>
std::conditional_t<(D > 1), std::array<T*, D>, T*> Buffer<T, D>::Raw()
{
    std::array<T*, D> raw{};
    for (auto d = 0; d < D; ++d)
        raw[d] = thrust::raw_pointer_cast(mBuffers[d].data());
    if constexpr (D > 1)
    {
        return raw;
    }
    else
    {
        return raw[0];
    }
}

template <class T, int D>
std::conditional_t<(D > 1), std::array<T const*, D>, T const*> Buffer<T, D>::Raw() const
{
    std::array<T const*, D> raw{};
    for (auto d = 0; d < D; ++d)
        raw[d] = thrust::raw_pointer_cast(mBuffers[d].data());
    if constexpr (D > 1)
    {
        return raw;
    }
    else
    {
        return raw[0];
    }
}

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_BUFFER_CUH