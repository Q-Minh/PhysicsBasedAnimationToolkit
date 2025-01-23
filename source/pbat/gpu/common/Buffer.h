#ifndef PBAT_GPU_COMMON_BUFFER_H
#define PBAT_GPU_COMMON_BUFFER_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

#include <cstddef>

namespace pbat::gpu::common {

class Buffer
{
  public:
    enum class EType { uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64 };
    static auto constexpr kMaxDims = 3;

    template <class T>
    using Data = Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> const>;

    PBAT_API Buffer(GpuIndex dims, GpuIndex n, EType type);
    Buffer(const Buffer& other)            = delete;
    Buffer& operator=(const Buffer& other) = delete;
    PBAT_API Buffer(Buffer&& other) noexcept;
    PBAT_API Buffer& operator=(Buffer&& other) noexcept;

    PBAT_API Buffer& operator=(Data<std::uint8_t> const& data);
    PBAT_API Buffer& operator=(Data<std::uint16_t> const& data);
    PBAT_API Buffer& operator=(Data<std::uint32_t> const& data);
    PBAT_API Buffer& operator=(Data<std::uint64_t> const& data);
    PBAT_API Buffer& operator=(Data<std::int8_t> const& data);
    PBAT_API Buffer& operator=(Data<std::int16_t> const& data);
    PBAT_API Buffer& operator=(Data<std::int32_t> const& data);
    PBAT_API Buffer& operator=(Data<std::int64_t> const& data);
    PBAT_API Buffer& operator=(Data<float> const& data);
    PBAT_API Buffer& operator=(Data<double> const& data);

    [[maybe_unused]] GpuIndex Dims() const { return mDims; }
    PBAT_API EType Type() const;
    PBAT_API std::size_t Size() const;

    PBAT_API void Resize(GpuIndex n);
    PBAT_API void Resize(GpuIndex dims, GpuIndex n);

    PBAT_API void* Impl();
    PBAT_API void const* Impl() const;

    PBAT_API ~Buffer();

  private:
    void Deallocate();

    GpuIndex mDims;
    std::string mType;
    void* mImpl; ///< impl::common::Buffer<ValueType, Dims>*
};

} // namespace pbat::gpu::common

#endif // PBAT_GPU_COMMON_BUFFER_H
