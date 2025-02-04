// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Buffer.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"

#include <exception>
#include <iostream>
#include <string>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace common {

Buffer::Buffer(GpuIndex dims, GpuIndex n, EType type) : mDims(-1), mType(), mImpl(nullptr)
{
    switch (type)
    {
        case EType::uint8: mType = typeid(std::uint8_t).name(); break;
        case EType::uint16: mType = typeid(std::uint16_t).name(); break;
        case EType::uint32: mType = typeid(std::uint32_t).name(); break;
        case EType::uint64: mType = typeid(std::uint64_t).name(); break;
        case EType::int8: mType = typeid(std::int8_t).name(); break;
        case EType::int16: mType = typeid(std::int16_t).name(); break;
        case EType::int32: mType = typeid(std::int32_t).name(); break;
        case EType::int64: mType = typeid(std::int64_t).name(); break;
        case EType::float32: mType = typeid(float).name(); break;
        case EType::float64: mType = typeid(double).name(); break;
        default: throw std::invalid_argument("Unknown type.");
    }
    Resize(dims, n);
}

Buffer::Buffer(Buffer&& other) noexcept : mDims(other.mDims), mType(other.mType), mImpl(other.mImpl)
{
    other.mDims = -1;
    other.mImpl = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other)
    {
        mDims       = other.mDims;
        mType       = other.mType;
        mImpl       = other.mImpl;
        other.mDims = -1;
        other.mImpl = nullptr;
    }
    return *this;
}

#define DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(T)                                               \
    Buffer::Buffer(Data<T> const& data) : mDims(-1), mType(typeid(T).name()), mImpl(nullptr) \
    {                                                                                        \
        *this = data;                                                                        \
    }

DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::uint8_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::uint16_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::uint32_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::uint64_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::int8_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::int16_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::int32_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(std::int64_t);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(float);
DEFINE_BUFFER_CONSTRUCTOR_FROM_DATA(double);

#define DEFINE_BUFFER_ASSIGNMENT_OPERATOR(T)                                              \
    Buffer& Buffer::operator=(Data<T> const& data)                                        \
    {                                                                                     \
        if (mType != typeid(T).name())                                                    \
        {                                                                                 \
            throw std::invalid_argument("Input data does not match this Buffer's type."); \
        }                                                                                 \
        Resize(static_cast<GpuIndex>(data.rows()), static_cast<GpuIndex>(data.cols()));   \
        pbat::common::ForRange<1, kMaxDims + 1>([&]<auto kDims>() {                       \
            if (mDims == kDims)                                                           \
            {                                                                             \
                impl::common::ToBuffer(                                                   \
                    data,                                                                 \
                    *static_cast<impl::common::Buffer<T, kDims>*>(mImpl));                \
            }                                                                             \
        });                                                                               \
        return *this;                                                                     \
    }

DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::uint8_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::uint16_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::uint32_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::uint64_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::int8_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::int16_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::int32_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(std::int64_t);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(float);
DEFINE_BUFFER_ASSIGNMENT_OPERATOR(double);

Buffer::EType Buffer::Type() const
{
    if (mType == typeid(std::uint8_t).name())
        return EType::uint8;
    if (mType == typeid(std::uint16_t).name())
        return EType::uint16;
    if (mType == typeid(std::uint32_t).name())
        return EType::uint32;
    if (mType == typeid(std::uint64_t).name())
        return EType::uint64;
    if (mType == typeid(std::int8_t).name())
        return EType::int8;
    if (mType == typeid(std::int16_t).name())
        return EType::int16;
    if (mType == typeid(std::int32_t).name())
        return EType::int32;
    if (mType == typeid(std::int64_t).name())
        return EType::int64;
    if (mType == typeid(float).name())
        return EType::float32;
    if (mType == typeid(double).name())
        return EType::float64;
    throw std::invalid_argument("Unknown type.");
}

std::size_t Buffer::Size() const
{
    std::size_t size{0};
    pbat::common::ForRange<1, kMaxDims + 1>([&]<auto kDims>() {
        if (mDims == kDims)
        {
            pbat::common::ForTypes<
                std::int8_t,
                std::int16_t,
                std::int32_t,
                std::int64_t,
                std::uint8_t,
                std::uint16_t,
                std::uint32_t,
                std::uint64_t,
                float,
                double>([&]<class T>() {
                if (mType == typeid(T).name())
                {
                    size = static_cast<impl::common::Buffer<T, kDims>*>(mImpl)->Size();
                }
            });
        }
    });
    return size;
}

void Buffer::Resize(GpuIndex n)
{
    Resize(mDims, n);
}

void Buffer::Resize(GpuIndex dims, GpuIndex n)
{
    if (dims < 1 || dims > kMaxDims)
    {
        throw std::invalid_argument("Expected 1 <= dims <= kMaxDims.");
    }
    bool const bShouldAllocate = dims != mDims;
    if (bShouldAllocate)
        Deallocate();
    pbat::common::ForRange<1, kMaxDims + 1>([&]<auto kDims>() {
        if (dims == kDims)
        {
            if (bShouldAllocate)
            {
                mDims = dims;
                pbat::common::ForTypes<
                    std::int8_t,
                    std::int16_t,
                    std::int32_t,
                    std::int64_t,
                    std::uint8_t,
                    std::uint16_t,
                    std::uint32_t,
                    std::uint64_t,
                    float,
                    double>([&]<class T>() {
                    if (mType == typeid(T).name())
                    {
                        mImpl = new impl::common::Buffer<T, kDims>(n);
                    }
                });
            }
            else
            {
                pbat::common::ForTypes<
                    std::int8_t,
                    std::int16_t,
                    std::int32_t,
                    std::int64_t,
                    std::uint8_t,
                    std::uint16_t,
                    std::uint32_t,
                    std::uint64_t,
                    float,
                    double>([&]<class T>() {
                    if (mType == typeid(T).name())
                    {
                        auto* impl = static_cast<impl::common::Buffer<T, kDims>*>(mImpl);
                        impl->Resize(n);
                    }
                });
            }
        }
    });
}

void* Buffer::Impl()
{
    return mImpl;
}

void const* Buffer::Impl() const
{
    return mImpl;
}

Buffer::~Buffer()
{
    Deallocate();
}

void Buffer::Deallocate()
{
    if (mImpl != nullptr)
    {
        pbat::common::ForRange<1, kMaxDims + 1>([&]<auto kDims>() {
            if (mDims == kDims)
            {
                pbat::common::ForTypes<
                    std::int8_t,
                    std::int16_t,
                    std::int32_t,
                    std::int64_t,
                    std::uint8_t,
                    std::uint16_t,
                    std::uint32_t,
                    std::uint64_t,
                    float,
                    double>([&]<class T>() {
                    if (mType == typeid(T).name())
                    {
                        delete static_cast<impl::common::Buffer<T, kDims>*>(mImpl);
                    }
                });
            }
        });
        mImpl = nullptr;
    }
}

} // namespace common
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[gpu][common] Buffer")
{
    using namespace pbat;
    using namespace pbat::gpu;
    using namespace pbat::gpu::common;

    SUBCASE("Buffer")
    {
        Buffer buf(1, 10, Buffer::EType::float32);
        CHECK(buf.Dims() == 1);
        CHECK(buf.Type() == Buffer::EType::float32);
        CHECK(buf.Size() == 10);
    }
    SUBCASE("Buffer constructor")
    {
        GpuMatrixX A = GpuMatrixX::Random(2, 10);
        Buffer buf(A);
        CHECK(buf.Dims() == 2);
        CHECK(buf.Type() == Buffer::EType::float32);
        CHECK(buf.Size() == 10);
        auto* impl    = static_cast<impl::common::Buffer<float, 2>*>(buf.Impl());
        GpuMatrixX AG = impl::common::ToEigen(*impl);
        CHECK(AG.isApprox(A));
    }
    SUBCASE("Buffer assignment")
    {
        Buffer buf(2, 10, Buffer::EType::float32);
        GpuMatrixX A = GpuMatrixX::Random(2, 10);
        buf          = A;
        CHECK(buf.Dims() == 2);
        CHECK(buf.Type() == Buffer::EType::float32);
        CHECK(buf.Size() == 10);
        auto* impl    = static_cast<impl::common::Buffer<float, 2>*>(buf.Impl());
        GpuMatrixX AG = impl::common::ToEigen(*impl);
        CHECK(AG.isApprox(A));
    }
    SUBCASE("Buffer resize")
    {
        Buffer buf(1, 10, Buffer::EType::float32);
        buf.Resize(20);
        CHECK(buf.Dims() == 1);
        CHECK(buf.Type() == Buffer::EType::float32);
        CHECK(buf.Size() == 20);
    }
    SUBCASE("Buffer resize with different dims")
    {
        Buffer buf(1, 10, Buffer::EType::float32);
        buf.Resize(2, 20);
        CHECK(buf.Dims() == 2);
        CHECK(buf.Type() == Buffer::EType::float32);
        CHECK(buf.Size() == 20);
    }
}