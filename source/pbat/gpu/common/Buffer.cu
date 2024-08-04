#include "Buffer.cuh"
#include "pbat/gpu/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[gpu][common] Buffer")
{
    using pbat::gpu::common::Buffer;
    using ValueType = pbat::GpuScalar;

    std::size_t constexpr kBufferPadding = 2ULL;
    std::vector<ValueType> const bufferExpected{{0.f, 1.f, 2.f, 3.f, 4.f}};
    Buffer<ValueType> gpuBuffer{bufferExpected.size() + kBufferPadding};
    gpuBuffer = std::span<ValueType const>(bufferExpected.data(), bufferExpected.size());
    std::vector<ValueType> cpuBuffer = gpuBuffer.View(0ULL, bufferExpected.size()).Get();
    CHECK_EQ(cpuBuffer, bufferExpected);
}