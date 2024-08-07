// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Buffer.cuh"
#include "pbat/gpu/Aliases.h"

#include <doctest/doctest.h>
#include <span>

TEST_CASE("[gpu][common] Buffer")
{
    using pbat::gpu::common::Buffer;
    using ValueType = pbat::GpuScalar;

    std::vector<ValueType> const bufferExpected{{0.f, 1.f, 2.f, 3.f, 4.f}};
    Buffer<ValueType> gpuBuffer{bufferExpected.size()};
    thrust::copy(bufferExpected.begin(), bufferExpected.end(), gpuBuffer.Data());
    std::vector<ValueType> cpuBuffer = gpuBuffer.Get();
    CHECK_EQ(cpuBuffer, bufferExpected);
}