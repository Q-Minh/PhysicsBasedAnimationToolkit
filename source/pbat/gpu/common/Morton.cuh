#ifndef PBAT_GPU_COMMON_MORTON_CUH
#define PBAT_GPU_COMMON_MORTON_CUH

#include <cstddef>
#include <cuda/std/cmath>
#include <pbat/gpu/Aliases.h>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace common {

using MortonCodeType = cuda::std::uint32_t;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ MortonCodeType ExpandBits(MortonCodeType v);

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ MortonCodeType Morton3D(std::array<GpuScalar, 3> x);

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_MORTON_CUH