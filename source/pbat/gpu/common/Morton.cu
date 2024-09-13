// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Morton.cuh"

namespace pbat {
namespace gpu {
namespace common {

MortonCodeType ExpandBits(MortonCodeType v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

MortonCodeType Morton3D(std::array<GpuScalar, 3> x)
{
    static_assert(
        std::is_same_v<GpuScalar, float>,
        "Morton code only supported for single precision floating point numbers");
    x[0]              = min(max(x[0] * 1024.0f, 0.0f), 1023.0f);
    x[1]              = min(max(x[1] * 1024.0f, 0.0f), 1023.0f);
    x[2]              = min(max(x[2] * 1024.0f, 0.0f), 1023.0f);
    MortonCodeType xx = ExpandBits(static_cast<MortonCodeType>(x[0]));
    MortonCodeType yy = ExpandBits(static_cast<MortonCodeType>(x[1]));
    MortonCodeType zz = ExpandBits(static_cast<MortonCodeType>(x[2]));
    return xx * 4 + yy * 2 + zz;
}

} // namespace common
} // namespace gpu
} // namespace pbat