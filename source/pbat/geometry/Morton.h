#ifndef PBAT_GEOMETRY_MORTON_H
#define PBAT_GEOMETRY_MORTON_H

#include "pbat/HostDevice.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace pbat {
namespace geometry {

using MortonCodeType = std::uint32_t;

// NOTE:
// We make these (otherwise non-templated) functions inline so that they
// are compiled by nvcc whenever this header is included in cuda sources.

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
PBAT_HOST_DEVICE inline MortonCodeType ExpandBits(MortonCodeType v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
[[maybe_unused]] PBAT_HOST_DEVICE inline MortonCodeType Morton3D(std::array<float, 3> x)
{
    using namespace std;
    x[0]              = min(max(x[0] * 1024.0f, 0.0f), 1023.0f);
    x[1]              = min(max(x[1] * 1024.0f, 0.0f), 1023.0f);
    x[2]              = min(max(x[2] * 1024.0f, 0.0f), 1023.0f);
    MortonCodeType xx = ExpandBits(static_cast<MortonCodeType>(x[0]));
    MortonCodeType yy = ExpandBits(static_cast<MortonCodeType>(x[1]));
    MortonCodeType zz = ExpandBits(static_cast<MortonCodeType>(x[2]));
    return xx * 4 + yy * 2 + zz;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MORTON_H