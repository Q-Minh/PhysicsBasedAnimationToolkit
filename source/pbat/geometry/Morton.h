/**
 * @file Morton.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to compute Morton codes.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_MORTON_H
#define PBAT_GEOMETRY_MORTON_H

#include "pbat/HostDevice.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

namespace pbat {
namespace geometry {

using MortonCodeType = std::uint32_t; ///< Type used to represent Morton codes

/**
 * @brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
 *
 * @note We make this (otherwise non-templated) function inline so that it gets compiled by nvcc
 * whenever this header is included in cuda sources.
 *
 * @param v 10-bit integer
 * @return Expanded 30-bit integer
 */
PBAT_HOST_DEVICE inline MortonCodeType ExpandBits(MortonCodeType v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

namespace detail {

template <class Point>
concept CMorton3dPoint = requires(Point p)
{
    {
        p[0]
    } -> std::convertible_to<float>;
    {
        p[1]
    } -> std::convertible_to<float>;
    {
        p[2]
    } -> std::convertible_to<float>;
};

} // namespace detail

/**
 * @brief Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
 * @tparam Point Type of the point
 * @param x 3D point located within the unit cube [0,1]
 * @return Morton code of x
 */
template <detail::CMorton3dPoint Point>
[[maybe_unused]] PBAT_HOST_DEVICE inline MortonCodeType Morton3D(Point x)
{
    using namespace std;
    using ScalarType  = std::remove_cvref_t<decltype(x[0])>;
    MortonCodeType xx = ExpandBits(static_cast<MortonCodeType>(
        min(max(x[0] * ScalarType(1024), ScalarType(0)), ScalarType(1023))));
    MortonCodeType yy = ExpandBits(static_cast<MortonCodeType>(
        min(max(x[1] * ScalarType(1024), ScalarType(0)), ScalarType(1023))));
    MortonCodeType zz = ExpandBits(static_cast<MortonCodeType>(
        min(max(x[2] * ScalarType(1024), ScalarType(0)), ScalarType(1023))));
    return xx * 4 + yy * 2 + zz;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MORTON_H
