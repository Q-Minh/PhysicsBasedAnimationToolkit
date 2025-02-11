/**
 * @file Hash.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Hash functions for std::pair, std::tuple, and pbat::IndexVector
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup common
 */

#ifndef PBAT_COMMON_H
#define PBAT_COMMON_H

#include "pbat/Aliases.h"

#include <cstddef>
#include <tuple>
#include <utility>

namespace pbat {
namespace common {

/**
 * @brief Incrementally combine hash values of multiple arguments
 *
 * @tparam T Hashable type
 * @param seed Starting hash value
 * @param val Value to hash
 * @ingroup common
 */
template <typename T>
void HashCombineAccumulate(std::size_t& seed, T const& val)
{
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * @brief Combine hash values of multiple arguments
 *
 * @tparam Types Hashable types
 * @param args Arguments to hash
 * @return std::size_t Hash value
 * @ingroup common
 */
template <typename... Types>
std::size_t HashCombine(const Types&... args)
{
    std::size_t seed = 0;
    (HashCombineAccumulate(seed, args), ...); // create hash value with seed over all args
    return seed;
}

} // namespace common
} // namespace pbat

namespace std {

template <>
struct hash<pair<pbat::Index, pbat::Index>>
{
    [[maybe_unused]] std::size_t operator()(pair<pbat::Index, pbat::Index> const& inds) const
    {
        return pbat::common::HashCombine(inds.first, inds.second);
    }
};

template <>
struct hash<tuple<pbat::Index, pbat::Index>>
{
    [[maybe_unused]] std::size_t operator()(tuple<pbat::Index, pbat::Index> const& inds) const
    {
        return pbat::common::HashCombine(get<0>(inds), get<1>(inds));
    }
};

template <>
struct hash<tuple<pbat::Index, pbat::Index, pbat::Index>>
{
    [[maybe_unused]] std::size_t
    operator()(tuple<pbat::Index, pbat::Index, pbat::Index> const& inds) const
    {
        return pbat::common::HashCombine(get<0>(inds), get<1>(inds), get<2>(inds));
    }
};

template <>
struct hash<pbat::IndexVector<2>>
{
    [[maybe_unused]] std::size_t operator()(pbat::IndexVector<2> const& inds) const
    {
        return pbat::common::HashCombine(inds(0), inds(1));
    }
};

template <>
struct hash<pbat::IndexVector<3>>
{
    [[maybe_unused]] std::size_t operator()(pbat::IndexVector<3> const& inds) const
    {
        return pbat::common::HashCombine(inds(0), inds(1), inds(2));
    }
};

} // namespace std

#endif // PBAT_COMMON_H