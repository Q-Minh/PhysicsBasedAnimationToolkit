#ifndef PBAT_COMMON_H
#define PBAT_COMMON_H

#include "pbat/Aliases.h"

#include <cstddef>
#include <tuple>
#include <utility>

namespace pbat {
namespace common {

template <typename T>
void HashCombineAccumulate(std::size_t& seed, T const& val)
{
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

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