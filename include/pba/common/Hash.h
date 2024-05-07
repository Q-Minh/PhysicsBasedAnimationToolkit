#ifndef PBA_CORE_COMMON_H
#define PBA_CORE_COMMON_H

#include <cstddef>

namespace pba {
namespace common {

template <typename T>
void hash_combine_accumulate(std::size_t& seed, T const& val)
{
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename... Types>
std::size_t hash_combine(const Types&... args)
{
    std::size_t seed = 0;
    (hash_combine_accumulate(seed, args), ...); // create hash value with seed over all args
    return seed;
}

} // namespace common
} // namespace pba

#endif // PBA_CORE_COMMON_H