#ifndef PBAT_COMMON_PERMUTE_H
#define PBAT_COMMON_PERMUTE_H

#include <algorithm>
#include <concepts>
#include <iterator>
#include <utility>

namespace pbat::common {

/**
 * @brief
 *
 * @tparam TValuesBegin
 * @tparam TValuesEnd
 * @tparam TPermutationBegin
 * @tparam TPermutationEnd
 * @param vb
 * @param ve
 * @param pb
 * @param pe
 */
template <
    std::random_access_iterator TValuesBegin,
    std::random_access_iterator TValuesEnd,
    std::random_access_iterator TPermutationBegin,
    std::random_access_iterator TPermutationEnd>
void Permute(TValuesBegin vb, TValuesEnd ve, TPermutationBegin pb, TPermutationEnd pe)
{
    using PermutationIndex = std::iterator_traits<TPermutationBegin>::value_type;
    static_assert(
        std::is_integral_v<PermutationIndex> and std::is_signed_v<PermutationIndex>,
        "Permutation index must be signed integer");
    auto n   = std::distance(vb, ve);
    auto vit = vb;
    auto pit = pb;
    for (decltype(n) i = 0; i < n; ++i)
    {
        auto pit = pb + i;
        while (*pit >= 0)
        {
            PermutationIndex const p = *pit;
            std::iter_swap(vb + i, vb + p);
            *pit -= n;
            pit = pb + p;
        }
    }
    for (decltype(n) i = 0; i < n; ++i)
    {
        *(pb + i) += n;
    }
}

} // namespace pbat::common

#endif // PBAT_COMMON_PERMUTE_H
