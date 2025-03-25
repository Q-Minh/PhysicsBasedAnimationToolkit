#ifndef PBAT_COMMON_PERMUTE_H
#define PBAT_COMMON_PERMUTE_H

#include <concepts>
#include <iterator>

namespace pbat::common {

/**
 * @brief Permute the values in-place according to the permutation
 *
 * Taken from [SO](https://stackoverflow.com/a/60917997/8239925)
 *
 * @tparam TValuesBegin Iterator type to the beginning of the values
 * @tparam TValuesEnd Iterator type to the end of the values
 * @tparam TPermutationBegin Iterator type to the beginning of the permutation
 * @param vb Iterator to the beginning of values
 * @param ve Iterator to the end of values
 * @param pb Iterator to the beginning of permutation
 *
 * @note The permutation is modified in-place for the duration of the function, but is restored to
 * its original state before returning.
 *
 * @post The permutation referenced by `pb` is un-modified.
 * @post The values referenced by `[vb, ve)` are permuted according to the permutation.
 */
template <
    std::random_access_iterator TValuesBegin,
    std::random_access_iterator TValuesEnd,
    std::random_access_iterator TPermutationBegin>
void Permute(TValuesBegin vb, TValuesEnd ve, TPermutationBegin pb)
{
    using PermutationIndex = std::iterator_traits<TPermutationBegin>::value_type;
    static_assert(
        std::is_integral_v<PermutationIndex> and std::is_signed_v<PermutationIndex>,
        "Permutation index must be signed integer");
    auto n            = std::distance(vb, ve);
    using PtrDiffType = decltype(n);
    for (PtrDiffType i = 0; i < n; ++i)
    {
        auto pi = *(pb + i);
        if (pi < 0)
            continue;
        auto value = *(vb + i);
        auto xi    = i;
        while (pi != i)
        {
            *(pb + xi) -= n;
            *(vb + xi) = *(vb + pi);
            xi         = pi;
            pi         = *(pb + xi);
        }
        *(vb + xi) = value;
        *(pb + xi) -= n;
    }
    for (PtrDiffType i = 0; i < n; ++i)
    {
        *(pb + i) += n;
    }
}

} // namespace pbat::common

#endif // PBAT_COMMON_PERMUTE_H
