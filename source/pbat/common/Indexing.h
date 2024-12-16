#ifndef PBAT_COMMON_INDEXING_H
#define PBAT_COMMON_INDEXING_H

#include "Concepts.h"
#include "pbat/Aliases.h"

#include <concepts>
#include <numeric>
#include <random>
#include <ranges>

namespace pbat {
namespace common {

template <CIndexRange R, std::integral TIndex = std::ranges::range_value_t<R>>
Eigen::Vector<TIndex, Eigen::Dynamic> CumSum(R&& sizes)
{
    namespace rng         = std::ranges;
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType cs(rng::size(sizes) + 1);
    auto bi    = cs.data();
    *bi++      = Index{0};
    auto begin = rng::begin(sizes);
    auto end   = rng::end(sizes);
    std::partial_sum(begin, end, bi);
    return cs;
}

template <std::integral TIndex>
Eigen::Vector<TIndex, Eigen::Dynamic> Counts(auto begin, auto end, TIndex ncounts)
{
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    IndexVectorType counts(ncounts);
    counts.setZero();
    for (auto it = begin; it != end; ++it)
        ++counts(*it);
    return counts;
}

template <std::integral TIndex>
Eigen::Vector<TIndex, Eigen::Dynamic> Shuffle(TIndex begin, TIndex end)
{
    auto iota = std::views::iota(begin, end);
    Eigen::Vector<TIndex, Eigen::Dynamic> inds(end - begin);
    std::copy(std::ranges::begin(iota), std::ranges::end(iota), inds.begin());
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::ranges::shuffle(inds, gen);
    return inds;
}

} // namespace common
} // namespace pbat

#endif // PBAT_COMMON_INDEXING_H