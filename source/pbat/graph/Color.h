#ifndef PBAT_GRAPH_COLOR_H
#define PBAT_GRAPH_COLOR_H

#include "pbat/Aliases.h"
#include "pbat/common/Stack.h"

#include <algorithm>
#include <concepts>
#include <ranges>

namespace pbat {
namespace graph {

template <
    class TDerivedPtr,
    class TDerivedAdj,
    int NC               = 128,
    std::integral TIndex = typename TDerivedPtr::Scalar>
Eigen::Vector<TIndex, Eigen::Dynamic>
GreedyColor(Eigen::DenseBase<TDerivedPtr> const& ptr, Eigen::DenseBase<TDerivedAdj> const& adj)
{
    common::Stack<TIndex, NC> palette{};
    common::Stack<bool, NC> usable{};
    auto const n = ptr.size() - 1;
    using Colors = Eigen::Vector<TIndex, Eigen::Dynamic>;
    Colors C(n);
    C.setConstant(TIndex(-1));
    for (auto u = 0; u < n; ++u)
    {
        // Reset usable color flags
        std::fill(usable.begin(), usable.end(), true);
        // Determine all unusable colors
        auto vBegin = ptr(u);
        auto vEnd   = ptr(u + TIndex(1));
        auto nUnusable{0};
        for (auto k = vBegin; k < vEnd; ++k)
        {
            auto v  = adj(k);
            auto cv = C(v);
            if (cv >= 0 and usable[cv])
            {
                usable[cv] = false;
                ++nUnusable;
            }
        }
        // Set vertex color
        bool bPaletteInsufficient = nUnusable == palette.Size();
        if (bPaletteInsufficient)
        {
            // Add color to palette
            C(u) = palette.Size();
            palette.Push(TIndex(1));
            usable.Push({});
        }
        else
        {
            // Find, in palette, the usable color with the smallest size, ignoring unusable colors.
            auto colors = std::views::iota(0, palette.Size());
            auto c      = *std::ranges::min_element(colors, [&](auto ci, auto cj) {
                return (usable[ci] and usable[cj]) ? palette[ci] < palette[cj] : usable[ci];
            });
            C(u)        = c;
            ++palette[c];
        }
    }
    return C;
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_COLOR_H