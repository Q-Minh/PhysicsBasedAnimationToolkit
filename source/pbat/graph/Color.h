#ifndef PBAT_GRAPH_COLOR_H
#define PBAT_GRAPH_COLOR_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Stack.h"

#include <algorithm>
#include <concepts>
#include <ranges>

namespace pbat {
namespace graph {

enum class EGreedyColorSelectionStrategy { LeastUsed, FirstAvailable };
enum class EGreedyColorOrderingStrategy { Natural, SmallestDegree, LargestDegree };

template <
    class TDerivedPtr,
    class TDerivedAdj,
    int NC               = 128,
    std::integral TIndex = typename TDerivedPtr::Scalar>
Eigen::Vector<TIndex, Eigen::Dynamic> GreedyColor(
    Eigen::DenseBase<TDerivedPtr> const& ptr,
    Eigen::DenseBase<TDerivedAdj> const& adj,
    EGreedyColorOrderingStrategy eOrderingStrategy   = EGreedyColorOrderingStrategy::Natural,
    EGreedyColorSelectionStrategy eSelectionStrategy = EGreedyColorSelectionStrategy::LeastUsed)
{
    common::Stack<TIndex, NC> palette{};
    common::Stack<bool, NC> usable{};
    TIndex const n        = ptr.size() - 1;
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    using Colors          = IndexVectorType;
    Colors C(n);
    C.setConstant(TIndex(-1));
    // Compute vertex visiting order
    IndexVectorType ordering(n);
    std::ranges::copy(std::views::iota(TIndex(0), n), ordering.data());
    switch (eOrderingStrategy)
    {
        case EGreedyColorOrderingStrategy::Natural: break;
        case EGreedyColorOrderingStrategy::SmallestDegree:
            ordering = common::ArgSort(n, [&](auto i, auto j) {
                auto di = ptr(i + 1) - ptr(i);
                auto dj = ptr(j + 1) - ptr(j);
                return di < dj;
            });
        case EGreedyColorOrderingStrategy::LargestDegree:
            ordering = common::ArgSort(n, [&](auto i, auto j) {
                auto di = ptr(i + 1) - ptr(i);
                auto dj = ptr(j + 1) - ptr(j);
                return di > dj;
            });
        default: break;
    }
    // Color vertices in user-defined order
    for (TIndex u : ordering)
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
                bool bLess{true};
                switch (eSelectionStrategy)
                {
                    case EGreedyColorSelectionStrategy::LeastUsed:
                        bLess =
                            (usable[ci] and usable[cj]) ? palette[ci] < palette[cj] : usable[ci];
                        break;
                    case EGreedyColorSelectionStrategy::FirstAvailable:
                        bLess = (usable[ci] and usable[cj]) ? ci < cj : usable[ci];
                        break;
                    default: break;
                }
                return bLess;
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