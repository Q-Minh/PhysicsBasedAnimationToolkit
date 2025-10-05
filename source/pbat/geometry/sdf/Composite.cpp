#include "Composite.h"

namespace pbat::geometry::sdf {

auto FindRootsAndParents(std::span<std::pair<int, int> const> children)
    -> std::pair<std::vector<int>, std::vector<int>>
{
    auto const nNodes = children.size();
    std::vector<int> parents(nNodes, -1);
    std::vector<int> roots;
    for (auto n = 0ULL; n < nNodes; ++n)
    {
        auto const [ci, cj] = children[n];
        if (ci >= 0)
            parents[static_cast<std::size_t>(ci)] = static_cast<int>(n);
        if (cj >= 0)
            parents[static_cast<std::size_t>(cj)] = static_cast<int>(n);
    }
    for (auto n = 0ULL; n < nNodes; ++n)
        if (parents[n] < 0)
            roots.push_back(static_cast<int>(n));
    return {roots, parents};
}

} // namespace pbat::geometry::sdf
