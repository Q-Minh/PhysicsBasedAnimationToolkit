#ifndef PBAT_GEOMETRY_SDF_FOREST_H
#define PBAT_GEOMETRY_SDF_FOREST_H

#include "Composite.h"
#include "Transform.h"
#include "pbat/common/Concepts.h"

#include <utility>
#include <vector>

namespace pbat::geometry::sdf {

/**
 * @brief CPU storage for a forest (of SDFs).
 */
template <common::CArithmetic TScalar>
struct Forest
{
    using ScalarType = TScalar;          ///< Scalar type
    std::vector<Node<ScalarType>> nodes; ///< `|# nodes|` nodes in the forest
    std::vector<Transform<ScalarType>>
        transforms;         ///< `|# nodes|` transforms associated to each node
    std::vector<int> roots; ///< `|# roots|` indices of the root nodes in the forest
    std::vector<std::pair<int, int>> children; ///< `|# nodes|` list of pairs of children indices
                                               ///< for each node, such that c* < 0 if no child
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_FOREST_H
