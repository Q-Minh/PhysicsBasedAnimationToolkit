/**
 * @file Enums.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Enums for graph algorithms API
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup graph
 */

#ifndef PBAT_GRAPH_ENUMS_H
#define PBAT_GRAPH_ENUMS_H

namespace pbat {
namespace graph {

/**
 * @brief Enumeration of color selection strategies for graph coloring algorithms
 * @ingroup graph
 */
enum class EGreedyColorSelectionStrategy {
    LeastUsed,     ///< Select the least used color from the color palette
    FirstAvailable ///< Select the first available color from the color palette
};

/**
 * @brief Enumeration of vertex traversal ordering strategies for graph coloring algorithms
 * @ingroup graph
 */
enum class EGreedyColorOrderingStrategy {
    Natural,        ///< Natural ordering of the vertices (i.e. [0,n-1])
    SmallestDegree, ///< Always visit the vertex with the smallest degree next
    LargestDegree   ///< Always visit the vertex with the largest degree next
};

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ENUMS_H
