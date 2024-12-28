#ifndef PBAT_GRAPH_ENUMS_H
#define PBAT_GRAPH_ENUMS_H

namespace pbat {
namespace graph {

enum class EGreedyColorSelectionStrategy { LeastUsed, FirstAvailable };
enum class EGreedyColorOrderingStrategy { Natural, SmallestDegree, LargestDegree };

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_ENUMS_H
