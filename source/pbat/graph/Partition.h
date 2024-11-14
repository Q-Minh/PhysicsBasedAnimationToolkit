#ifndef PBAT_GRAPH_PARTITION_H
#define PBAT_GRAPH_PARTITION_H

#include "pbat/Aliases.h"

#include <utility>
#include <vector>

namespace pbat {
namespace graph {

#ifdef PBAT_USE_METIS
std::vector<Index> Partition(
    std::vector<Index> const& ptr,
    std::vector<Index> const& adj,
    std::vector<Index> const& wadj,
    std::size_t nPartitions);
#endif PBAT_USE_METIS

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_PARTITION_H