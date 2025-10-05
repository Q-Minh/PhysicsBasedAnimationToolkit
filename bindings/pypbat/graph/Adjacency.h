#ifndef PYPBAT_GRAPH_ADJACENCY_H
#define PYPBAT_GRAPH_ADJACENCY_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace graph {

void BindAdjacency(nanobind::module_& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_ADJACENCY_H