#ifndef PYPBAT_GRAPH_ADJACENCY_H
#define PYPBAT_GRAPH_ADJACENCY_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace graph {

void BindAdjacency(pybind11::module& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_ADJACENCY_H