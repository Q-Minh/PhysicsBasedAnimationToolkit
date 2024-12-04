#ifndef PYPBAT_GRAPH_PARTITION_H
#define PYPBAT_GRAPH_PARTITION_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace graph {

void BindPartition(pybind11::module& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_PARTITION_H