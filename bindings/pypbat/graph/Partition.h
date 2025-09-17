#ifndef PYPBAT_GRAPH_PARTITION_H
#define PYPBAT_GRAPH_PARTITION_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace graph {

void BindPartition(nanobind::module_& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_PARTITION_H