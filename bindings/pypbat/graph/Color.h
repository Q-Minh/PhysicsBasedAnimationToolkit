#ifndef PYPBAT_GRAPH_COLOR_H
#define PYPBAT_GRAPH_COLOR_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace graph {

void BindColor(pybind11::module& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_COLOR_H