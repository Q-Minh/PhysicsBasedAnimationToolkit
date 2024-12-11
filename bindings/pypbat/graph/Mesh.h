#ifndef PYPBAT_GRAPH_MESH_H
#define PYPBAT_GRAPH_MESH_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace graph {

void BindMesh(pybind11::module& m);

} // namespace graph
} // namespace py
} // namespace pbat

#endif // PYPBAT_GRAPH_MESH_H