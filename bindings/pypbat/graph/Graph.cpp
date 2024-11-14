#include "Graph.h"

#include "Partitioning.h"

namespace pbat {
namespace py {
namespace graph {

void Bind(pybind11::module& m)
{
    BindPartitioning(m);
}

} // namespace graph
} // namespace py
} // namespace pbat
