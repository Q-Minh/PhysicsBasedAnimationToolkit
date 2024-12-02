#include "Graph.h"

#include "Partition.h"

namespace pbat {
namespace py {
namespace graph {

void Bind(pybind11::module& m)
{
    BindPartition(m);
}

} // namespace graph
} // namespace py
} // namespace pbat
