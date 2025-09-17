#include "Graph.h"

#include "Adjacency.h"
#include "Color.h"
#include "Mesh.h"
#include "Partition.h"

namespace pbat {
namespace py {
namespace graph {

void Bind(nanobind::module_& m)
{
    BindAdjacency(m);
    BindColor(m);
    BindMesh(m);
    BindPartition(m);
}

} // namespace graph
} // namespace py
} // namespace pbat
