#include "Partitioning.h"

#include <pbat/graph/Partition.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace graph {

void BindPartitioning([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
#ifdef PBAT_USE_METIS
    m.def(
        "partition",
        pbat::graph::Partition,
        pyb::arg("ptr"),
        pyb::arg("adj"),
        pyb::arg("wgt"),
        pyb::arg("n_partitions"),
        "Partition a weighted directed graph (ptr, adj, wgt) given in sparse compressed format "
        "into n_partitions. Returns the |len(ptr)-1| partitioning map p such that p[v] yields the "
        "partition containing vertex v.");
#endif // PBAT_USE_METIS
}

} // namespace graph
} // namespace py
} // namespace pbat