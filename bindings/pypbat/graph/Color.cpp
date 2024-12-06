#include "Color.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/graph/Color.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace graph {

void BindColor(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "greedy_color",
        [](Eigen::Ref<IndexVectorX const> const& ptr,
           Eigen::Ref<IndexVectorX const> const& adj,
           int NC) {
            IndexVectorX C;
            pbat::common::ForValues<2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096>(
                [&]<int kNC>() {
                    if (NC == kNC)
                        C = pbat::graph::GreedyColor(ptr, adj);
                });
            return C;
        },
        pyb::arg("ptr"),
        pyb::arg("adj"),
        pyb::arg("cmax") = 128,
        "Returns a graph coloring of the compressed sparse format graph (ptr,adj) using a greedy "
        "approach.\n"
        "Args:\n"
        "ptr (np.ndarray): |#nodes+1| offset/ptr array\n"
        "adj (np.ndarray): |#edges| indices array\n"
        "cmax (int): Maximum number of colors to allocate");
}

} // namespace graph
} // namespace py
} // namespace pbat