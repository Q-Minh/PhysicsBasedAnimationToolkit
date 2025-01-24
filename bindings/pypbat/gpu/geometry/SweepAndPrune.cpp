#include "SweepAndPrune.h"

#include <cstddef>
#include <pbat/gpu/geometry/Aabb.h>
#include <pbat/gpu/geometry/SweepAndPrune.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindSweepAndPrune([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    using namespace pbat::gpu::geometry;
    using pbat::gpu::common::Buffer;
    pyb::class_<SweepAndPrune>(m, "SweepAndPrune")
        .def(pyb::init<std::size_t, std::size_t>(), pyb::arg("max_boxes"), pyb::arg("max_overlaps"))
        .def(
            "sort_and_sweep",
            [](SweepAndPrune& sap, Aabb& aabbs) { return sap.SortAndSweep(aabbs); },
            pyb::arg("aabbs"),
            "Detect all overlaps between bounding boxes in aabbs.")
        .def(
            "sort_and_sweep",
            [](SweepAndPrune& sap, Buffer const& set, Aabb& aabbs) {
                return sap.SortAndSweep(set, aabbs);
            },
            pyb::arg("set"),
            pyb::arg("aabbs"),
            "Detect all overlaps between bounding boxes of subsets S1 (of size n) and S2 (of size "
            "#aabbs-n) of aabbs.\n"
            "Args:\n"
            "set (np.ndarray): Map of indices of aabbs to their corresponding set, i.e. set[i] = "
            "j, where i is a box and j is its corresponding set.\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat