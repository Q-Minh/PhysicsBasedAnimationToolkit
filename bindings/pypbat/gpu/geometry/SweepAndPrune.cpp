#include "SweepAndPrune.h"

#include <cstddef>
#include <pbat/gpu/geometry/Aabb.h>
#include <pbat/gpu/geometry/SweepAndPrune.h>
#include <pbat/profiling/Profiling.h>
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
    pyb::class_<SweepAndPrune>(m, "SweepAndPrune")
        .def(
            pyb::init([](std::size_t maxBoxes, std::size_t maxOverlaps) {
                return pbat::profiling::Profile("pbat.gpu.geometry.SweepAndPrune.Construct", [=]() {
                    SweepAndPrune sap(maxBoxes, maxOverlaps);
                    return sap;
                });
            }),
            pyb::arg("max_boxes"),
            pyb::arg("max_overlaps"))
        .def(
            "sort_and_sweep",
            [](SweepAndPrune& sap, Aabb& aabbs) {
                return pbat::profiling::Profile(
                    "pbat.gpu.geometry.SweepAndPrune.SortAndSweep",
                    [&]() {
                        auto O = sap.SortAndSweep(aabbs);
                        return O;
                    });
            },
            pyb::arg("aabbs"),
            "Detect all overlaps between bounding boxes in aabbs.")
        .def(
            "sort_and_sweep",
            [](SweepAndPrune& sap, GpuIndex n, Aabb& aabbs) {
                return pbat::profiling::Profile(
                    "pbat.gpu.geometry.SweepAndPrune.SortAndSweep",
                    [&]() {
                        auto O = sap.SortAndSweep(n, aabbs);
                        return O;
                    });
            },
            pyb::arg("n"),
            pyb::arg("aabbs"),
            "Detect all overlaps between bounding boxes of subsets S1 (of size n) and S2 (of size "
            "#aabbs-n) of aabbs.\n"
            "Args:\n"
            "n (int): Number of primitives in the first set [0, n)\n"
            "aabbs (Aabb): AABBs over primitives of the first [0,n) and second set [n, "
            "aabbs.size())");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat