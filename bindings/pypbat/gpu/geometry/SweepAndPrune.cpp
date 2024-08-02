#include "SweepAndPrune.h"

#include <cstddef>
#include <pbat/gpu/geometry/Primitives.h>
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
            [](SweepAndPrune& sap,
               Points const& P,
               Simplices const& S1,
               Simplices const& S2,
               Scalar expansion) {
                return pbat::profiling::Profile(
                    "pbat.gpu.geometry.SweepAndPrune.SortAndSweep",
                    [&]() {
                        IndexMatrixX O = sap.SortAndSweep(P, S1, S2, expansion);
                        return O;
                    });
            },
            pyb::arg("points"),
            pyb::arg("lsimplices"),
            pyb::arg("rsimplices"),
            pyb::arg("expansion") = 0.f,
            "Detect overlaps between bounding boxes of lsimplices and rsimplices, where lsimplices "
            "= rsimplices yields self-overlapping primitive pairs. Simplices in lsimplices and "
            "rsimplices must index into points.");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat