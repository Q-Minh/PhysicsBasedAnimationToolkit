#include "SweepAndPrune.h"

#include <cstddef>
#include <pbat/gpu/geometry/Primitives.h>
#include <pbat/gpu/geometry/SweepAndPrune.h>

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
        .def(pyb::init<std::size_t, std::size_t>(), pyb::arg("max_boxes"), pyb::arg("max_overlaps"))
        .def(
            "sort_and_sweep",
            &SweepAndPrune::SortAndSweep,
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