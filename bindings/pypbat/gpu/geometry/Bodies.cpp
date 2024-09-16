#include "Bodies.h"

#include <pbat/gpu/geometry/Primitives.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBodies([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;

    using namespace pbat::gpu;
    using namespace pbat::gpu::geometry;

    pyb::class_<Bodies>(m, "Bodies")
        .def(
            pyb::init<Eigen::Ref<GpuIndexMatrixX const> const&>(),
            pyb::arg("B"),
            "Construct bodies of Simplices on GPU via |#simplices|x1 array of body indices b"
            "associated with corresponding simplices s, such that B[s] = b")
        .def_property_readonly("n_bodies", &Bodies::NumberOfBodies, "Number of bodies")
        .def_property_readonly("body", &Bodies::Get, "Load GPU bodies to CPU");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat