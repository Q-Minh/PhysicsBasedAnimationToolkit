#include "Points.h"

#include <pbat/gpu/geometry/Primitives.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindPoints([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    pyb::class_<pbat::gpu::geometry::Points>(m, "Points")
        .def(
            pyb::init<Eigen::Ref<MatrixX const> const&>(),
            pyb::arg("V"),
            "Construct points on GPU via |#dims|x|#vertices| array of vertex positions")
        .def_property(
            "V",
            &pbat::gpu::geometry::Points::Get,
            &pbat::gpu::geometry::Points::Update,
            "Get/set points on GPU via |#dims|x|#vertices| array of vertex positions");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat