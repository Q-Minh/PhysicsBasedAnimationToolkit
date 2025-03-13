#include "Aabb.h"

#include <pbat/gpu/geometry/Aabb.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindAabb([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;

    using namespace pbat::gpu;
    using namespace pbat::gpu::geometry;

    pyb::class_<Aabb>(m, "Aabb")
        .def(
            pyb::init<GpuIndex, GpuIndex>(),
            pyb::arg("dims")   = 3,
            pyb::arg("n_aabb") = 0,
            "Allocate memory for n_aabb axis-aligned bounding boxes in dims dimensions")
        .def("resize", &Aabb::Resize, pyb::arg("dims"), pyb::arg("n_aabb"), "Resize aabb array")
        .def(
            "construct",
            [](Aabb& aabb,
               Eigen::Ref<GpuMatrixX const> const& L,
               Eigen::Ref<GpuMatrixX const> const& U) { aabb.Construct(L, U); },
            pyb::arg("min"),
            pyb::arg("max"),
            "Set min/max endpoints of aabbs as |#dims|x|#boxes| arrays")
        .def(
            "construct",
            [](Aabb& aabb,
               Eigen::Ref<GpuMatrixX const> const& P,
               Eigen::Ref<GpuIndexMatrixX const> const& S) { aabb.Construct(P, S); },
            pyb::arg("P"),
            pyb::arg("S"),
            "Construct aabbs from simplex mesh (P,S)\n\n"
            "Args:\n"
            "P (np.ndarray): 3x|#pts| array of points\n"
            "S (np.ndarray): Kx|#simplices| array of simplices where K is the number of vertices "
            "per simplex")
        .def_property_readonly("n_boxes", &Aabb::Size, "Number of aabbs")
        .def_property_readonly("dims", &Aabb::Dimensions, "Embedding dimensionality")
        .def_property_readonly("min", &Aabb::Lower, "min endpoints of aabbs")
        .def_property_readonly("max", &Aabb::Upper, "max endpoints of aabbs");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat