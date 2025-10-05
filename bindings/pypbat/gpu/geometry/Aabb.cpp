#include "Aabb.h"

#include <pbat/gpu/geometry/Aabb.h>
#include <nanobind/eigen/dense.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindAabb([[maybe_unused]] nanobind::module_& m)
{
#ifdef PBAT_USE_CUDA
    namespace nb = nanobind;

    using namespace pbat::gpu;
    using namespace pbat::gpu::geometry;

    nb::class_<Aabb>(m, "Aabb")
        .def(
            nb::init<GpuIndex, GpuIndex>(),
            nb::arg("dims")   = 3,
            nb::arg("n_aabb") = 0,
            "Allocate memory for n_aabb axis-aligned bounding boxes in dims dimensions")
        .def("resize", &Aabb::Resize, nb::arg("dims"), nb::arg("n_aabb"), "Resize aabb array")
        .def(
            "construct",
            [](Aabb& aabb,
               Eigen::Ref<GpuMatrixX const> const& L,
               Eigen::Ref<GpuMatrixX const> const& U) { aabb.Construct(L, U); },
            nb::arg("min"),
            nb::arg("max"),
            "Set min/max endpoints of aabbs as |#dims|x|#boxes| arrays")
        .def(
            "construct",
            [](Aabb& aabb,
               Eigen::Ref<GpuMatrixX const> const& P,
               Eigen::Ref<GpuIndexMatrixX const> const& S) { aabb.Construct(P, S); },
            nb::arg("P"),
            nb::arg("S"),
            "Construct aabbs from simplex mesh (P,S)\n\n"
            "Args:\n"
            "P (np.ndarray): 3x|#pts| array of points\n"
            "S (np.ndarray): Kx|#simplices| array of simplices where K is the number of vertices "
            "per simplex")
        .def_prop_ro("n_boxes", &Aabb::Size, "Number of aabbs")
        .def_prop_ro("dims", &Aabb::Dimensions, "Embedding dimensionality")
        .def_prop_ro("min", &Aabb::Lower, "min endpoints of aabbs")
        .def_prop_ro("max", &Aabb::Upper, "max endpoints of aabbs");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat