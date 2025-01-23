#include "Bvh.h"

#include <pbat/gpu/geometry/Aabb.h>
#include <pbat/gpu/geometry/Bvh.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBvh([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    using namespace pbat::gpu::geometry;
    using pbat::gpu::common::Buffer;
    pyb::class_<Bvh>(m, "Bvh")
        .def(
            pyb::init([](GpuIndex nPrimitives, GpuIndex nOverlaps) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Bvh.Construct", [&]() {
                    Bvh bvh(nPrimitives, nOverlaps);
                    return bvh;
                });
            }),
            pyb::arg("max_boxes"),
            pyb::arg("max_overlaps"),
            "Allocate BVH on GPU for max_boxes primitives, which can detect a maximum of "
            "max_overlaps box overlaps.")
        .def(
            "build",
            [](Bvh& bvh,
               Aabb& aabbs,
               Eigen::Vector<GpuScalar, 3> const& min,
               Eigen::Vector<GpuScalar, 3> const& max) {
                pbat::profiling::Profile("pbat.gpu.geometry.Bvh.Build", [&]() {
                    bvh.Build(aabbs, min, max);
                });
            },
            pyb::arg("aabbs"),
            pyb::arg("min"),
            pyb::arg("max"),
            "Constructs, on the GPU, a bounding volume hierarchy of axis-aligned boxes. (min,max) "
            "denote the extremeties of an axis-aligned bounding box embedding."
            "Args:\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives\n"
            "min (np.ndarray[3]): World axis-aligned box's min endpoint\n"
            "max (np.ndarray[3]): World axis-aligned box's max endpoint")
        .def(
            "detect_overlaps",
            [](Bvh& bvh, Aabb const& aabbs) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Bvh.DetectOverlaps", [&]() {
                    return bvh.DetectOverlaps(aabbs);
                });
            },
            pyb::arg("aabbs"),
            "Detect self-overlaps (bi,bj) between bounding boxes of aabbs into a "
            "2x|#overlaps| array, where bi < bj. aabbs must be the one used in the "
            "last call to build()."
            "Args:\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives")
        .def(
            "detect_overlaps",
            [](Bvh& bvh, Buffer const& set, Aabb const& aabbs) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Bvh.DetectOverlaps", [&]() {
                    return bvh.DetectOverlaps(set, aabbs);
                });
            },
            pyb::arg("aabbs"),
            pyb::arg("set"),
            "Detect self-overlaps (bi,bj) between bounding boxes of aabbs into a "
            "2x|#overlaps| array, where bi < bj. Additionally, we only consider pairs (bi,bj) s.t. "
            "set[bi] != set[bj], i.e. overlaps are detected between different sets. aabbs must be "
            "the one used in the last call to build().\n"
            "Args:\n"
            "set (np.ndarray): Map of indices of aabbs to their corresponding set, i.e. set[i] = "
            "j, where i is a box and j is its corresponding set.\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives")
        .def_property_readonly("min", &Bvh::Min, "BVH nodes' box minimums")
        .def_property_readonly("max", &Bvh::Max, "BVH nodes' box maximums")
        .def_property_readonly(
            "ordering",
            &Bvh::LeafOrdering,
            "Box indices ordered by Morton encoding")
        .def_property_readonly("morton", &Bvh::MortonCodes, "Sorted morton codes of simplices")
        .def_property_readonly(
            "child",
            &Bvh::Child,
            "Radix-tree left and right children of each node as a |#simplices - 1|x2 array")
        .def_property_readonly(
            "parent",
            &Bvh::Parent,
            "Radix-tree parents of each node as a |2*#simplices - 1| array")
        .def_property_readonly(
            "rightmost",
            &Bvh::Rightmost,
            "Radix-tree left-subtree and right-subtree largest nodal indices as a |#simplices - "
            "1|x2 array")
        .def_property_readonly(
            "visits",
            &Bvh::Visits,
            "Number of visits per internal node for bounding box computation as a |#simplices - 1| "
            "array");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat