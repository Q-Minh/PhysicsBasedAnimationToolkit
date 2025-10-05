#include "Bvh.h"

#include <nanobind/eigen/dense.h>
#include <pbat/gpu/geometry/Aabb.h>
#include <pbat/gpu/geometry/Bvh.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBvh([[maybe_unused]] nanobind::module_& m)
{
#ifdef PBAT_USE_CUDA
    namespace nb = nanobind;
    using namespace pbat::gpu::geometry;
    using pbat::gpu::common::Buffer;
    nb::class_<Bvh>(m, "Bvh")
        .def(
            nb::init<GpuIndex, GpuIndex>(),
            nb::arg("max_boxes"),
            nb::arg("max_overlaps"),
            "Allocate BVH on GPU for max_boxes primitives, which can detect a maximum of "
            "max_overlaps box overlaps.")
        .def(
            "build",
            &Bvh::Build,
            nb::arg("aabbs"),
            nb::arg("min"),
            nb::arg("max"),
            "Constructs, on the GPU, a bounding volume hierarchy of axis-aligned boxes. (min,max) "
            "denote the extremeties of an axis-aligned bounding box embedding."
            "Args:\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives\n"
            "min (np.ndarray[3]): World axis-aligned box's min endpoint\n"
            "max (np.ndarray[3]): World axis-aligned box's max endpoint")
        .def(
            "detect_overlaps",
            [](Bvh& bvh, Aabb const& aabbs) { return bvh.DetectOverlaps(aabbs); },
            nb::arg("aabbs"),
            "Detect self-overlaps (bi,bj) between bounding boxes of aabbs into a "
            "2x|#overlaps| array, where bi < bj. aabbs must be the one used in the "
            "last call to build()."
            "Args:\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives")
        .def(
            "detect_overlaps",
            [](Bvh& bvh, Buffer const& set, Aabb const& aabbs) {
                return bvh.DetectOverlaps(set, aabbs);
            },
            nb::arg("aabbs"),
            nb::arg("set"),
            "Detect self-overlaps (bi,bj) between bounding boxes of aabbs into a "
            "2x|#overlaps| array, where bi < bj. Additionally, we only consider pairs (bi,bj) s.t. "
            "set[bi] != set[bj], i.e. overlaps are detected between different sets. aabbs must be "
            "the one used in the last call to build().\n"
            "Args:\n"
            "set (np.ndarray): Map of indices of aabbs to their corresponding set, i.e. set[i] = "
            "j, where i is a box and j is its corresponding set.\n"
            "aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over primitives")
        .def(
            "point_triangle_nearest_neighbours",
            &Bvh::PointTriangleNearestNeighbors,
            nb::arg("aabbs"),
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            "Find the nearest triangle to each point in X. The output is a |#X| matrix of nearest "
            "triangle indices to corresponding columns in X.\n\n"
            "Args:\n"
            "   aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over triangles given "
            "to Build()\n"
            "   X (np.ndarray): 3x|#pts| query points to find nearest triangles to\n"
            "   V (np.ndarray): Triangle vertex positions\n"
            "   F (np.ndarray): Triangle vertex indices")
        .def(
            "point_tetrahedron_nearest_neighbours",
            &Bvh::PointTetrahedronNearestNeighbors,
            nb::arg("aabbs"),
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("T"),
            "Find the nearest tetrahedron to each point in X. The output is a |#X| matrix of "
            "nearest tetrahedron indices to corresponding columns in X.\n\n"
            "Args:\n"
            "   aabbs (pbat.gpu.geometry.Aabb): Axis-aligned bounding boxes over tetrahedra given "
            "to Build()\n"
            "   X (np.ndarray): 3x|#pts| query points to find nearest tetrahedra to\n"
            "   V (np.ndarray): Tetrahedron vertex positions\n"
            "   T (np.ndarray): Tetrahedron vertex indices")
        .def_prop_ro("min", &Bvh::Min, "BVH nodes' box minimums")
        .def_prop_ro("max", &Bvh::Max, "BVH nodes' box maximums")
        .def_prop_ro("ordering", &Bvh::LeafOrdering, "Box indices ordered by Morton encoding")
        .def_prop_ro("morton", &Bvh::MortonCodes, "Sorted morton codes of simplices")
        .def_prop_ro(
            "child",
            &Bvh::Child,
            "Radix-tree left and right children of each node as a |#simplices - 1|x2 array")
        .def_prop_ro(
            "parent",
            &Bvh::Parent,
            "Radix-tree parents of each node as a |2*#simplices - 1| array")
        .def_prop_ro(
            "rightmost",
            &Bvh::Rightmost,
            "Radix-tree left-subtree and right-subtree largest nodal indices as a |#simplices - "
            "1|x2 array")
        .def_prop_ro(
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