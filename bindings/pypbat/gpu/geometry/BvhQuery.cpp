#include "BvhQuery.h"

#include <pbat/gpu/geometry/Bvh.h>
#include <pbat/gpu/geometry/BvhQuery.h>
#include <pbat/gpu/geometry/Primitives.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindBvhQuery(pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    using namespace pbat::gpu::geometry;
    pyb::class_<BvhQuery>(m, "BvhQuery")
        .def(
            pyb::init(
                [](std::size_t nPrimitives, std::size_t nOverlaps, std::size_t nNearestNeighbours) {
                    return pbat::profiling::Profile("pbat.gpu.geometry.BvhQuery.Construct", [&]() {
                        BvhQuery bvhQuery(nPrimitives, nOverlaps, nNearestNeighbours);
                        return bvhQuery;
                    });
                }),
            pyb::arg("max_boxes"),
            pyb::arg("max_overlaps"),
            pyb::arg("max_nearest_neighbours"),
            "Allocate data on GPU for max_boxes queries, which can detect a maximum of "
            "max_overlaps box overlaps, and handle max_nearest_neighbours nearest neighbour "
            "queries.")
        .def(
            "build",
            [](BvhQuery& bvhQuery,
               Points const& P,
               Simplices const& S,
               Eigen::Vector<GpuScalar, 3> const& min,
               Eigen::Vector<GpuScalar, 3> const& max,
               GpuScalar expansion) {
                pbat::profiling::Profile("pbat.gpu.geometry.BvhQuery.Build", [&]() {
                    bvhQuery.Build(P, S, min, max, expansion);
                });
            },
            pyb::arg("P"),
            pyb::arg("S"),
            pyb::arg("min"),
            pyb::arg("max"),
            pyb::arg("expansion") = GpuScalar{0},
            "Prepares, on the GPU, the queried simplices S for overlap tests against downstream "
            "simplex sets. Morton encoding is used to sort S, using min and max as an embedding "
            "axis-aligned bounding box for (P,S).")
        .def(
            "detect_overlaps",
            [](BvhQuery& bvhQuery,
               Points const& P,
               Simplices const& S1,
               Simplices const& S2,
               Bvh const& bvh) {
                return pbat::profiling::Profile("pbat.gpu.geometry.BvhQuery.DetectOverlaps", [&]() {
                    return bvhQuery.DetectOverlaps(P, S1, S2, bvh);
                });
            },
            pyb::arg("P"),
            pyb::arg("S1"),
            pyb::arg("S2"),
            pyb::arg("bvh"),
            "Detect self-overlaps (si,sj) between bounding boxes of simplices si in S1, sj in S2 "
            "into a 2x|#overlaps| array. Both S1 and S2 must index into points P and S1 was used "
            "in the call to build.")
        .def(
            "detect_contact_pairs",
            [](BvhQuery& bvhQuery,
               Points const& P,
               Simplices const& S1,
               Simplices const& S2,
               Bodies const& B1,
               Bodies const& B2,
               Bvh const& bvh,
               GpuScalar dhat,
               GpuScalar dzero) {
                return pbat::profiling::Profile(
                    "pbat.gpu.geometry.BvhQuery.DetectContactPairs",
                    [&]() {
                        return bvhQuery
                            .DetectContactPairsFromOverlaps(P, S1, S2, B1, B2, bvh, dhat, dzero);
                    });
            },
            pyb::arg("P"),
            pyb::arg("S1"),
            pyb::arg("S2"),
            pyb::arg("B1"),
            pyb::arg("B2"),
            pyb::arg("bvh"),
            pyb::arg("dhat")  = std::numeric_limits<GpuScalar>::max(),
            pyb::arg("dzero") = std::numeric_limits<GpuScalar>::epsilon(),
            "Detect contact pairs from previously computed overlaps. S1 must have type vertex, and "
            "S2 must have type triangle. B1 and B2 refer to bodies of simplex sets S1 and S2, "
            "respectively. bvh is a bounding volume hierarchy built over S2. dhat is the search "
            "radius around each query simplex in S1. dzero is the numerical error within which 2 "
            "nearest neighbours pairs are considered to have the same distance.");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat