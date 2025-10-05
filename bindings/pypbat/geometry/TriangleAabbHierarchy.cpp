#include "TriangleAabbHierarchy.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/TriangleAabbHierarchy.h>
#include <string>

namespace pbat {
namespace py {
namespace geometry {

void BindTriangleAabbHierarchy(nanobind::module_& m)
{
    namespace nb = nanobind;
    pbat::common::ForTypes<
        pbat::geometry::TriangleAabbHierarchy2D,
        pbat::geometry::TriangleAabbHierarchy3D>([&]<class BvhType>() {
        std::string const className = []() {
            auto constexpr kDims = BvhType::kDims;
            if constexpr (kDims == 2)
                return "TriangleAabbHierarchy2D";
            else if constexpr (kDims == 3)
                return "TriangleAabbHierarchy3D";
            else
                static_assert(kDims == 2 || kDims == 3, "Only 2D and 3D BVHs are supported.");
        }();
        nb::class_<BvhType>(m, className.data())
            .def(
                "__init__",
                [](BvhType* self,
                   Eigen::Ref<MatrixX const> const& V,
                   Eigen::Ref<IndexMatrixX const> const& C,
                   Index maxPointsInLeaf) { new (self) BvhType(V, C, maxPointsInLeaf); },
                nb::arg("V").noconvert(),
                nb::arg("C").noconvert(),
                nb::arg("max_points_in_leaf") = 8)
            .def_prop_ro_static("dims", [](nb::object /*self*/) { return BvhType::kDims; })
            .def(
                "overlapping_primitives",
                &BvhType::OverlappingPrimitives,
                nb::arg("bvh"),
                nb::arg("reserve") = 1000ULL)
            .def(
                "primitives_containing_points",
                [](BvhType const& self, Eigen::Ref<MatrixX const> const& P, bool bParallelize) {
                    return self.PrimitivesContainingPoints(P, bParallelize);
                },
                nb::arg("P"),
                nb::arg("parallelize") = true)
            .def(
                "nearest_primitives_to_points",
                [](BvhType const& self, Eigen::Ref<MatrixX const> const& P, bool bParallelize) {
                    return self.NearestPrimitivesToPoints(P, bParallelize);
                },
                nb::arg("P"),
                nb::arg("parallelize") = true)
            .def("update", &BvhType::Update)
            .def_prop_ro("bounding_volumes", [](BvhType const& self) {
                return self.GetBoundingVolumes();
            });
    });
}

} // namespace geometry
} // namespace py
} // namespace pbat
