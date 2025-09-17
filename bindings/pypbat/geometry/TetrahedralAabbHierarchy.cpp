#include "TetrahedralAabbHierarchy.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <pbat/geometry/TetrahedralAabbHierarchy.h>

namespace pbat {
namespace py {
namespace geometry {

void BindTetrahedralAabbHierarchy(nanobind::module_& m)
{
    namespace nb                = nanobind;
    std::string const className = "TetrahedralAabbHierarchy";
    using pbat::geometry::TetrahedralAabbHierarchy;
    nb::class_<TetrahedralAabbHierarchy>(m, className.data())
        .def(
            "__init__",
            [](TetrahedralAabbHierarchy* self,
               Eigen::Ref<MatrixX const> const& V,
               Eigen::Ref<IndexMatrixX const> const& C,
               Index maxPointsInLeaf) {
                new (self) TetrahedralAabbHierarchy(V, C, maxPointsInLeaf);
            },
            nb::arg("V").noconvert(),
            nb::arg("C").noconvert(),
            nb::arg("max_points_in_leaf") = 8)
        .def_prop_ro_static(
            "dims",
            [](nb::object /*self*/) { return TetrahedralAabbHierarchy::kDims; })
        .def(
            "overlapping_primitives",
            &TetrahedralAabbHierarchy::OverlappingPrimitives,
            nb::arg("bvh"),
            nb::arg("reserve") = 1000ULL)
        .def(
            "primitives_containing_points",
            [](TetrahedralAabbHierarchy const& self,
               Eigen::Ref<MatrixX const> const& P,
               bool bParallelize) { return self.PrimitivesContainingPoints(P, bParallelize); },
            nb::arg("P"),
            nb::arg("parallelize") = true)
        .def(
            "nearest_primitives_to_points",
            [](TetrahedralAabbHierarchy const& self,
               Eigen::Ref<MatrixX const> const& P,
               bool bParallelize) { return self.NearestPrimitivesToPoints(P, bParallelize); },
            nb::arg("P"),
            nb::arg("parallelize") = true)
        .def("update", &TetrahedralAabbHierarchy::Update)
        .def_prop_ro(
            "bounding_volumes",
            [](TetrahedralAabbHierarchy const& self) { return self.GetBoundingVolumes(); })
        .def_prop_ro("points", [](TetrahedralAabbHierarchy const& self) { return self.GetV(); })
        .def_prop_ro("primitives", [](TetrahedralAabbHierarchy const& self) {
            return self.GetC();
        });
}

} // namespace geometry
} // namespace py
} // namespace pbat
