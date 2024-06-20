#include "TetrahedralAabbHierarchy.h"

#include <pbat/geometry/TetrahedralAabbHierarchy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace geometry {

void BindTetrahedralAabbHierarchy(pybind11::module& m)
{
    namespace pyb               = pybind11;
    std::string const className = "TetrahedralAabbHierarchy";
    using pbat::geometry::TetrahedralAabbHierarchy;
    pyb::class_<TetrahedralAabbHierarchy>(m, className.data())
        .def(
            pyb::init([](Eigen::Ref<MatrixX const> const& V,
                         Eigen::Ref<IndexMatrixX const> const& T,
                         std::size_t maxPointsInLeaf) {
                return TetrahedralAabbHierarchy(V, T, maxPointsInLeaf);
            }),
            pyb::arg("V"),
            pyb::arg("T"),
            pyb::arg("max_points_in_leaf") = 10ULL)
        .def_property_readonly_static(
            "dims",
            [](pyb::object /*self*/) { return TetrahedralAabbHierarchy::kDims; })
        .def(
            "overlapping_primitives",
            &TetrahedralAabbHierarchy::OverlappingPrimitives,
            pyb::arg("bvh"),
            pyb::arg("reserve") = 1000ULL)
        .def(
            "primitives_containing_points",
            [](TetrahedralAabbHierarchy const& self,
               Eigen::Ref<MatrixX const> const& P,
               bool bParallelize) { return self.PrimitivesContainingPoints(P, bParallelize); },
            pyb::arg("P"),
            pyb::arg("parallelize") = false)
        .def(
            "nearest_primitives_to_points",
            [](TetrahedralAabbHierarchy const& self,
               Eigen::Ref<MatrixX const> const& P,
               bool bParallelize) { return self.NearestPrimitivesToPoints(P, bParallelize); },
            pyb::arg("P"),
            pyb::arg("parallelize") = false)
        .def_property_readonly("bounding_volumes", [](TetrahedralAabbHierarchy const& self) {
            return self.GetBoundingVolumes();
        });
}

} // namespace geometry
} // namespace py
} // namespace pbat
