#include "TriangleAabbHierarchy.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/TriangleAabbHierarchy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace pbat {
namespace py {
namespace geometry {

void BindTriangleAabbHierarchy(pybind11::module& m)
{
    namespace pyb = pybind11;
    pbat::common::ForValues<2, 3>([&]<auto Dims>() {
        auto constexpr kDims        = Dims;
        std::string const className = "TriangleAabbHierarchy" + std::to_string(kDims) + "D";
        using BvhType               = pbat::geometry::TriangleAabbHierarchy<kDims>;
        pyb::class_<BvhType>(m, className.data())
            .def(
                pyb::init(
                    [](Eigen::Ref<MatrixX const> const& V,
                       Eigen::Ref<IndexMatrixX const> const& C,
                       std::size_t maxPointsInLeaf) { return BvhType(V, C, maxPointsInLeaf); }),
                pyb::arg("V").noconvert(),
                pyb::arg("C").noconvert(),
                pyb::arg("max_points_in_leaf") = 10ULL)
            .def_property_readonly_static(
                "dims",
                [](pyb::object /*self*/) { return BvhType::kDims; })
            .def(
                "overlapping_primitives",
                &BvhType::OverlappingPrimitives,
                pyb::arg("bvh"),
                pyb::arg("reserve") = 1000ULL)
            .def(
                "primitives_containing_points",
                [](BvhType const& self, Eigen::Ref<MatrixX const> const& P, bool bParallelize) {
                    return self.PrimitivesContainingPoints(P, bParallelize);
                },
                pyb::arg("P"),
                pyb::arg("parallelize") = false)
            .def(
                "nearest_primitives_to_points",
                [](BvhType const& self, Eigen::Ref<MatrixX const> const& P, bool bParallelize) {
                    return self.NearestPrimitivesToPoints(P, bParallelize);
                },
                pyb::arg("P"),
                pyb::arg("parallelize") = false)
            .def_property_readonly("bounding_volumes", [](BvhType const& self) {
                return self.GetBoundingVolumes();
            });
    });
}

} // namespace geometry
} // namespace py
} // namespace pbat
