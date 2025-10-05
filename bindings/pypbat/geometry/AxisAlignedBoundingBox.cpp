#include "AxisAlignedBoundingBox.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/AxisAlignedBoundingBox.h>
#include <string>

namespace pbat {
namespace py {
namespace geometry {

void BindAxisAlignedBoundingBox(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::geometry::AxisAlignedBoundingBox;
    pbat::common::
        ForTypes<AxisAlignedBoundingBox<1>, AxisAlignedBoundingBox<2>, AxisAlignedBoundingBox<3>>(
            [&]<class AabbType>() {
                std::string const className =
                    "AxisAlignedBoundingBox" + std::to_string(AabbType::kDims);
                nb::class_<AabbType>(m, className.data())
                    .def(
                        "__init__",
                        [](AabbType* self,
                           Eigen::Ref<Vector<AabbType::kDims> const> const& L,
                           Eigen::Ref<Vector<AabbType::kDims> const> const& U) {
                            new (self) AabbType(L, U);
                        },
                        nb::arg("min"),
                        nb::arg("max"))
                    .def(
                        "__init__",
                        [](AabbType* self, Eigen::Ref<MatrixX const> const& P) {
                            new (self) AabbType(P);
                        },
                        nb::arg("pts"))
                    .def_prop_ro_static("dims", [](nb::object /*self*/) { return AabbType::kDims; })
                    .def_prop_rw(
                        "min",
                        [](AabbType const& self) { return self.min(); },
                        [](AabbType& self, Eigen::Ref<Vector<AabbType::kDims> const> const& min) {
                            self.min() = min;
                        })
                    .def_prop_rw(
                        "max",
                        [](AabbType const& self) { return self.max(); },
                        [](AabbType& self, Eigen::Ref<Vector<AabbType::kDims> const> const& max) {
                            self.max() = max;
                        })
                    .def_prop_ro("center", [](AabbType const& self) { return self.center(); })
                    .def(
                        "contains",
                        [](AabbType const& self, AabbType const& other) -> bool {
                            return self.contains(other);
                        },
                        nb::arg("aabb"))
                    .def(
                        "contains",
                        [](AabbType const& self, Eigen::Ref<Vector<AabbType::kDims> const> const& P)
                            -> bool { return self.contains(P); })
                    .def(
                        "contained",
                        [](AabbType const& self, Eigen::Ref<MatrixX const> const& P)
                            -> std::vector<Index> { return self.contained(P); })
                    .def(
                        "intersection",
                        [](AabbType const& self, AabbType const& other) -> AabbType {
                            return self.intersection(other);
                        },
                        nb::arg("aabb"))
                    .def(
                        "intersects",
                        [](AabbType const& self, AabbType const& other) -> bool {
                            return self.intersects(other);
                        },
                        nb::arg("aabb"))
                    .def(
                        "merged",
                        [](AabbType const& self, AabbType const& other) -> AabbType {
                            return self.merged(other);
                        },
                        nb::arg("aabb"))
                    .def(
                        "sizes",
                        [](AabbType const& self) -> Vector<AabbType::kDims> {
                            return self.sizes();
                        })
                    .def(
                        "squared_exterior_distance",
                        [](AabbType const& self, Eigen::Ref<Vector<AabbType::kDims> const> const& P)
                            -> Scalar { return self.squaredExteriorDistance(P); },
                        nb::arg("pt"))
                    .def(
                        "translated",
                        [](AabbType const& self, Eigen::Ref<Vector<AabbType::kDims> const> const& t)
                            -> AabbType { return self.translated(t); },
                        nb::arg("t"))
                    .def_prop_ro("volume", &AabbType::volume);
            });
    m.def(
        "mesh_to_aabbs",
        [](Eigen::Ref<MatrixX const> const& X, Eigen::Ref<IndexMatrixX const> const& E) {
            MatrixX B(2 * X.rows(), E.cols());
            common::ForRange<1, 4>([&]<auto kDims>() {
                common::ForRange<1, 5>([&]<auto kElemNodes>() {
                    if (kDims == X.rows() and E.cols() == kElemNodes)
                    {
                        pbat::geometry::MeshToAabbs<kDims, kElemNodes>(X, E, B);
                    }
                });
            });
            return B;
        },
        nb::arg("X"),
        nb::arg("E"),
        "Compute axis-aligned bounding boxes (AABBs) for a mesh.\n\n"
        "Args:\n"
        "    X: `kDims x |# vertices|` matrix of vertex positions\n"
        "    E: `kElemNodes x |# elements|` matrix of element connectivity\n\n"
        "Returns:\n"
        "    `2*kDims x |# elements|` matrix of axis-aligned bounding boxes\n\n");
}

} // namespace geometry
} // namespace py
} // namespace pbat
