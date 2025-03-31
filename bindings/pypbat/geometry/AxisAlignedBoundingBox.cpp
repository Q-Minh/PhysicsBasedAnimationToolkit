#include "AxisAlignedBoundingBox.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/geometry/AxisAlignedBoundingBox.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <string>

namespace pbat {
namespace py {
namespace geometry {

void BindAxisAlignedBoundingBox(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::geometry::AxisAlignedBoundingBox;
    pbat::common::ForValues<1, 2, 3>([&]<auto Dims>() {
        using AabbType              = pbat::geometry::AxisAlignedBoundingBox<Dims>;
        std::string const className = "AxisAlignedBoundingBox" + std::to_string(Dims);
        pyb::class_<AabbType>(m, className.data())
            .def(
                pyb::init([](Eigen::Ref<Vector<Dims> const> const& L,
                             Eigen::Ref<Vector<Dims> const> const& U) { return AabbType(L, U); }),
                pyb::arg("min"),
                pyb::arg("max"))
            .def(
                pyb::init([](Eigen::Ref<MatrixX const> const& P) { return AabbType(P); }),
                pyb::arg("pts"))
            .def_property_readonly_static(
                "dims",
                [](pyb::object /*self*/) { return AabbType::kDims; })
            .def_property(
                "min",
                [](AabbType const& self) { return self.min(); },
                [](AabbType& self, Eigen::Ref<Vector<Dims> const> const& min) { self.min() = min; })
            .def_property(
                "max",
                [](AabbType const& self) { return self.max(); },
                [](AabbType& self, Eigen::Ref<Vector<Dims> const> const& max) { self.max() = max; })
            .def_property_readonly("center", [](AabbType const& self) { return self.center(); })
            .def(
                "contains",
                [](AabbType const& self, AabbType const& other) -> bool {
                    return self.contains(other);
                },
                pyb::arg("aabb"))
            .def(
                "contains",
                [](AabbType const& self, Eigen::Ref<Vector<Dims> const> const& P) -> bool {
                    return self.contains(P);
                })
            .def(
                "contained",
                [](AabbType const& self, Eigen::Ref<MatrixX const> const& P) -> std::vector<Index> {
                    return self.contained(P);
                })
            .def(
                "intersection",
                [](AabbType const& self, AabbType const& other) -> AabbType {
                    return self.intersection(other);
                },
                pyb::arg("aabb"))
            .def(
                "intersects",
                [](AabbType const& self, AabbType const& other) -> bool {
                    return self.intersects(other);
                },
                pyb::arg("aabb"))
            .def(
                "merged",
                [](AabbType const& self, AabbType const& other) -> AabbType {
                    return self.merged(other);
                },
                pyb::arg("aabb"))
            .def("sizes", [](AabbType const& self) -> Vector<Dims> { return self.sizes(); })
            .def(
                "squared_exterior_distance",
                [](AabbType const& self, Eigen::Ref<Vector<Dims> const> const& P) -> Scalar {
                    return self.squaredExteriorDistance(P);
                },
                pyb::arg("pt"))
            .def(
                "translated",
                [](AabbType const& self, Eigen::Ref<Vector<Dims> const> const& t) -> AabbType {
                    return self.translated(t);
                },
                pyb::arg("t"))
            .def_property_readonly("volume", &AabbType::volume);
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
        pyb::arg("X"),
        pyb::arg("E"),
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
