#include "Transform.h"

#include <nanobind/eigen/dense.h>
#include <pbat/geometry/sdf/Transform.h>
#include <pbat/math/linalg/mini/Eigen.h>

namespace pbat::py::geometry::sdf {

void BindTransform(nanobind::module_& m)
{
    namespace nb        = nanobind;
    using ScalarType    = Scalar;
    using TransformType = pbat::geometry::sdf::Transform<ScalarType>;
    using Mat3          = Eigen::Matrix<ScalarType, 3, 3>;
    using Vec3          = Eigen::Vector<ScalarType, 3>;
    using pbat::math::linalg::mini::FromEigen;
    using pbat::math::linalg::mini::ToEigen;
    nb::class_<TransformType>(m, "Transform")
        .def(nb::init<>())
        .def(
            "__init__",
            [](TransformType* self, Mat3 R, Vec3 t) {
                new (self) TransformType(FromEigen(R), FromEigen(t));
            },
            nb::arg("R"),
            nb::arg("t"),
            "Constructor with rotation matrix and translation vector\n\n"
            "Args:\n"
            "    R (numpy.ndarray): `3 x 3` rotation matrix\n"
            "    t (numpy.ndarray): `3 x 1` translation vector")
        .def_prop_rw(
            "R",
            [](TransformType& self) -> Mat3 { return ToEigen(self.R); },
            [](TransformType& self, Mat3 const& R) { self.R = FromEigen(R); },
            "(numpy.ndarray) `3 x 3` rotation matrix")
        .def_prop_rw(
            "t",
            [](TransformType& self) -> Vec3 { return ToEigen(self.t); },
            [](TransformType& self, Vec3 const& t) { self.t = FromEigen(t); },
            "(numpy.ndarray) `3 x 1` translation vector")
        .def(
            "apply",
            [](TransformType& self, Vec3 const& p) -> Vec3 { return ToEigen(self(FromEigen(p))); },
            nb::arg("p"),
            "Apply the transform to a vector\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` vector to transform\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x 1` transformed vector")
        .def(
            "revert",
            [](TransformType& self, Vec3 const& p) -> Vec3 { return ToEigen(self / FromEigen(p)); },
            nb::arg("p"),
            "Revert the transform on a vector\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` vector to revert\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x 1` reverted vector");
}

} // namespace pbat::py::geometry::sdf