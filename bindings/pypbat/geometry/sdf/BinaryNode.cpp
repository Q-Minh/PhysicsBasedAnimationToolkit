#include "BinaryNode.h"

#include <pbat/geometry/sdf/BinaryNode.h>

namespace pbat::py::geometry::sdf {

void BindBinaryNode(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using Union      = pbat::geometry::sdf::Union<ScalarType>;
    nb::class_<Union>(m, "Union")
        .def(nb::init<>())
        .def(
            "eval",
            [](Union const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the union of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the union of two shapes");

    using Difference = pbat::geometry::sdf::Difference<ScalarType>;
    nb::class_<Difference>(m, "Difference")
        .def(nb::init<>())
        .def(
            "eval",
            [](Difference const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the difference of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the difference of two shapes");

    using Intersection = pbat::geometry::sdf::Intersection<ScalarType>;
    nb::class_<Intersection>(m, "Intersection")
        .def(nb::init<>())
        .def(
            "eval",
            [](Intersection const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the intersection of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the intersection of two shapes");

    using ExclusiveOr = pbat::geometry::sdf::ExclusiveOr<ScalarType>;
    nb::class_<ExclusiveOr>(m, "ExclusiveOr")
        .def(nb::init<>())
        .def(
            "eval",
            [](ExclusiveOr const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the exclusive or of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the exclusive or of two shapes");

    using SmoothUnion = pbat::geometry::sdf::SmoothUnion<ScalarType>;
    nb::class_<SmoothUnion>(m, "SmoothUnion")
        .def(nb::init<>())
        .def(
            "__init__",
            [](SmoothUnion* self, ScalarType k) {
                new (self) SmoothUnion();
                self->k = k;
            },
            nb::arg("k"),
            "Constructor with smoothness factor\n\n"
            "Args:\n"
            "    k (float): Smoothness factor")
        .def(
            "eval",
            [](SmoothUnion const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the smooth union of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the smooth union of two shapes");

    using SmoothDifference = pbat::geometry::sdf::SmoothDifference<ScalarType>;
    nb::class_<SmoothDifference>(m, "SmoothDifference")
        .def(nb::init<>())
        .def(
            "__init__",
            [](SmoothDifference* self, ScalarType k) {
                new (self) SmoothDifference();
                self->k = k;
            },
            nb::arg("k"),
            "Constructor with smoothness factor\n\n"
            "Args:\n"
            "    k (float): Smoothness factor")
        .def(
            "eval",
            [](SmoothDifference const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the smooth difference of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the smooth difference of two shapes");

    using SmoothIntersection = pbat::geometry::sdf::SmoothIntersection<ScalarType>;
    nb::class_<SmoothIntersection>(m, "SmoothIntersection")
        .def(nb::init<>())
        .def(
            "__init__",
            [](SmoothIntersection* self, ScalarType k) {
                new (self) SmoothIntersection();
                self->k = k;
            },
            nb::arg("k"),
            "Constructor with smoothness factor\n\n"
            "Args:\n"
            "    k (float): Smoothness factor")
        .def(
            "eval",
            [](SmoothIntersection const& self, ScalarType sd1, ScalarType sd2) -> ScalarType {
                return self.Eval(sd1, sd2);
            },
            nb::arg("sd1"),
            nb::arg("sd2"),
            "Evaluate the signed distance function of the smooth intersection of two shapes\n\n"
            "Args:\n"
            "    sd1 (float): Signed distance to the first shape\n"
            "    sd2 (float): Signed distance to the second shape\n\n"
            "Returns:\n"
            "    float: Signed distance to the smooth intersection of two shapes");
}

} // namespace pbat::py::geometry::sdf