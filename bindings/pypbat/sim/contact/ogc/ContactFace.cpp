#include "ContactFace.h"

#include <nanobind/nanobind.h>
#include <pbat/sim/contact/ogc/ContactFace.h>

namespace pbat::py::sim::contact::ogc {

void BindContactFace(nanobind::module_& m)
{
    namespace nb          = nanobind;
    using IndexType       = Index;
    using ContactFaceType = pbat::sim::contact::ogc::ContactFace<IndexType>;

    nb::class_<ContactFaceType>(m, "ContactFace")
        .def(
            nb::init<IndexType, IndexType>(),
            nb::arg("a"),
            nb::arg("eFace"),
            "Construct a contact face.\n\n"
            "Args:\n"
            "    a (int): Face (vertex, half-edge, edge or triangle) index.\n"
            "    eFace (int): Face type indicator: (0 | 1 | 2) -> (triangle | (half-)edge | vertex).")
        .def(
            "is_triangle",
            &ContactFaceType::IsTriangle,
            "Check if the contact face is a triangle.\n\n"
            "Returns:\n"
            "    bool: True if triangle, False otherwise.")
        .def(
            "is_edge",
            &ContactFaceType::IsEdge,
            "Check if the contact face is an edge.\n\n"
            "Returns:\n"
            "    bool: True if edge, False otherwise.")
        .def(
            "is_vertex",
            &ContactFaceType::IsVertex,
            "Check if the contact face is a vertex.\n\n"
            "Returns:\n"
            "    bool: True if vertex, False otherwise.")
        .def(
            "__lt__",
            &ContactFaceType::operator<,
            nb::arg("other"),
            "Less-than operator for ordering contact faces.\n\n"
            "Args:\n"
            "    other (ContactFace): Other contact face.\n\n"
            "Returns:\n"
            "    bool: True if less than other, False otherwise.")
        .def(
            "__eq__",
            &ContactFaceType::operator==,
            nb::arg("other"),
            "Equality operator for contact faces.\n\n"
            "Args:\n"
            "    other (ContactFace): Other contact face.\n\n"
            "Returns:\n"
            "    bool: True if equal to other, False otherwise.")
        .def_rw(
            "a",
            &ContactFaceType::a,
            "(int) Face (vertex, half-edge, edge or triangle) index.")
        .def_rw(
            "eFace",
            &ContactFaceType::eFace,
            "(int) Face type indicator: (0 | 1 | 2) -> (triangle | (half-)edge | vertex).");
}

} // namespace pbat::py::sim::contact::ogc