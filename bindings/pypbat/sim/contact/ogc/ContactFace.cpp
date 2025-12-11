#include "ContactFace.h"

#include <nanobind/nanobind.h>
#include <pbat/sim/contact/ogc/ContactFace.h>

namespace pbat::py::sim::contact::ogc {

void BindContactFace(nanobind::module_& m)
{
    namespace nb                      = nanobind;
    using IndexType                   = Index;
    using ContactFaceType             = pbat::sim::contact::ogc::ContactFace<IndexType>;
    using EVertexFacetClosestFaceType = pbat::sim::contact::ogc::EVertexFacetClosestFaceType;
    using EEdgeEdgeClosestFaceType    = pbat::sim::contact::ogc::EEdgeEdgeClosestFaceType;

    nb::enum_<EVertexFacetClosestFaceType>(m, "VertexFacetClosestFaceType")
        .value("Vertex", EVertexFacetClosestFaceType::Vertex, "Closest face is a vertex")
        .value("Edge", EVertexFacetClosestFaceType::Edge, "Closest face is an edge")
        .value("Facet", EVertexFacetClosestFaceType::Facet, "Closest face is a facet/triangle")
        .export_values();

    nb::enum_<EEdgeEdgeClosestFaceType>(m, "EdgeEdgeClosestFaceType")
        .value("Edge", EEdgeEdgeClosestFaceType::Edge, "Closest face is an edge")
        .value("Vertex", EEdgeEdgeClosestFaceType::Vertex, "Closest face is a vertex")
        .export_values();

    nb::class_<ContactFaceType>(m, "ContactFace")
        .def(
            nb::init<IndexType, IndexType>(),
            nb::arg("a"),
            nb::arg("eFace"),
            "Construct a contact face.\n\n"
            "Args:\n"
            "    a (int): Face (vertex, half-edge, edge or triangle) index.\n"
            "    eFace (int): Face type indicator: (0 | 1 | 2) -> (triangle | (half-)edge | "
            "vertex).")
        .def_prop_ro(
            "vertex_facet_closest_face_type",
            &ContactFaceType::VertexFacetClosestFaceType,
            "(VertexFacetClosestFaceType) The vertex-facet closest face type.")
        .def_prop_ro(
            "edge_edge_closest_face_type",
            &ContactFaceType::EdgeEdgeClosestFaceType,
            "(EdgeEdgeClosestFaceType) The edge-edge closest face type.")
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
        .def_rw("a", &ContactFaceType::a, "(int) Face (vertex, half-edge, edge or triangle) index.")
        .def_rw(
            "eFace",
            &ContactFaceType::eFace,
            "(int) Face type indicator (EVertexFacetClosestFaceType | EEdgeEdgeClosestFaceType).");
}

} // namespace pbat::py::sim::contact::ogc