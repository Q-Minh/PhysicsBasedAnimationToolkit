#include "Debug.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/sim/contact/ogc/ContactFace.h>
#include <pbat/sim/contact/ogc/Debug.h>
#include <pbat/sim/contact/ogc/Input.h>
#include <pbat/sim/contact/ogc/State.h>

namespace pbat::py::sim::contact::ogc {

void BindDebug(nanobind::module_& m)
{
    namespace nb                 = nanobind;
    using ScalarType             = Scalar;
    using IndexType              = Index;
    using VisualDebugContactType = pbat::sim::contact::ogc::VisualDebugContact<ScalarType>;
    using InputType              = pbat::sim::contact::ogc::Input<ScalarType, IndexType>;
    using StateType              = pbat::sim::contact::ogc::State<ScalarType, IndexType>;

    nb::class_<VisualDebugContactType>(m, "VisualDebugContact")
        .def(nb::init<>(), "Construct an empty visual debug contact.")
        .def_rw("xi", &VisualDebugContactType::xi, "(numpy.ndarray) 3D contact point on feature i.")
        .def_rw(
            "xj",
            &VisualDebugContactType::xj,
            "(numpy.ndarray) 3D contact point on feature j.");

    m.def(
        "visual_debug_contacts",
        [](InputType const& input, StateType const& state)
            -> std::tuple<
                std::vector<VisualDebugContactType>,
                std::vector<VisualDebugContactType>,
                std::vector<VisualDebugContactType>,
                std::vector<VisualDebugContactType>> {
            std::vector<VisualDebugContactType> vertexVertexContacts;
            std::vector<VisualDebugContactType> vertexEdgeContacts;
            std::vector<VisualDebugContactType> vertexFacetContacts;
            std::vector<VisualDebugContactType> edgeEdgeContacts;
            pbat::sim::contact::ogc::ToVisualDebugContacts(
                input,
                state,
                vertexVertexContacts,
                vertexEdgeContacts,
                vertexFacetContacts,
                edgeEdgeContacts);
            return std::make_tuple(
                vertexVertexContacts,
                vertexEdgeContacts,
                vertexFacetContacts,
                edgeEdgeContacts);
        },
        nb::arg("input"),
        nb::arg("state"),
        "Converts OGC contact sets to visual debug contacts.\n\n"
        "Args:\n"
        "    input (Input): OGC input data containing geometry information.\n"
        "    state (State): OGC simulation state containing contact sets.\n\n"
        "Returns:\n"
        "    A tuple containing four lists of VisualDebugContact objects:\n"
        "    - (list[VisualDebugContact]): List of vertex-vertex contacts.\n"
        "    - (list[VisualDebugContact]): List of vertex-edge contacts.\n"
        "    - (list[VisualDebugContact]): List of vertex-facet contacts.\n"
        "    - (list[VisualDebugContact]): List of edge-edge contacts.\n");
}

} // namespace pbat::py::sim::contact::ogc
