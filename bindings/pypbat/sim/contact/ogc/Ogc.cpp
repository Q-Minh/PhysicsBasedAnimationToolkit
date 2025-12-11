#include "Ogc.h"

#include "ContactFace.h"
#include "Input.h"
#include "Params.h"
#include "State.h"

#include <nanobind/nanobind.h>
#include <pbat/sim/contact/ogc/Input.h>
#include <pbat/sim/contact/ogc/Ogc.h>
#include <pbat/sim/contact/ogc/Params.h>
#include <pbat/sim/contact/ogc/State.h>

namespace pbat::py::sim::contact::ogc {

void Bind(nanobind::module_& m)
{
    BindContactFace(m);
    BindInput(m);
    BindParams(m);
    BindState(m);

    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using InputType  = pbat::sim::contact::ogc::Input<ScalarType, IndexType>;
    using ParamsType = pbat::sim::contact::ogc::Params<ScalarType>;
    using StateType  = pbat::sim::contact::ogc::State<ScalarType, IndexType>;

    m.def(
        "vertex_facet_contact_detection",
        &pbat::sim::contact::ogc::VertexFacetContactDetection<ScalarType, IndexType>,
        nb::arg("input"),
        nb::arg("params"),
        nb::arg("state"),
        "Performs vertex-facet contact detection.\n\n"
        "Detects contact between dynamic vertices and dynamic facets (triangles), as well as\n"
        "between dynamic vertices and static facets, and static vertices and dynamic facets.\n"
        "Updates the state's contact sets and displacement bounds.\n\n"
        "Args:\n"
        "    input (Input): OGC's input parameters.\n"
        "    params (Params): OGC's parameters.\n"
        "    state (State): OGC's state (modified in-place).\n\n"
        "Preconditions:\n"
        "    state.prepare_for_execution() must have been called before this function.");

    m.def(
        "edge_edge_contact_detection",
        &pbat::sim::contact::ogc::EdgeEdgeContactDetection<ScalarType, IndexType>,
        nb::arg("input"),
        nb::arg("params"),
        nb::arg("state"),
        "Performs edge-edge contact detection.\n\n"
        "Detects contact between dynamic edges and dynamic edges, as well as between\n"
        "dynamic edges and static edges. Updates the state's contact sets and displacement "
        "bounds.\n\n"
        "Args:\n"
        "    input (Input): OGC's input parameters.\n"
        "    params (Params): OGC's parameters.\n"
        "    state (State): OGC's state (modified in-place).\n\n"
        "Preconditions:\n"
        "    state.prepare_for_execution() must have been called before this function.");

    m.def(
        "update_displacement_bounds",
        &pbat::sim::contact::ogc::UpdateDisplacementBounds<ScalarType, IndexType>,
        nb::arg("input"),
        nb::arg("params"),
        nb::arg("state"),
        "Updates displacement bounds.\n\n"
        "Computes per-vertex displacement bounds based on local vertex, face, and half-edge\n"
        "displacement bounds computed during contact detection. These bounds are used to\n"
        "determine when to trigger collision detection in the OGC algorithm.\n\n"
        "Args:\n"
        "    input (Input): OGC's input parameters.\n"
        "    params (Params): OGC's parameters.\n"
        "    state (State): OGC's state (modified in-place).");
}

} // namespace pbat::py::sim::contact::ogc