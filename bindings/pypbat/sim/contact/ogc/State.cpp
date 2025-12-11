#include "State.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/geometry/Device.h>
#include <pbat/sim/contact/ogc/Input.h>
#include <pbat/sim/contact/ogc/Params.h>
#include <pbat/sim/contact/ogc/State.h>

namespace pbat::py::sim::contact::ogc {

void BindState(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using StateType  = pbat::sim::contact::ogc::State<ScalarType, IndexType>;
    using InputType  = pbat::sim::contact::ogc::Input<ScalarType, IndexType>;
    using ParamsType = pbat::sim::contact::ogc::Params<ScalarType>;
    using DeviceType = pbat::geometry::Device;

    nb::class_<StateType>(m, "State")
        .def(nb::init<>(), "Construct an empty OGC state.")
        .def(
            "__init__",
            [](StateType* self,
               DeviceType device,
               InputType const& input,
               ParamsType const& params) { new (self) StateType(device, input, params); },
            nb::arg("device"),
            nb::arg("input"),
            nb::arg("params"),
            "Construct and initialize a new OGC state object.\n\n"
            "Args:\n"
            "    device (pbat.geometry.Device): Device to use for acceleration structures.\n"
            "    input (Input): OGC's input parameters.\n"
            "    params (Params): OGC's parameters.")
        .def(
            "initialize",
            &StateType::Initialize,
            nb::arg("device"),
            nb::arg("input"),
            nb::arg("params"),
            "Initialize OGC state from input.\n\n"
            "Args:\n"
            "    device (pbat.geometry.Device): Device to use for acceleration structures.\n"
            "    input (Input): OGC's input parameters.\n"
            "    params (Params): OGC's parameters.")
        .def(
            "prepare_for_execution",
            &StateType::PrepareForExecution,
            nb::arg("input"),
            nb::arg("params"),
            "Prepare for an OGC algorithm execution.\n\n"
            "Args:\n"
            "    input (Input): OGC's input parameters.\n"
            "    params (Params): OGC's parameters.")
        .def_ro(
            "dynamic_contact_faces_of_vertex",
            &StateType::mDynamicContactFacesOfVertex,
            "(list[list[ContactFace]]) `|# vertices|` per-vertex dynamic contact face sets. The "
            "ContactFace stores vertex, half-edge or triangle.")
        .def_ro(
            "dynamic_contact_vertices_of_triangle",
            &StateType::mDynamicContactVerticesOfTriangle,
            "(list[list[int]]) `|# triangles|` per-triangle dynamic contact vertex sets.")
        .def_ro(
            "dynamic_contact_faces_of_half_edge",
            &StateType::mDynamicContactFacesOfHalfEdge,
            "(list[list[ContactFace]]) `|# half-edges|` per-half-edge dynamic contact face sets. "
            "The ContactFace stores vertex or half-edge.")
        .def_ro(
            "static_contact_faces_of_vertex",
            &StateType::mStaticContactFacesOfVertex,
            "(list[list[ContactFace]]) `|# vertices|` per-vertex static contact face sets. The "
            "ContactFace stores environment vertex, half-edge or triangle.")
        .def_ro(
            "static_contact_vertices_of_triangle",
            &StateType::mStaticContactVerticesOfTriangle,
            "(list[list[int]]) `|# triangles|` per-triangle static contact vertex sets.")
        .def_ro(
            "static_contact_faces_of_half_edge",
            &StateType::mStaticContactFacesOfHalfEdge,
            "(list[list[ContactFace]]) `|# half-edges|` per-half-edge static contact face sets. "
            "The ContactFace stores environment vertex or edge.")
        .def_rw(
            "bv",
            &StateType::bv,
            "(numpy.ndarray) `|# vertices|` array of total vertex displacement bounds.")
        .def_rw(
            "dminv",
            &StateType::dminv,
            "(numpy.ndarray) `|# vertices|` array of vertex local displacement bounds.")
        .def_rw(
            "dminf",
            &StateType::dminf,
            "(numpy.ndarray) `|# facets|` array of face local displacement bounds.")
        .def_rw(
            "dmine",
            &StateType::dmine,
            "(numpy.ndarray) `|# half-edges|` array of half-edge local displacement bounds.");
}

} // namespace pbat::py::sim::contact::ogc