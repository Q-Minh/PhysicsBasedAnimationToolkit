#include "State.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
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

    nb::class_<StateType>(m, "State")
        .def(nb::init<>(), "Construct an empty OGC state.")
        .def(
            "__init__",
            [](StateType* self,
               pbat::geometry::Device device,
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
        .def(
            "collect_contact_pairs",
            &StateType::CollectContactPairs,
            "Collect and deduplicate contact pairs from thread-local storage into the global "
            "contact pair lists (mXX, mXE, mXF, mEE). Contact pairs are lexicographically sorted.")
        .def(
            "clear_contact_pairs",
            &StateType::ClearContactPairs,
            "Clear all contact pairs (both thread-local and global).")
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
            "(numpy.ndarray) `|# half-edges|` array of half-edge local displacement bounds.")
        .def_ro(
            "point_geometry_prefix",
            &StateType::mPointGeometryPrefix,
            "(list[int]) Prefix sum over points of each geometry type (dynamic, static). "
            "Array of size 3: [0, |# dynamic points|, |# dynamic points| + |# static points|].")
        .def_ro(
            "half_edge_geometry_prefix",
            &StateType::mHalfEdgeGeometryPrefix,
            "(list[int]) Prefix sum over half-edges of each geometry type (dynamic, static). "
            "Array of size 3.")
        .def_ro(
            "triangle_geometry_prefix",
            &StateType::mTriangleGeometryPrefix,
            "(list[int]) Prefix sum over triangles of each geometry type (dynamic, static). "
            "Array of size 3.")
        .def_ro("XX", &StateType::mXX, "(list[tuple[int, int]]) Point-point contact pairs.")
        .def_ro("XE", &StateType::mXE, "(list[tuple[int, int]]) Point-(half-)edge contact pairs.")
        .def_ro("XF", &StateType::mXF, "(list[tuple[int, int]]) Point-triangle contact pairs.")
        .def_ro(
            "EE",
            &StateType::mEE,
            "(list[tuple[int, int]]) (Half-)edge-(half-)edge contact pairs.");
}

} // namespace pbat::py::sim::contact::ogc