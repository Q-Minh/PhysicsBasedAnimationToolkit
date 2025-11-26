#include "OffsetGeometryContact.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <pbat/geometry/Device.h>
#include <pbat/sim/contact/OffsetGeometryContact.h>

namespace pbat::py::sim::contact {

void BindOffsetGeometryContact(nanobind::module_& m)
{
    namespace nb                = nanobind;
    using IndexType             = Index;
    using ScalarType            = Scalar;
    using OgcParams             = pbat::sim::contact::OgcParams;
    using OffsetGeometryContact = pbat::sim::contact::OffsetGeometryContact;

    // NOTE:
    // We should probably move these enums somewhere more general/reusable at some point,
    // when we have more algorithms that rely on Embree under the hood (or an in-house BVH
    // of BVH implementation).
    nb::enum_<OgcParams::EBuildQuality>(m, "EBuildQuality")
        .value("Low", OgcParams::EBuildQuality::Low)
        .value("Medium", OgcParams::EBuildQuality::Medium)
        .value("High", OgcParams::EBuildQuality::High)
        .export_values();
    nb::enum_<OgcParams::ESceneFeatures>(m, "ESceneFeatures")
        .value("None", OgcParams::ESceneFeatures::None)
        .value("Dynamic", OgcParams::ESceneFeatures::Dynamic)
        .value("Compact", OgcParams::ESceneFeatures::Compact)
        .value("Robust", OgcParams::ESceneFeatures::Robust)
        .export_values();

    nb::class_<OgcParams>(m, "OgcParams")
        .def(nb::init<>())
        .def(
            "with_radii",
            &OgcParams::WithRadii,
            nb::arg("r"),
            nb::arg("rq"),
            nb::rv_policy::reference_internal,
            "Set contact and query radii.\n\n"
            "Args:\n"
            "    r (float): Contact radius.\n"
            "    rq (float): Query inflation radius.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def(
            "with_scene_features",
            &OgcParams::WithSceneFeatures,
            nb::arg("features"),
            nb::rv_policy::reference_internal,
            "Set scene features.\n\n"
            "Args:\n"
            "    features (pbat.sim.contact.ESceneFeatures): Scene features.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def(
            "with_build_quality",
            &OgcParams::WithBuildQuality,
            nb::arg("scene"),
            nb::arg("mesh"),
            nb::rv_policy::reference_internal,
            "Set both scene and mesh BVH build quality.\n\n"
            "Args:\n"
            "    scene (pbat.sim.contact.EBuildQuality): Scene BVH build quality.\n"
            "    mesh (pbat.sim.contact.EBuildQuality): Mesh BVH build quality.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def(
            "with_max_contact_estimates",
            &OgcParams::WithMaxContactEstimates,
            nb::arg("nvf"),
            nb::arg("nfv"),
            nb::arg("nef"),
            nb::rv_policy::reference_internal,
            "Set contact count estimates used for pre-allocation.\n\n"
            "Args:\n"
            "    nvf (int): Max vertex-face contacts estimate.\n"
            "    nfv (int): Max face-vertex contacts estimate.\n"
            "    nef (int): Max edge-face contacts estimate.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def(
            "with_displacement_bound_config",
            &OgcParams::WithDisplacementBoundConfig,
            nb::arg("gammap"),
            nb::arg("gammae"),
            nb::rv_policy::reference_internal,
            "Set displacement bound relaxation parameter. Must satisfy 0 < gammap < 0.5.\n\n"
            "Args:\n"
            "    gammap (float): Relaxation parameter for vertex displacement bound.\n"
            "    gammae (float): Proportion of bounds-violating vertices to trigger collision "
            "detection.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def(
            "construct",
            &OgcParams::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Validate and finalize the parameters.\n\n"
            "Args:\n"
            "    validate (bool): Throw if invalid.\n"
            "Returns:\n"
            "    self (OgcParams): Reference to this.")
        .def_rw("r", &OgcParams::r, "(float) Contact radius.")
        .def_rw("rq", &OgcParams::rq, "(float) Contact query radius.")
        .def_rw(
            "scene_features",
            &OgcParams::eSceneFeatures,
            "(ESceneFeatures) Scene features for BVH construction.")
        .def_rw(
            "scene_bvh_quality",
            &OgcParams::eSceneBvhQuality,
            "(EBuildQuality) Scene BVH build quality.")
        .def_rw(
            "mesh_bvh_quality",
            &OgcParams::eMeshBvhQuality,
            "(EBuildQuality) Mesh BVH build quality.")
        .def_rw(
            "max_vertex_face_contacts_estimate",
            &OgcParams::nMaxVertexFaceContactsEstimate,
            "(int) Max vertex-face contacts estimate.")
        .def_rw(
            "max_face_vertex_contacts_estimate",
            &OgcParams::nMaxFaceVertexContactsEstimate,
            "(int) Max face-vertex contacts estimate.")
        .def_rw(
            "max_edge_face_contacts_estimate",
            &OgcParams::nMaxEdgeFaceContactsEstimate,
            "(int) Max edge-face contacts estimate.")
        .def_rw(
            "gammap",
            &OgcParams::gammap,
            "(float) Relaxation parameter for vertex displacement bound.")
        .def_rw(
            "gammae",
            &OgcParams::gammae,
            "(float) Proportion of bounds-violating vertices to trigger collision detection.");

    nb::class_<OffsetGeometryContact::ContactFace>(m, "ContactFace")
        .def(nb::init<IndexType, IndexType>(), nb::arg("a"), nb::arg("eFace"))
        .def_prop_ro(
            "is_triangle",
            &OffsetGeometryContact::ContactFace::IsTriangle,
            "Check if the contact face is a triangle.")
        .def_prop_ro(
            "is_edge",
            &OffsetGeometryContact::ContactFace::IsEdge,
            "Check if the contact face is an edge.")
        .def_prop_ro(
            "is_vertex",
            &OffsetGeometryContact::ContactFace::IsVertex,
            "Check if the contact face is a vertex.")
        .def(
            "__lt__",
            &OffsetGeometryContact::ContactFace::operator<,
            nb::arg("other"),
            "Less-than operator for ordering contact faces.")
        .def(
            "__eq__",
            &OffsetGeometryContact::ContactFace::operator==,
            nb::arg("other"),
            "Equality operator for contact faces.")
        .def_ro("a", &OffsetGeometryContact::ContactFace::a)
        .def_ro("eFace", &OffsetGeometryContact::ContactFace::eFace);

    nb::class_<OffsetGeometryContact>(m, "OffsetGeometryContact")
        .def(
            nb::init<pbat::geometry::Device>(),
            nb::arg("device"),
            "Construct an offset geometry contact engine with the given device.\n\n"
            "Args:\n"
            "    device (pbat.geometry.Device): Spatial acceleration device.\n")
        .def(
            "__init__",
            [](OffsetGeometryContact* self,
               pbat::geometry::Device device,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
               OgcParams const& params) {
                new (self) OffsetGeometryContact(device, X, V, F, E, VP, FP, EP, params);
            },
            nb::arg("device"),
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("E"),
            nb::arg("VP"),
            nb::arg("FP"),
            nb::arg("EP"),
            nb::arg("params"),
            "Construct and build the OGC engine from multi-mesh.\n\n"
            "Args:\n"
            "    device (pbat.geometry.Device): Spatial acceleration device.\n"
            "    X (numpy.ndarray): `3 x |# points|` point positions.\n"
            "    V (numpy.ndarray): `|# vertices| x 1` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices into X.\n"
            "    E (numpy.ndarray): `2 x |# edges|` undirected edges into V.\n"
            "    VP (numpy.ndarray): `|# components+1|` vertex prefix.\n"
            "    FP (numpy.ndarray): `|# components+1|` face prefix.\n"
            "    EP (numpy.ndarray): `|# components+1|` edge prefix.\n"
            "    params (OgcParams): OGC parameters.\n")
        .def(
            "initialize",
            [](OffsetGeometryContact& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP) {
                // Need a device to re-initialize; reuse a lightweight default
                pbat::geometry::Device dev{pbat::geometry::DeviceConfig{}};
                self.Initialize(dev, X, V, F, E, VP, FP, EP);
            },
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("E"),
            nb::arg("VP"),
            nb::arg("FP"),
            nb::arg("EP"),
            "Initialize OGC acceleration structures.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` point positions.\n"
            "    V (numpy.ndarray): `|# vertices|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices into X.\n"
            "    E (numpy.ndarray): `2 x |# edges|` undirected edges into V.\n"
            "    VP (numpy.ndarray): `|# components+1|` vertex prefix.\n"
            "    FP (numpy.ndarray): `|# components+1|` face prefix.\n"
            "    EP (numpy.ndarray): `|# components+1|` edge prefix.\n")
        .def(
            "prepare_iteration",
            [](OffsetGeometryContact& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP) {
                self.PrepareIteration(X, V, F, E, VP, FP, EP);
            },
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("E"),
            nb::arg("VP"),
            nb::arg("FP"),
            nb::arg("EP"),
            "Prepare contact detection for this iteration.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` point positions.\n"
            "    V (numpy.ndarray): `|# vertices|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices into X.\n"
            "    E (numpy.ndarray): `2 x |# edges|` undirected edges into V.\n"
            "    VP (numpy.ndarray): `|# components+1|` vertex prefix.\n"
            "    FP (numpy.ndarray): `|# components+1|` face prefix.\n"
            "    EP (numpy.ndarray): `|# components+1|` edge prefix.\n")
        .def(
            "vertex_facet_contact_detection",
            [](OffsetGeometryContact& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF) {
                self.VertexFacetContactDetection(X, V, F, VP, FP, GVHEp, GVHEadj, GHEF);
            },
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("VP"),
            nb::arg("FP"),
            nb::arg("GVHEp"),
            nb::arg("GVHEadj"),
            nb::arg("GHEF"),
            "Compute vertex-facet contact sets.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` positions.\n"
            "    V (numpy.ndarray): `|# vertices|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices into X.\n"
            "    VP (numpy.ndarray): `|# components+1|` vertex prefix.\n"
            "    FP (numpy.ndarray): `|# components+1|` face prefix.\n"
            "    GVHEp (numpy.ndarray): `|# points + 1|` point-to-half-edge adjacency prefix.\n"
            "    GVHEadj (numpy.ndarray): `|# half edges|` point-to-half-edge adjacency.\n"
            "    GHEF (numpy.ndarray): `2 x |# half edges|` half-edge to face adjacency.\n")
        .def(
            "edge_edge_contact_detection",
            [](OffsetGeometryContact& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE) {
                self.EdgeEdgeContactDetection(X, V, F, E, VP, EP, GVHEp, GVHEadj, EHE);
            },
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("E"),
            nb::arg("VP"),
            nb::arg("EP"),
            nb::arg("GVHEp"),
            nb::arg("GVHEadj"),
            nb::arg("EHE"),
            "Compute edge-edge contact sets.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` positions.\n"
            "    V (numpy.ndarray): `|# vertices|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices into X.\n"
            "    E (numpy.ndarray): `2 x |# edges|` undirected edges.\n"
            "    VP (numpy.ndarray): `|# components+1|` vertex prefix.\n"
            "    EP (numpy.ndarray): `|# components+1|` edge prefix.\n"
            "    GVHEp (numpy.ndarray): `|# points + 1|` vertex-to-half-edge adjacency prefix.\n"
            "    GVHEadj (numpy.ndarray): `|# half edges|` vertex-to-half-edge adjacency.\n"
            "    EHE (numpy.ndarray): `2 x |# edges|` edge-to-half-edge adjacency.\n")
        .def(
            "compute_displacement_bounds",
            [](OffsetGeometryContact& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj) {
                self.ComputeDisplacementBounds(V, F, GVHEp, GVHEadj);
            },
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("GVHEp"),
            nb::arg("GVHEadj"),
            "Compute per-vertex displacement bounds.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `|# vertices|` vertex indices into X.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices.\n"
            "    GVHEp (numpy.ndarray): `|# points + 1|` vertex-to-half-edge adjacency prefix.\n"
            "    GVHEadj (numpy.ndarray): `|# half edges|` vertex-to-half-edge adjacency.\n")
        .def_prop_ro(
            "bounds",
            &OffsetGeometryContact::Bounds,
            "Get the scene axis-aligned bounding box.\n\n"
            "Returns:\n"
            "    Tuple[numpy.ndarray, numpy.ndarray]: (lower, upper), each 3-vector.\n")
        .def_ro("FOGC", &OffsetGeometryContact::FOGC, "`|# vertices|` per-vertex contact face sets")
        .def_ro(
            "VOGC",
            &OffsetGeometryContact::VOGC,
            "`|# triangles|` per-triangle contact vertex sets")
        .def_ro(
            "EOGC",
            &OffsetGeometryContact::EOGC,
            "`|# half-edges|` per-edge contact vertex sets")
        .def_ro(
            "bv",
            &OffsetGeometryContact::bv,
            "`|# vertices|` per-vertex total displacement bounds")
        .def_ro(
            "dminv",
            &OffsetGeometryContact::dminv,
            "`|# vertices|` per-vertex local displacement bounds")
        .def_ro(
            "dminf",
            &OffsetGeometryContact::dminf,
            "`|# triangles|` per-triangle local displacement bounds")
        .def_ro(
            "dmine",
            &OffsetGeometryContact::dmine,
            "`|# half-edges|` per-half-edge local displacement bounds")
        .def_rw(
            "params",
            &OffsetGeometryContact::params,
            "OgcParams used to configure this OGC engine.");
}

} // namespace pbat::py::sim::contact
