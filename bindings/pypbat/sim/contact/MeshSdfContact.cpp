#include "MeshSdfContact.h"

#include <algorithm>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/geometry/sdf/Composite.h>
#include <pbat/io/Archive.h>
#include <pbat/sim/contact/MeshSdfContact.h>

namespace pbat::py::sim::contact {

void BindMeshSdfContact(nanobind::module_& m)
{
    namespace nb               = nanobind;
    using ScalarType           = Scalar;
    using IndexType            = Index;
    using MeshSdfContactParams = pbat::sim::contact::MeshSdfContactParams;
    using MeshSdfContactType   = pbat::sim::contact::MeshSdfContact;
    using Composite            = pbat::geometry::sdf::Composite<ScalarType>;

    nb::class_<MeshSdfContactParams>(m, "MeshSdfContactParams")
        .def(nb::init<>(), "Create default mesh-SDF contact detection parameters.")
        .def_rw(
            "sigmaR",
            &MeshSdfContactParams::sigmaR,
            "(float) Multiple of triangle size for initial trust-region radius Δ0 = sigmaR * |T|.")
        .def_rw(
            "sigmaB",
            &MeshSdfContactParams::sigmaB,
            "(float) Multiple of triangle size for initial Hessian approximation B0 = sigmaB * |T| "
            "I.")
        .def_rw(
            "tauAred",
            &MeshSdfContactParams::tauAred,
            "(float) Proportion of triangle size below which actual reduction is considered small.")
        .def_rw(
            "tauPred",
            &MeshSdfContactParams::tauPred,
            "(float) Proportion of triangle size below which predicted reduction is considered "
            "small.")
        .def_rw(
            "n_max_opt_iters",
            &MeshSdfContactParams::nMaxOptimizationIterations,
            "(int) Maximum trust-region iterations.")
        .def_rw(
            "coord_zero",
            &MeshSdfContactParams::coordZero,
            "(float) Barycentric coordinate tolerance for duplicate contact filtering.")
        .def_rw(
            "hfd",
            &MeshSdfContactParams::hfd,
            "(float) Finite difference step size for SDF gradient estimation.")
        .def_rw(
            "r",
            &MeshSdfContactParams::r,
            "(float) Proximity threshold for a penetrating SDF value (distance <= r).")
        .def(
            "with_initialization_strategy",
            &MeshSdfContactParams::WithInitializationStrategy,
            nb::arg("sigmaR"),
            nb::arg("sigmaB"),
            nb::rv_policy::reference_internal,
            "Set the contact initialization strategy.\n\n"
            "Args:\n"
            "    sigmaR (float): Multiple of triangle size for initial trust-region radius Δ0 = "
            "sigmaR * |T|.\n"
            "    sigmaB (float): Multiple of triangle size for initial Hessian approximation B0 = "
            "sigmaB * |T| I.\n"
            "Returns:\n"
            "    MeshSdfContactParams: Reference to this parameter set.")
        .def(
            "with_termination_criteria",
            &MeshSdfContactParams::WithTerminationCriteria,
            nb::arg("tauAred"),
            nb::arg("tauPred"),
            nb::arg("n_max_opt_iters_per_triangle"),
            nb::rv_policy::reference_internal,
            "Set the trust-region optimization termination criteria.\n\n"
            "Args:\n"
            "    tauAred (float): Proportion of triangle size below which actual reduction is "
            "considered small.\n"
            "    tauPred (float): Proportion of triangle size below which predicted reduction is "
            "considered small.\n"
            "    n_max_opt_iters_per_triangle (int): Maximum trust-region iterations per "
            "triangle.\n"
            "Returns:\n"
            "    MeshSdfContactParams: Reference to this parameter set.")
        .def(
            "with_numerical_parameters",
            &MeshSdfContactParams::WithNumericalParameters,
            nb::arg("coord_zero"),
            nb::arg("hfd"),
            nb::arg("r"),
            nb::rv_policy::reference_internal,
            "Set numerical parameters.\n\n"
            "Args:\n"
            "    coord_zero (float): Tolerance for comparing if 2 contact points are to be "
            "considered duplicates.\n"
            "    hfd (float): Finite difference step size used for SDF gradient estimation.\n"
            "    r (float): Proximity threshold for a penetrating SDF value (distance <= r).\n"
            "Returns:\n"
            "    MeshSdfContactParams: Reference to this parameter set.")
        .def(
            "construct",
            &MeshSdfContactParams::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Validate and finalize parameter set.\n\n"
            "Args:\n"
            "    validate (bool): If true, validate parameters and throw on invalid values.\n\n"
            "Returns:\n"
            "    MeshSdfContactParams: Reference to this parameter set.")
        .def(
            "serialize",
            &MeshSdfContactParams::Serialize,
            nb::arg("archive"),
            "Serialize parameters to an archive.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Target archive.")
        .def(
            "deserialize",
            &MeshSdfContactParams::Deserialize,
            nb::arg("archive"),
            "Deserialize parameters from an archive.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Source archive.");

    nb::class_<MeshSdfContactType>(m, "MeshSdfContact")
        .def(
            nb::init<>(),
            "Construct an empty mesh-SDF contact detector. "
            "Call initialize() before detection.")
        .def(
            "__init__",
            [](MeshSdfContactType* self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               MeshSdfContactParams const& params) { new (self) MeshSdfContactType(V, F, params); },
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("params") = MeshSdfContactParams(),
            "Construct and initialize a mesh-SDF contact detector.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `|# vertices| x 1` vertices.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices.\n"
            "    params (MeshSdfContactParams): Contact detection parameters.\n")
        .def(
            "initialize",
            [](MeshSdfContactType& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F) {
                self.Initialize(V, F);
            },
            nb::arg("V"),
            nb::arg("F"),
            "Initialize internal storage.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `|# vertices| x 1` vertices.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices.\n")
        .def(
            "prepare_iteration",
            &MeshSdfContactType::PrepareIteration,
            "Prepare for a new iteration of contact detection.")
        .def(
            "triangle_sdf_contact_detection",
            [](MeshSdfContactType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
               nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV,
               Composite const& sdf) {
                self.TriangleSdfContactDetection(X, V, F, GHEF, EHE, GXV, sdf);
            },
            nb::arg("X"),
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("GHEF"),
            nb::arg("EHE"),
            nb::arg("GXV"),
            nb::arg("sdf"),
            "Run triangle-SDF contact detection.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.\n"
            "    V (numpy.ndarray): |# vertices| x 1 vertex indices.\n"
            "    F (numpy.ndarray): 3 x |# triangles| triangle vertex indices.\n"
            "    GHEF (numpy.ndarray): 2 x |# half-edges| half-edge to triangle adjacency.\n"
            "    EHE (numpy.ndarray): 2 x |# edges| edge to half-edge adjacency.\n"
            "    GXV (numpy.ndarray): |# points| x 1 point to vertex adjacency.\n"
            "    sdf (pbat.geometry.sdf.Composite): Environment SDF composite.")
        .def(
            "serialize",
            &MeshSdfContactType::Serialize,
            nb::arg("archive"),
            "Serialize contact detector state.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Target archive.")
        .def(
            "deserialize",
            &MeshSdfContactType::Deserialize,
            nb::arg("archive"),
            "Deserialize contact detector state.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Source archive.")
        .def_ro(
            "triangle_contact_mask",
            &MeshSdfContactType::mTriangleContactMask,
            "(numpy.ndarray) `|# triangles|` per-triangle contact mask.")
        .def_ro(
            "triangle_contact_points",
            &MeshSdfContactType::mTriangleContactPoints,
            "(numpy.ndarray) `2 x |# triangles|` per-triangle contact points `u,v` in each column.")
        .def_ro(
            "half_edge_contact_mask",
            &MeshSdfContactType::mHalfEdgeContactMask,
            "(numpy.ndarray) `|# half-edges|` per-half-edge contact mask.")
        .def_ro(
            "half_edge_contact_points",
            &MeshSdfContactType::mHalfEdgeContactPoints,
            "(numpy.ndarray) `|# half-edges| x 1` per-half-edge contact points `u` in each "
            "coefficient.")
        .def_ro(
            "vertex_contact_mask",
            &MeshSdfContactType::mVertexContactMask,
            "(numpy.ndarray) `|# vertices|` per-vertex contact mask.")
        .def_ro(
            "vertex_displacement_bounds",
            &MeshSdfContactType::mVertexDisplacementBounds,
            "(numpy.ndarray) `|# vertices|` per-vertex displacement bounds.")
        .def_ro(
            "params",
            &MeshSdfContactType::mParams,
            "(MeshSdfContactParams) Parameter set used for detection.");
}

} // namespace pbat::py::sim::contact
