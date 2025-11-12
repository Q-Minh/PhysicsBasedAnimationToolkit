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
            "n_max_contacts_per_triangle",
            &MeshSdfContactParams::nMaxContactsPerTriangle,
            "(int) Maximum number of stored contacts per triangle.")
        .def_rw(
            "n_max_opt_iters_per_triangle",
            &MeshSdfContactParams::nMaxOptimizationIterationsPerTriangle,
            "(int) Maximum trust-region iterations per triangle.")
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
            "with_contact_storage_limits",
            &MeshSdfContactParams::WithContactStorageLimits,
            nb::arg("n_max_contacts_per_triangle"),
            nb::rv_policy::reference_internal,
            "Set the contact storage limits.\n\n"
            "Args:\n"
            "    n_max_contacts_per_triangle (int): Maximum number of stored contacts per "
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
            nb::arg("params"),
            "Construct and initialize a mesh-SDF contact detector.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `|# vertices| x 1` vertices.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices.\n"
            "    params (MeshSdfContactParams): Detection parameters.")
        .def(
            "initialize",
            [](MeshSdfContactType& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               MeshSdfContactParams const& params) { self.Initialize(V, F, params); },
            nb::arg("V"),
            nb::arg("F"),
            nb::arg("params"),
            "Initialize internal storage.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `|# vertices| x 1` vertices.\n"
            "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices.\n"
            "    params (MeshSdfContactParams): Detection parameters.")
        .def(
            "prepare_iteration",
            &MeshSdfContactType::PrepareIteration,
            "Prepare for a new iteration of contact detection.")
        .def(
            "triangle_sdf_contact_detection",
            [](MeshSdfContactType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               Composite const& sdf) { self.TriangleSdfContactDetection(X, F, sdf); },
            nb::arg("X"),
            nb::arg("F"),
            nb::arg("sdf"),
            "Run triangle-SDF contact detection.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.\n"
            "    F (numpy.ndarray): 3 x |# triangles| triangle vertex indices.\n"
            "    sdf (pbat.geometry.sdf.Composite): Environment SDF composite.")
        .def(
            "deduplicate_contact_set",
            [&](MeshSdfContactType& self,
                nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
                nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
                nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV) {
                self.DeduplicateContactSet(F, GHEF, GXV);
            },
            nb::arg("F"),
            nb::arg("GHEF"),
            nb::arg("GXV"),
            "De-duplicate all contacts after detection.\n\n"
            "Args:\n"
            "    F (numpy.ndarray): 3 x |# triangles| triangle vertex indices.\n"
            "    GHEF (numpy.ndarray): 2 x |# half edges| half-edge to face adjacency.\n"
            "    GXV (numpy.ndarray): |# points| point to vertex adjacency.")
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
        .def_prop_ro(
            "triangle_contact_counts",
            [](MeshSdfContactType const& self) {
                Eigen::Vector<int, Eigen::Dynamic> counts(self.mTriangleContactPoints.size());
                std::transform(
                    self.mTriangleContactPoints.begin(),
                    self.mTriangleContactPoints.end(),
                    counts.data(),
                    [](auto const& pts) { return static_cast<int>(pts.size()); });
                return counts;
            },
            "(numpy.ndarray) |# triangles| array of contact counts per triangle.")
        .def_prop_ro(
            "triangle_contacts",
            [](MeshSdfContactType const& self) {
                std::size_t const nContacts = std::accumulate(
                    self.mTriangleContactPoints.begin(),
                    self.mTriangleContactPoints.end(),
                    std::size_t(0),
                    [](std::size_t sum, auto const& pts) { return sum + pts.size(); });
                Eigen::Matrix<ScalarType, 2, Eigen::Dynamic> UV(2, nContacts);
                std::size_t k = 0;
                for (auto const& pts : self.mTriangleContactPoints)
                    for (auto const& uvw : pts)
                        UV.col(k++) = uvw;
                return UV;
            },
            "(numpy.ndarray) 2 x |# triangle-sdf contacts| stacked UV barycentric "
            "contact coordinates (may contain padding beyond counts).")
        .def_prop_ro(
            "half_edge_contact_counts",
            [](MeshSdfContactType const& self) {
                Eigen::Vector<int, Eigen::Dynamic> counts(self.mHalfEdgeContactPoints.size());
                std::transform(
                    self.mHalfEdgeContactPoints.begin(),
                    self.mHalfEdgeContactPoints.end(),
                    counts.data(),
                    [](auto const& pts) { return static_cast<int>(pts.size()); });
                return counts;
            },
            "(numpy.ndarray) |# half-edges| array of contact counts per half-edge.")
        .def_prop_ro(
            "half_edge_contacts",
            [](MeshSdfContactType const& self) {
                std::size_t const nContacts = std::accumulate(
                    self.mHalfEdgeContactPoints.begin(),
                    self.mHalfEdgeContactPoints.end(),
                    std::size_t(0),
                    [](std::size_t sum, auto const& pts) { return sum + pts.size(); });
                Eigen::Vector<ScalarType, Eigen::Dynamic> U(nContacts);
                std::size_t k = 0;
                for (auto const& pts : self.mHalfEdgeContactPoints)
                    for (typename MeshSdfContactType::ScalarType t : pts)
                        U(k++) = t;
                return U;
            },
            "(numpy.ndarray) |# half-edge contacts| stacked t barycentric contact "
            "coordinates.")
        .def_prop_ro(
            "vertex_contacts",
            [](MeshSdfContactType const& self) {
                Eigen::Vector<IndexType, Eigen::Dynamic> vc;
                vc.resize(self.mVertexContactPoints.array().count());
                std::size_t k        = 0;
                auto const nVertices = static_cast<IndexType>(self.mVertexContactPoints.size());
                for (IndexType v = 0; v < nVertices; ++v)
                    if (self.mVertexContactPoints[v])
                        vc(k++) = v;
                return vc;
            },
            "(numpy.ndarray[int]) `|# vertex contacts|` array of vertex contact vertex indices.")
        .def_ro(
            "params",
            &MeshSdfContactType::mParams,
            "(MeshSdfContactParams) Parameter set used for detection.");
}

} // namespace pbat::py::sim::contact
