#include "MeshSdfContact.h"

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
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               MeshSdfContactParams const& params) { new (self) MeshSdfContactType(X, F, params); },
            nb::arg("X"),
            nb::arg("F"),
            nb::arg("params"),
            "Construct and initialize a mesh-SDF contact detector.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.\n"
            "    F (numpy.ndarray): 3 x |# triangles| triangle vertex indices.\n"
            "    params (MeshSdfContactParams): Detection parameters.")
        .def(
            "initialize",
            [](MeshSdfContactType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
               MeshSdfContactParams const& params) { self.Initialize(X, F, params); },
            nb::arg("X"),
            nb::arg("F"),
            nb::arg("params"),
            "Initialize internal storage.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.\n"
            "    F (numpy.ndarray): 3 x |# triangles| triangle vertex indices.\n"
            "    params (MeshSdfContactParams): Detection parameters.")
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
        .def_rw(
            "triangle_contact_counts",
            &MeshSdfContactType::mTriangleContactCounts,
            "(numpy.ndarray) |# triangles| contact counts per triangle.")
        .def_rw(
            "triangle_sdf_contacts",
            &MeshSdfContactType::mTriangleSdfContacts,
            "(numpy.ndarray) 3 * n_max_contacts_per_triangle x |# triangles| stacked barycentric "
            "contact coordinates (may contain padding beyond counts).")
        .def_ro(
            "params",
            &MeshSdfContactType::mParams,
            "(MeshSdfContactParams) Parameter set used for detection.")
        .def_prop_ro(
            "flat_contacts",
            [](MeshSdfContactType const& self) {
                // Build a flat 3 x N matrix of valid contacts across all triangles
                Eigen::Index total = self.mTriangleContactCounts.sum();
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> C(3, total);
                Eigen::Index k = 0;
                for (Eigen::Index f = 0; f < self.mTriangleContactCounts.size(); ++f)
                {
                    Eigen::Index nc = self.mTriangleContactCounts(f);
                    for (Eigen::Index c = 0; c < nc; ++c)
                    {
                        C.col(k++) = self.mTriangleSdfContacts.col(f).segment<3>(3 * c);
                    }
                }
                return C;
            },
            "(numpy.ndarray) 3 x |# contacts| matrix of barycentric triangle SDF contact points "
            "(each column stores (b0,b1,b2)).");
}

} // namespace pbat::py::sim::contact
