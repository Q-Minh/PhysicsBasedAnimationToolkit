#include "MeshDynamics.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <pbat/geometry/Device.h>
#include <pbat/sim/contact/MeshDynamics.h>
#include <pbat/sim/contact/MultiMesh.h>

namespace pbat::py::sim::contact {

void BindMeshDynamics(nanobind::module_& m)
{
    namespace nb                 = nanobind;
    using ScalarType             = Scalar;
    using IndexType              = Index;
    using MeshDynamicsType       = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>;
    using MeshDynamicsParamsType = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>::Params;
    using MultiMeshType          = pbat::sim::contact::MultiMesh<IndexType>;

    nb::class_<MeshDynamicsParamsType>(m, "MeshDynamicsParams")
        .def(nb::init<>(), "Construct default MeshDynamicsParams.")
        .def(
            "with_ogc_params",
            &MeshDynamicsParamsType::WithOgcParams,
            nb::arg("params"),
            nb::rv_policy::reference_internal,
            "Set the OGC parameters.\n\n"
            "Args:\n"
            "    params (ogc.Params): OGC parameters.\n")
        .def(
            "with_frictional_contact",
            &MeshDynamicsParamsType::WithFrictionalContact,
            nb::arg("mu"),
            nb::arg("epsv"),
            nb::rv_policy::reference_internal,
            "Set the frictional contact parameters.\n\n"
            "Args:\n"
            "    mu (float): Friction coefficient.\n"
            "    epsv (float): Relative velocity threshold for static to dynamic friction's smooth "
            "transition.\n")
        .def(
            "with_normal_contact",
            &MeshDynamicsParamsType::WithNormalContact,
            nb::arg("kc"),
            nb::rv_policy::reference_internal,
            "Set the normal contact parameters.\n\n"
            "Args:\n"
            "    kc (float): Contact stiffness parameter, `kc > 0`.\n")
        .def(
            "with_query_radius_initialization",
            &MeshDynamicsParamsType::WithQueryRadiusInitialization,
            nb::arg("rqstart"),
            nb::arg("betarq"),
            nb::rv_policy::reference_internal,
            "Set the query radius initialization parameters.\n\n"
            "Args:\n"
            "    rqstart (float): Base query radius (larger than contact radius `r`) on which we "
            "add a linear function of inertial target distance to initialize the actual query "
            "radius.\n"
            "    betarq (float): Slope of the linear function of inertial target distance to add "
            "to `rqstart` to initialize the actual query radius.\n")
        .def(
            "with_sequential_primal_interior_point",
            &MeshDynamicsParamsType::WithSequentialAugmentedLagrangian,
            nb::arg("gamma"),
            nb::arg("dmin"),
            nb::arg("epsP"),
            nb::rv_policy::reference_internal,
            "Set the sequential primal interior point parameters.\n\n"
            "Args:\n"
            "    gamma (float): Multiple of dynamics hessian curvature in constraint gradient "
            "direction for barrier parameter computation, `gamma > 0`.\n"
            "    dmin (float): Loose target minimum contact distance, `dmin > 0`.\n"
            "    epsP (float): Size of the smooth transition region, `0 < epsP < r` and "
            "`dmin < r - epsP`.\n")
        .def(
            "construct",
            &MeshDynamicsParamsType::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Construct the Params object.\n\n"
            "Args:\n"
            "    validate (bool, optional): Whether to validate parameters. Default is True.\n")
        .def(
            "compute_query_radius",
            &MeshDynamicsParamsType::ComputeQueryRadius,
            nb::arg("inertial_target_distance"),
            "Compute the contact query radius given the inertial target distance.\n\n"
            "Args:\n"
            "    inertial_target_distance (float): Inertial target distance (or other relevant "
            "distance).\n")
        .def(
            "activate",
            &MeshDynamicsParamsType::Activate,
            nb::arg("active") = true,
            "Activate (or deactivate) contacts.\n")
        .def("deactivate", &MeshDynamicsParamsType::Deactivate, "Deactivate contacts.\n")
        .def(
            "serialize",
            &MeshDynamicsParamsType::Serialize,
            nb::arg("archive"),
            "Serialize to archive.\n\n"
            "Args:\n"
            "    archive (io.Archive): Archive to serialize to.\n")
        .def(
            "deserialize",
            &MeshDynamicsParamsType::Deserialize,
            nb::arg("archive"),
            "Deserialize from archive.\n\n"
            "Args:\n"
            "    archive (io.Archive): Archive to deserialize from.\n")
        .def_rw("ogc_params", &MeshDynamicsParamsType::mOgcParams, "OGC parameters.")
        .def_rw(
            "epsv",
            &MeshDynamicsParamsType::epsv,
            "(float) IPC's relative velocity threshold for static to dynamic friction's smooth "
            "transition.")
        .def_rw("mu", &MeshDynamicsParamsType::mu, "(float) Dynamic friction coefficient.")
        .def_rw(
            "deactivate",
            &MeshDynamicsParamsType::bDeactivate,
            "(bool) Whether to deactivate contacts.")
        .def_rw("kc", &MeshDynamicsParamsType::kc, "(float) OGC contact stiffness parameter.")
        .def_rw("rqstart", &MeshDynamicsParamsType::rqstart, "(float) Initial query radius base.")
        .def_rw("betarq", &MeshDynamicsParamsType::betarq, "(float) Query radius growth factor.")
        .def_rw(
            "gamma",
            &MeshDynamicsParamsType::gamma,
            "(float) Multiple of dynamics hessian curvature in constraint gradient direction for "
            "barrier parameter computation.")
        .def_rw(
            "dmin",
            &MeshDynamicsParamsType::dmin,
            "(float) Loose target minimum contact distance.")
        .def_rw(
            "epsP",
            &MeshDynamicsParamsType::epsP,
            "(float) Size of smooth transition region for the polynomial step function "
            "approximation.");

    nb::class_<MeshDynamicsType>(m, "MeshDynamics")
        .def(nb::init<>(), "Construct an empty mesh contact dynamics engine.")
        .def(
            "construct",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
               MultiMeshType const& dynamicMeshes,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
               MultiMeshType const& staticMeshes) {
                self.Construct(Xdynamic, dynamicMeshes, Xstatic, staticMeshes);
            },
            nb::arg("Xdynamic"),
            nb::arg("dynamicMeshes"),
            nb::arg("Xstatic").none(),
            nb::arg("staticMeshes").none(),
            "Construct a MeshDynamics object with contact geometries.\n\n"
            "Args:\n"
            "    Xdynamic (numpy.ndarray): `3 x |# points|` dynamic point positions "
            "(column-major: one point per column).\n"
            "    dynamicMeshes (MultiMesh): Dynamic mesh contact geometry representation.\n"
            "    Xstatic (numpy.ndarray): `3 x |# points|` static point positions (column-major: "
            "one point per column).\n"
            "    staticMeshes (MultiMesh): Static mesh contact geometry representation.\n")
        .def(
            "set_dynamic_geometry",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
               MultiMeshType const& dynamicMeshes) {
                self.SetDynamicGeometry(Xdynamic, dynamicMeshes);
            },
            nb::arg("Xdynamic"),
            nb::arg("dynamicMeshes"),
            "Set the dynamic contact geometry.\n\n"
            "Args:\n"
            "    Xdynamic (numpy.ndarray): `3 x |# points|` dynamic point positions "
            "(column-major: one point per column).\n"
            "    dynamicMeshes (MultiMesh): Dynamic mesh contact geometry representation.\n")
        .def(
            "set_static_geometry",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
               MultiMeshType const& staticMeshes) {
                self.SetStaticGeometry(Xstatic, staticMeshes);
            },
            nb::arg("Xstatic"),
            nb::arg("staticMeshes"),
            "Set the static contact geometry.\n\n"
            "Args:\n"
            "    Xstatic (numpy.ndarray): `3 x |# points|` static point positions (column-major: "
            "one point per column).\n"
            "    staticMeshes (MultiMesh): Static mesh contact geometry representation.\n")
        .def(
            "initialize",
            &MeshDynamicsType::Initialize,
            nb::arg("device"),
            "Initialize the mesh dynamics engine.\n\n"
            "Args:\n"
            "    device (Device): Device to use for acceleration structures.\n")
        .def(
            "restore_feasibility",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xkp1,
               std::optional<Eigen::Vector<bool, Eigen::Dynamic> const> mask) {
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> Xkp1Copy = Xkp1;
                if (mask)
                    self.RestoreFeasibility(Xkp1Copy, *mask);
                else
                    self.RestoreFeasibility(Xkp1Copy);
                return Xkp1Copy;
            },
            nb::arg("xkp1"),
            nb::arg("mask").none(),
            "Restore the feasibility of the displaced positions within the precomputed bounds.\n\n"
            "Args:\n"
            "    xkp1 (numpy.ndarray): `3 x |# points|` point positions (column-major: one point "
            "per column).\n"
            "    mask (numpy.ndarray | None): `|# points| x 1` mask of points to ignore (true = "
            "ignore, false = process).\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# points|` truncated point positions.\n")
        .def(
            "make_step_feasible",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Dxkp1,
               std::optional<Eigen::Vector<bool, Eigen::Dynamic> const> mask) {
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> Dxkp1Copy = Dxkp1;
                if (mask)
                    self.MakeStepFeasible(Dxkp1Copy, *mask);
                else
                    self.MakeStepFeasible(
                        Dxkp1Copy,
                        Eigen::Vector<bool, Eigen::Dynamic>::Constant(Dxkp1Copy.cols(), false));
                return Dxkp1Copy;
            },
            nb::arg("dx"),
            nb::arg("mask").none(),
            "Make the step feasible by truncating the displacements to be within the precomputed "
            "bounds.\n\n"
            "Args:\n"
            "    dx (numpy.ndarray): `3 x |# points|` point displacements (column-major: "
            "one point per column).\n"
            "    mask (numpy.ndarray | None): `|# points| x 1` mask of points to ignore (true = "
            "ignore, false = process).\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# points|` truncated displacements.\n")
        .def(
            "request_constraint_set_update",
            &MeshDynamicsType::RequestConstraintSetUpdate,
            "Request constraint set update.")
        .def_prop_ro(
            "requires_constraint_set_update",
            &MeshDynamicsType::RequiresConstraintSetUpdate,
            "Whether constraint set needs to be recomputed.")
        .def_prop_ro(
            "num_truncated_points",
            &MeshDynamicsType::NumTruncatedPoints,
            "Number of truncated points.")
        .def(
            "update_constraint_set",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                self.UpdateConstraintSet(X);
            },
            nb::arg("X"),
            "Update the constraint set based on the current point positions.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# points|` current point positions (column-major: one "
            "point per column).\n")
        .def(
            "linearize_constraints",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& x) {
                self.LinearizeConstraints(x);
            },
            nb::arg("x"),
            "Linearize all contact constraints at the given positions.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `3 x |# points|` or `3*|# points| x 1` current point "
            "positions.\n")
        .def(
            "update_barrier_parameters",
            [](MeshDynamicsType& self,
               Eigen::SparseMatrix<ScalarType, Eigen::ColMajor, IndexType> const& H) {
                self.UpdateBarrierParameters(H);
            },
            nb::arg("H"),
            "Compute barrier parameters for all constraints based on an estimate of the "
            "objective function's Hessian.\n\n"
            "Args:\n"
            "    H (scipy.sparse.csc_matrix): Symmetric Hessian estimate in CSC format.\n")
        .def(
            "update_barrier_parameters",
            [](MeshDynamicsType& self,
               Eigen::SparseMatrix<ScalarType, Eigen::RowMajor, IndexType> const& H) {
                self.UpdateBarrierParameters(H);
            },
            nb::arg("H"),
            "Compute barrier parameters for all constraints based on an estimate of the "
            "objective function's Hessian.\n\n"
            "Args:\n"
            "    H (scipy.sparse.csc_matrix): Symmetric Hessian estimate in CSR format.\n")
        .def(
            "potential",
            [](MeshDynamicsType const& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& x,
               bool bForLinearSubproblem) { return self.Potential(x, bForLinearSubproblem); },
            nb::arg("x"),
            nb::arg("for_linear_subproblem") = false,
            "Compute the total contact potential energy.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `3 x |# points|` or `3*|# points| x 1` current point "
            "positions.\n"
            "    for_linear_subproblem (bool, optional): Whether the energy is being computed for "
            "a linear subproblem (from last `linearize_constraints()` call). Default is False.\n\n"
            "Returns:\n"
            "    float: Total contact potential energy.\n")
        .def(
            "gradient",
            [](MeshDynamicsType const& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& x,
               bool bForLinearSubproblem) { return self.Gradient(x, bForLinearSubproblem); },
            nb::arg("x"),
            nb::arg("for_linear_subproblem") = false,
            "Compute the total contact gradient.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `3 x |# points|` or `3*|# points| x 1` current point "
            "positions.\n"
            "    for_linear_subproblem (bool, optional): Whether the gradient is being computed "
            "for a linear subproblem (from last `linearize_constraints()` call). Default is "
            "False.\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3*|# points| x 1` total contact gradient.\n")
        .def(
            "serialize",
            &MeshDynamicsType::Serialize,
            nb::arg("archive"),
            "Serialize to archive.\n\n"
            "Args:\n"
            "    archive (io.Archive): Archive to serialize to.\n")
        .def(
            "deserialize",
            &MeshDynamicsType::Deserialize,
            nb::arg("archive"),
            "Deserialize from archive. initialize() must be called after deserialization for "
            "usability.\n\n"
            "Args:\n"
            "    archive (io.Archive): Archive to deserialize from.\n")
        .def_prop_rw(
            "params",
            [](MeshDynamicsType const& self) -> MeshDynamicsParamsType const& {
                return self.GetParams();
            },
            [](MeshDynamicsType& self, MeshDynamicsParamsType const& params) {
                self.GetParams() = params;
            },
            nb::rv_policy::reference_internal,
            "Mesh dynamics parameters.")
        .def_prop_ro("Xstatic", &MeshDynamicsType::StaticPointPositions, "Static point positions.")
        .def_prop_ro("dynamic_meshes", &MeshDynamicsType::DynamicMeshes, "Dynamic meshes.")
        .def_prop_ro("static_meshes", &MeshDynamicsType::StaticMeshes, "Static meshes.")
        .def_prop_ro(
            "ogc_input",
            [](MeshDynamicsType& self) -> decltype(auto) { return self.OgcInput(); },
            "OGC input.")
        .def_prop_ro(
            "ogc_state",
            [](MeshDynamicsType& self) -> decltype(auto) { return self.OgcState(); },
            "OGC state.")
        .def_prop_ro("num_contacts", &MeshDynamicsType::NumContacts, "Total number of contacts.");
}

} // namespace pbat::py::sim::contact
