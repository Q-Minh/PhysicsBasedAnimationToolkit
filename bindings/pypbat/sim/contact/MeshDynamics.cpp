#include "MeshDynamics.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/geometry/Device.h>
#include <pbat/geometry/sdf/Composite.h>
#include <pbat/geometry/sdf/Forest.h>
#include <pbat/sim/contact/MeshDynamics.h>
#include <pbat/sim/contact/MeshSdfContact.h>
#include <pbat/sim/contact/MultiMesh.h>

namespace pbat::py::sim::contact {

void BindMeshDynamics(nanobind::module_& m)
{
    namespace nb                               = nanobind;
    using ScalarType                           = Scalar;
    using IndexType                            = Index;
    using MeshDynamicsType                     = pbat::sim::contact::MeshDynamics;
    using EnvironmentContactConstraintType     = MeshDynamicsType::EnvironmentContact;
    using EnvironmentContactDynamicsParamsType = MeshDynamicsType::EnvironmentContactDynamicsParams;
    using MultiMeshType                        = pbat::sim::contact::MultiMesh<IndexType>;
    using MeshSdfContactType                   = pbat::sim::contact::MeshSdfContact;
    using MeshSdfContactParamsType             = pbat::sim::contact::MeshSdfContactParams;
    using ForestType                           = pbat::geometry::sdf::Forest<ScalarType>;
    using CompositeType                        = pbat::geometry::sdf::Composite<ScalarType>;
    using DeviceType                           = pbat::geometry::Device;

    nb::class_<EnvironmentContactConstraintType>(m, "EnvironmentContact")
        .def(nb::init<>(), "Construct an empty environment contact constraint.")
        .def_rw(
            "O",
            &EnvironmentContactConstraintType::O,
            "(numpy.ndarray) Contact basis origin (3 x 1).")
        .def_rw(
            "B",
            &EnvironmentContactConstraintType::B,
            "(numpy.ndarray) Contact basis (3 x 3 matrix; columns: normal, tangent, bitangent).")
        .def_rw(
            "C",
            &EnvironmentContactConstraintType::C,
            "(numpy.ndarray) Contact constraint values (normal, tangent, bitangent) (3 x 1).")
        .def_rw(
            "lagrange",
            &EnvironmentContactConstraintType::lambda,
            "(numpy.ndarray) Contact Lagrange multiplier estimates (normal, tangent, bitangent) (3 "
            "x 1).")
        .def_rw(
            "k",
            &EnvironmentContactConstraintType::k,
            "(numpy.ndarray) Contact stiffness (normal, tangent, bitangent) (3 x 1).")
        .def_rw("mu", &EnvironmentContactConstraintType::mu, "(float) Friction coefficient.");

    nb::class_<EnvironmentContactDynamicsParamsType>(m, "EnvironmentContactDynamicsParams")
        .def(nb::init<>(), "Create default environment contact dynamics parameters.")
        .def_rw("mu", &EnvironmentContactDynamicsParamsType::mu, "(float) Friction coefficient.")
        .def_rw(
            "kstart",
            &EnvironmentContactDynamicsParamsType::kstart,
            "(float) Initial contact stiffness for new contacts.")
        .def_rw(
            "beta",
            &EnvironmentContactDynamicsParamsType::beta,
            "(float) Multiplier of constraint error for stiffness update.")
        .def_rw(
            "gamma",
            &EnvironmentContactDynamicsParamsType::gamma,
            "(float) Decay factor for Lagrange multiplier and stiffness initialization at time "
            "step begin.")
        .def_rw(
            "Fnmax",
            &EnvironmentContactDynamicsParamsType::Fnmax,
            "(float) Maximum normal contact force density magnitude.")
        .def_rw(
            "kmax",
            &EnvironmentContactDynamicsParamsType::kmax,
            "(float) Maximum contact stiffness.")
        .def_rw(
            "epsv",
            &EnvironmentContactDynamicsParamsType::epsv,
            "(float) Relative velocity threshold for static to dynamic friction transition.");

    nb::class_<MeshDynamicsType>(m, "MeshDynamics")
        .def(nb::init<>(), "Construct an empty mesh contact dynamics engine.")
        .def(
            "set_static_geometry",
            [](MeshDynamicsType& self, ForestType sdfForest) {
                self.SetStaticGeometry(std::move(sdfForest));
            },
            nb::arg("sdf_forest"),
            "Set the static geometry.\n\n"
            "Args:\n"
            "    sdf_forest (pbat.geometry.sdf.Forest): SDF static geometry storage.")
        .def(
            "set_dynamic_geometry",
            [](MeshDynamicsType& self, MultiMeshType meshes) {
                self.SetDynamicGeometry(std::move(meshes));
            },
            nb::arg("meshes"),
            "Set the dynamic geometry.\n\n"
            "Args:\n"
            "    meshes (MultiMesh): Mesh contact geometry representation.")
        .def(
            "allocate_environment_contact_data_structures",
            &MeshDynamicsType::AllocateEnvironmentContactDataStructures,
            "Allocate data structures for environment contact detection.\n\n")
        .def(
            "construct",
            [](MeshDynamicsType& self, MultiMeshType meshes, ForestType sdfForest) {
                self.Construct(std::move(meshes), std::move(sdfForest));
            },
            nb::arg("meshes"),
            nb::arg("sdf_forest"),
            "Set the contact geometries and initialize data structures.\n\n"
            "Args:\n"
            "    meshes (MultiMesh): Mesh contact geometry representation.\n"
            "    sdf_forest (pbat.geometry.sdf.Forest): SDF static geometry storage.\n")
        .def(
            "initialize_mesh_mesh_contact_detection",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
               DeviceType const& device) { self.InitializeMeshMeshContactDetection(X, device); },
            nb::arg("X"),
            nb::arg("device"),
            "Initialize mesh-mesh contact detection.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.\n"
            "    device (pbat.geometry.Device): Spatial acceleration device.")
        .def(
            "initialize_mesh_environment_contact_detection",
            &MeshDynamicsType::InitializeMeshEnvironmentContactDetection,
            "Initialize mesh-SDF contact detection.\n\n")
        .def(
            "update_environment_contact_constraints",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                self.UpdateEnvironmentContactConstraints(X);
            },
            nb::arg("X"),
            "Reformulate mesh-SDF contact constraints, zeroing out inactive constraints and their "
            "Lagrange multipliers.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.")
        .def(
            "prepare_environment_contacts_for_dual_iteration",
            &MeshDynamicsType::PrepareEnvironmentContactsForDualIteration,
            "Prepare Lagrange multiplier estimates and contact stiffnesses for solver dual "
            "iteration.")
        .def(
            "dual_update_environment_contacts",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                self.DualUpdateEnvironmentContacts(X);
            },
            nb::arg("X"),
            "Update Lagrange multiplier estimates and contact stiffnesses for mesh-SDF contact.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3 x |# points| point positions.")
        .def_rw(
            "CF",
            &MeshDynamicsType::CF,
            "(List[EnvironmentContact]) |# triangle-env contacts| mesh-SDF triangle "
            "contact constraints, sorted by triangle index.")
        .def_rw(
            "CHE",
            &MeshDynamicsType::CHE,
            "(List[EnvironmentContact]) |# half-edge-env contacts| mesh-SDF half-edge "
            "contact constraints, sorted by half-edge index.")
        .def_rw(
            "CV",
            &MeshDynamicsType::CV,
            "(List[EnvironmentContact]) |# vertex-env contacts| mesh-SDF vertex contact "
            "constraints, sorted by vertex index.")
        .def_rw(
            "env_contact_dynamics_params",
            &MeshDynamicsType::mEnvContactDynamicsParams,
            "(EnvironmentContactDynamicsParams) Environment contact dynamics parameters.")
        .def_rw("FA", &MeshDynamicsType::FA, "(numpy.ndarray) |# triangles| x 1 triangle areas.")
        .def_rw(
            "HEA",
            &MeshDynamicsType::HEA,
            "(numpy.ndarray) |# half-edges| x 1 half-edge areas.")
        .def_rw("VA", &MeshDynamicsType::VA, "(numpy.ndarray) |# vertices| x 1 vertex areas.")
        .def_rw("meshes", &MeshDynamicsType::mMeshes, "(MultiMesh) Dynamic geometry.")
        .def_ro(
            "sdf_forest",
            &MeshDynamicsType::mSdfForest,
            "(pbat.geometry.sdf.Forest) Static geometry representation.")
        .def_ro(
            "sdf",
            &MeshDynamicsType::mSdf,
            "(pbat.geometry.sdf.Composite) SDF of static geometry.")
        .def_ro(
            "mesh_sdf_contact",
            &MeshDynamicsType::mMeshSdfContact,
            "(MeshSdfContact) Mesh-SDF contact detection.");
}

} // namespace pbat::py::sim::contact
