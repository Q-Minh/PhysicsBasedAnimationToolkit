#include "MeshDynamics.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <pbat/geometry/Device.h>
#include <pbat/math/linalg/mini/Eigen.h>
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
    using DeviceType             = pbat::geometry::Device;
    using EMeshEnergyComputationFlags = pbat::sim::contact::EMeshEnergyComputationFlags;
    using MeshContactEnergy1 = pbat::sim::contact::MeshContactEnergy<ScalarType, IndexType, 1>;
    using MeshContactEnergy2 = pbat::sim::contact::MeshContactEnergy<ScalarType, IndexType, 2>;
    using MeshContactEnergy3 = pbat::sim::contact::MeshContactEnergy<ScalarType, IndexType, 3>;
    using MeshContactEnergy4 = pbat::sim::contact::MeshContactEnergy<ScalarType, IndexType, 4>;

    nb::enum_<EMeshEnergyComputationFlags>(m, "EMeshEnergyComputationFlags", nb::is_arithmetic())
        .value("Potential", EMeshEnergyComputationFlags::Potential, "Compute potential energy.")
        .value("Gradient", EMeshEnergyComputationFlags::Gradient, "Compute contact gradients.")
        .value("Hessian", EMeshEnergyComputationFlags::Hessian, "Compute contact hessians.")
        .export_values();

    nb::class_<MeshContactEnergy1>(m, "MeshContactEnergy_1NodeStencil")
        .def(nb::init<>(), "Construct a MeshContactEnergy with 1-node stencil (3 DOFs).")
        .def_ro("En", &MeshContactEnergy1::En, "(float) Normal contact energy.")
        .def_ro("Ef", &MeshContactEnergy1::Ef, "(float) Frictional contact energy.")
        .def_prop_ro(
            "gradEn",
            [](MeshContactEnergy1 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy1::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEn);
            },
            "(numpy.ndarray) Normal contact energy gradient (3x1).")
        .def_prop_ro(
            "hessEn",
            [](MeshContactEnergy1 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy1::kDofs, MeshContactEnergy1::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEn);
            },
            "(numpy.ndarray) Normal contact energy Hessian (3x3).")
        .def_prop_ro(
            "gradEf",
            [](MeshContactEnergy1 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy1::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEf);
            },
            "(numpy.ndarray) Frictional contact energy gradient (3x1).")
        .def_prop_ro(
            "hessEf",
            [](MeshContactEnergy1 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy1::kDofs, MeshContactEnergy1::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEf);
            },
            "(numpy.ndarray) Frictional contact energy Hessian (3x3).")
        .def_ro(
            "stencil",
            &MeshContactEnergy1::stencil,
            "(numpy.ndarray) Indices of involved vertices (1x1).");

    nb::class_<MeshContactEnergy2>(m, "MeshContactEnergy_2NodeStencil")
        .def(nb::init<>(), "Construct a MeshContactEnergy with 2-node stencil (6 DOFs).")
        .def_ro("En", &MeshContactEnergy2::En, "(float) Normal contact energy.")
        .def_ro("Ef", &MeshContactEnergy2::Ef, "(float) Frictional contact energy.")
        .def_prop_ro(
            "gradEn",
            [](MeshContactEnergy2 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy2::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEn);
            },
            "(numpy.ndarray) Normal contact energy gradient (6x1).")
        .def_prop_ro(
            "hessEn",
            [](MeshContactEnergy2 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy2::kDofs, MeshContactEnergy2::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEn);
            },
            "(numpy.ndarray) Normal contact energy Hessian (6x6).")
        .def_prop_ro(
            "gradEf",
            [](MeshContactEnergy2 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy2::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEf);
            },
            "(numpy.ndarray) Frictional contact energy gradient (6x1).")
        .def_prop_ro(
            "hessEf",
            [](MeshContactEnergy2 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy2::kDofs, MeshContactEnergy2::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEf);
            },
            "(numpy.ndarray) Frictional contact energy Hessian (6x6).")
        .def_ro(
            "stencil",
            &MeshContactEnergy2::stencil,
            "(numpy.ndarray) Indices of involved vertices (2x1).");

    nb::class_<MeshContactEnergy3>(m, "MeshContactEnergy_3NodeStencil")
        .def(nb::init<>(), "Construct a MeshContactEnergy with 3-node stencil (9 DOFs).")
        .def_ro("En", &MeshContactEnergy3::En, "(float) Normal contact energy.")
        .def_ro("Ef", &MeshContactEnergy3::Ef, "(float) Frictional contact energy.")
        .def_prop_ro(
            "gradEn",
            [](MeshContactEnergy3 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy3::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEn);
            },
            "(numpy.ndarray) Normal contact energy gradient (9x1).")
        .def_prop_ro(
            "hessEn",
            [](MeshContactEnergy3 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy3::kDofs, MeshContactEnergy3::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEn);
            },
            "(numpy.ndarray) Normal contact energy Hessian (9x9).")
        .def_prop_ro(
            "gradEf",
            [](MeshContactEnergy3 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy3::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEf);
            },
            "(numpy.ndarray) Frictional contact energy gradient (9x1).")
        .def_prop_ro(
            "hessEf",
            [](MeshContactEnergy3 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy3::kDofs, MeshContactEnergy3::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEf);
            },
            "(numpy.ndarray) Frictional contact energy Hessian (9x9).")
        .def_ro(
            "stencil",
            &MeshContactEnergy3::stencil,
            "(numpy.ndarray) Indices of involved vertices (3x1).");

    nb::class_<MeshContactEnergy4>(m, "MeshContactEnergy_4NodeStencil")
        .def(nb::init<>(), "Construct a MeshContactEnergy with 4-node stencil (12 DOFs).")
        .def_ro("En", &MeshContactEnergy4::En, "(float) Normal contact energy.")
        .def_ro("Ef", &MeshContactEnergy4::Ef, "(float) Frictional contact energy.")
        .def_prop_ro(
            "gradEn",
            [](MeshContactEnergy4 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy4::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEn);
            },
            "(numpy.ndarray) Normal contact energy gradient (12x1).")
        .def_prop_ro(
            "hessEn",
            [](MeshContactEnergy4 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy4::kDofs, MeshContactEnergy4::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEn);
            },
            "(numpy.ndarray) Normal contact energy Hessian (12x12).")
        .def_prop_ro(
            "gradEf",
            [](MeshContactEnergy4 const& self)
                -> Eigen::Vector<ScalarType, MeshContactEnergy4::kDofs> {
                return math::linalg::mini::ToEigen(self.gradEf);
            },
            "(numpy.ndarray) Frictional contact energy gradient (12x1).")
        .def_prop_ro(
            "hessEf",
            [](MeshContactEnergy4 const& self)
                -> Eigen::Matrix<ScalarType, MeshContactEnergy4::kDofs, MeshContactEnergy4::kDofs> {
                return math::linalg::mini::ToEigen(self.hessEf);
            },
            "(numpy.ndarray) Frictional contact energy Hessian (12x12).")
        .def_ro(
            "stencil",
            &MeshContactEnergy4::stencil,
            "(numpy.ndarray) Indices of involved vertices (4x1).");

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
        .def_rw("kc", &MeshDynamicsParamsType::kc, "(float) OGC contact stiffness parameter.")
        .def_rw("rqstart", &MeshDynamicsParamsType::rqstart, "(float) Initial query radius base.")
        .def_rw("betarq", &MeshDynamicsParamsType::betarq, "(float) Query radius growth factor.");

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
            "    Xdynamic (Eigen.Matrix): `3 x |# points|` dynamic point positions (column-major: "
            "one point per column).\n"
            "    dynamicMeshes (MultiMesh): Dynamic mesh contact geometry representation.\n"
            "    Xstatic (Eigen.Matrix): `3 x |# points|` static point positions (column-major: "
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
            "    Xdynamic (Eigen.Matrix): `3 x |# points|` dynamic point positions (column-major: "
            "one point per column).\n"
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
            "    Xstatic (Eigen.Matrix): `3 x |# points|` static point positions (column-major: "
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
            "truncate_displaced_positions",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xkp1,
               std::optional<Eigen::Vector<bool, Eigen::Dynamic> const> mask) {
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> Xkp1Copy = Xkp1;
                if (mask)
                    self.TruncateDisplacedPositions(Xkp1Copy, *mask);
                else
                    self.TruncateDisplacedPositions(Xkp1Copy);
                return Xkp1Copy;
            },
            nb::arg("xkp1"),
            nb::arg("mask").none(),
            "Truncate the displacements to be within the precomputed bounds.\n\n"
            "Args:\n"
            "    xkp1 (Eigen.Matrix): `3 x |# points|` point positions (column-major: one point "
            "per column).\n"
            "    mask (Eigen.Matrix, optional): `|# points| x 1` mask of points to ignore (true = "
            "ignore, false = process).\n\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# points|` truncated point positions.\n")
        .def(
            "truncate_displacements",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Dxkp1,
               std::optional<Eigen::Vector<bool, Eigen::Dynamic> const> mask) {
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> Dxkp1Copy = Dxkp1;
                if (mask)
                    self.TruncateDisplacements(Dxkp1Copy, *mask);
                else
                    self.TruncateDisplacements(
                        Dxkp1Copy,
                        Eigen::Vector<bool, Eigen::Dynamic>::Constant(Dxkp1Copy.cols(), false));
                return Dxkp1Copy;
            },
            nb::arg("dx"),
            nb::arg("mask").none(),
            "Truncate the displacements to be within the precomputed bounds.\n\n"
            "Args:\n"
            "    dx (numpy.ndarray): `3 x |# points|` point displacements (column-major: "
            "one point per column).\n"
            "    mask (numpy.ndarray | None): `|# points| x 1` mask of points to ignore (true = "
            "ignore, false = process).\n")
        .def(
            "request_displacement_bounds_computation",
            &MeshDynamicsType::RequestDisplacementBoundsComputation,
            "Request recomputation of displacement bounds.")
        .def_prop_ro(
            "requires_bounds_computation",
            &MeshDynamicsType::RequiresBoundsComputation,
            "Whether displacement bounds need to be recomputed.")
        .def_prop_ro(
            "num_truncated_points",
            &MeshDynamicsType::NumTruncatedPoints,
            "Number of truncated points.")
        .def(
            "compute_displacement_bounds",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                self.ComputeDisplacementBounds(X);
            },
            nb::arg("X"),
            "Compute the per-point displacement bounds based on current geometry and OGC state.\n\n"
            "Args:\n"
            "    X (Eigen.Matrix): `3 x |# points|` current point positions (column-major: one "
            "point per column).\n")
        .def(
            "compute_energies",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& x,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> const& xt,
               ScalarType h,
               EMeshEnergyComputationFlags eFlags) { self.ComputeEnergies(x, xt, h, eFlags); },
            nb::arg("x"),
            nb::arg("xt"),
            nb::arg("h"),
            nb::arg("computation_flags"),
            "Compute the contact energies based on current geometry.\n\n"
            "Args:\n"
            "    x (Eigen.Matrix): `3*|# points| x 1` or `3 x |# points|` current point positions "
            "(column-major: one point per column).\n"
            "    xt (Eigen.Matrix): `3*|# points| x 1` or `3 x |# points|` previous point "
            "positions (column-major: one point per column).\n"
            "    h (float): Time step size.\n"
            "    computation_flags (EMeshEnergyComputationFlags): Flags controlling which energy "
            "components to compute (e.g., potential, gradient, hessian).\n")
        .def_prop_ro("potential", &MeshDynamicsType::Potential, "Contact potential energy.")
        .def_prop_ro(
            "gradient",
            &MeshDynamicsType::Gradient,
            "`3*|# points| x 1` contact energy gradient.")
        .def_prop_ro(
            "normal_gradient",
            &MeshDynamicsType::NormalGradient,
            "`3*|# points| x 1` normal contact energy gradient.")
        .def_prop_ro(
            "frictional_gradient",
            &MeshDynamicsType::FrictionalGradient,
            "`3*|# points| x 1` frictional contact energy gradient.")
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
        .def_prop_ro(
            "vertex_vertex_energies",
            &MeshDynamicsType::VertexVertexEnergies,
            "List of vertex-vertex contact energies (list of MeshContactEnergy2).")
        .def_prop_ro(
            "vertex_edge_energies",
            &MeshDynamicsType::VertexEdgeEnergies,
            "List of vertex-edge contact energies (list of MeshContactEnergy3).")
        .def_prop_ro(
            "vertex_triangle_energies",
            &MeshDynamicsType::VertexTriangleEnergies,
            "List of vertex-triangle contact energies (list of MeshContactEnergy4).")
        .def_prop_ro(
            "edge_edge_energies",
            &MeshDynamicsType::EdgeEdgeEnergies,
            "List of edge-edge contact energies (list of MeshContactEnergy4).")
        .def_prop_ro(
            "vertex_environment_energies",
            &MeshDynamicsType::VertexEnvironmentEnergies,
            "List of vertex-environment contact energies (list of MeshContactEnergy1).")
        .def_prop_ro(
            "edge_environment_energies",
            &MeshDynamicsType::EdgeEnvironmentEnergies,
            "List of edge-environment contact energies (list of MeshContactEnergy2).")
        .def_prop_ro(
            "triangle_environment_energies",
            &MeshDynamicsType::TriangleEnvironmentEnergies,
            "List of triangle-environment contact energies (list of MeshContactEnergy3).")
        .def_prop_ro("num_contacts", &MeshDynamicsType::NumContacts, "Number of contacts.");
}

} // namespace pbat::py::sim::contact
