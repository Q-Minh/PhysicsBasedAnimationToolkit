#include "MeshDynamics.h"

#include <nanobind/eigen/dense.h>
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
    using DeviceType             = pbat::geometry::Device;

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
        .def_rw("kc", &MeshDynamicsParamsType::kc, "(float) OGC contact stiffness parameter.");

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
            "truncate_displacement",
            [](MeshDynamicsType& self,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xkp1,
               std::optional<Eigen::Vector<bool, Eigen::Dynamic> const> mask) {
                Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> Xkp1Copy = Xkp1;
                if (mask)
                    self.TruncateDisplacement(Xkp1Copy, *mask);
                else
                    self.TruncateDisplacement(Xkp1Copy);
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
            &MeshDynamicsType::ComputeDisplacementBounds,
            "Compute the per-point displacement bounds based on current geometry and OGC state.")
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
        .def_prop_ro("params", &MeshDynamicsType::GetParams, "Mesh dynamics parameters.")
        .def_prop_ro("dynamic_meshes", &MeshDynamicsType::DynamicMeshes, "Dynamic meshes.")
        .def_prop_ro("static_meshes", &MeshDynamicsType::StaticMeshes, "Static meshes.")
        .def_prop_ro("ogc_input", &MeshDynamicsType::OgcInput, "OGC input.")
        .def_prop_ro("ogc_state", &MeshDynamicsType::OgcState, "OGC state.");
}

} // namespace pbat::py::sim::contact
