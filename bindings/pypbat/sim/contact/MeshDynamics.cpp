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
    using MeshDynamicsParamsType = pbat::sim::contact::MeshDynamicsParams<ScalarType, IndexType>;
    using MultiMeshType          = pbat::sim::contact::MultiMesh<IndexType>;
    using DeviceType             = pbat::geometry::Device;

    nb::class_<MeshDynamicsParamsType>(m, "MeshDynamicsParams")
        .def(nb::init<>(), "Construct default MeshDynamicsParams.")
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
            nb::arg("params"),
            "Initialize the mesh dynamics engine.\n\n"
            "Args:\n"
            "    device (Device): Device to use for acceleration structures.\n"
            "    params (MeshDynamicsParams): Initialization parameters.\n")
        .def_prop_ro("dynamic_meshes", &MeshDynamicsType::DynamicMeshes, "Dynamic meshes.")
        .def_prop_ro("static_meshes", &MeshDynamicsType::StaticMeshes, "Static meshes.")
        .def_prop_ro("ogc_input", &MeshDynamicsType::OgcInput, "OGC input.")
        .def_prop_ro("ogc_state", &MeshDynamicsType::OgcState, "OGC state.");
}

} // namespace pbat::py::sim::contact
