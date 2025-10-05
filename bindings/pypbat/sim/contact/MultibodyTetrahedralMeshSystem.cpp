#include "MultibodyTetrahedralMeshSystem.h"

#include <nanobind/eigen/dense.h>
#include <pbat/sim/contact/MultibodyTetrahedralMeshSystem.h>

namespace pbat::py::sim::contact {

void BindMultibodyTetrahedralMeshSystem(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::contact::MultibodyTetrahedralMeshSystem;
    using IndexType                          = Index;
    using ScalarType                         = Scalar;
    using MultibodyTetrahedralMeshSystemType = MultibodyTetrahedralMeshSystem<IndexType>;
    nb::class_<MultibodyTetrahedralMeshSystemType>(m, "MultibodyTetrahedralMeshSystem")
        .def(nb::init<>())
        .def(
            "Construct",
            [](MultibodyTetrahedralMeshSystemType& self,
               Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic>> X,
               Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T) { self.Construct(X, T); },
            nb::arg("X").noconvert(),
            nb::arg("T").noconvert(),
            "Construct a new multibody tetrahedral mesh system from vertex positions and "
            "tetrahedral connectivity.\n\n"
            "This method sorts the input mesh vertex positions and element indices by body.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# mesh vertices|` matrix of vertex positions.\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity.\n"
            "Returns:\n"
            "    None\n")
        .def_prop_ro("n_bodies", &MultibodyTetrahedralMeshSystemType::NumBodies)
        .def_ro("V", &MultibodyTetrahedralMeshSystemType::V)
        .def_ro("E", &MultibodyTetrahedralMeshSystemType::E)
        .def_ro("F", &MultibodyTetrahedralMeshSystemType::F)
        .def_ro("VP", &MultibodyTetrahedralMeshSystemType::VP)
        .def_ro("EP", &MultibodyTetrahedralMeshSystemType::EP)
        .def_ro("FP", &MultibodyTetrahedralMeshSystemType::FP)
        .def_ro("TP", &MultibodyTetrahedralMeshSystemType::TP)
        .def_ro("CC", &MultibodyTetrahedralMeshSystemType::CC)
        .def(
            "vertices_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.ContactVerticesOf(body).eval();
            },
            nb::arg("body"),
            "Get vertices of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "Returns:\n"
            "    numpy.ndarray: `|# contact vertices of body body| x 1` indices into mesh "
            "vertices.\n")
        .def(
            "edges_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.ContactEdgesOf(body).eval();
            },
            nb::arg("body"),
            "Get edges of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "Returns:\n"
            "    numpy.ndarray: `2 x |# contact edges of body body|` edges into mesh vertices.\n")
        .def(
            "triangles_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.ContactTrianglesOf(body).eval();
            },
            nb::arg("body"),
            "Get triangles of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# contact triangles of body body|` triangles into mesh "
            "vertices.\n")
        .def(
            "tetrahedra_of",
            [](MultibodyTetrahedralMeshSystemType& self,
               IndexType body,
               nb::DRef<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T) {
                return self.TetrahedraOf(body, T).eval();
            },
            nb::arg("body"),
            nb::arg("T"),
            "Get tetrahedra of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity.\n"
            "Returns:\n"
            "    numpy.ndarray: `4 x |# contact tetrahedra of body body|` tetrahedra into input "
            "tetrahedral mesh `T`.\n")
        .def(
            "vertex_positions_of",
            [](MultibodyTetrahedralMeshSystemType& self,
               IndexType body,
               nb::DRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                return self.ContactVertexPositionsOf(body, X).eval();
            },
            nb::arg("body"),
            nb::arg("X"),
            "Get vertex positions of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "    X (numpy.ndarray): `3 x |# mesh vertices|` matrix of vertex positions.\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# contact vertices of body body|` vertex positions into "
            "input vertex position matrix `X`.\n");
}

} // namespace pbat::py::sim::contact