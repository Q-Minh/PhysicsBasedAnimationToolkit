#include "MultibodyTetrahedralMeshSystem.h"

#include <pbat/sim/contact/MultibodyTetrahedralMeshSystem.h>
#include <pybind11/eigen.h>

namespace pbat::py::sim::contact {

void BindMultibodyTetrahedralMeshSystem(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::contact::MultibodyTetrahedralMeshSystem;
    using IndexType                          = Index;
    using ScalarType                         = Scalar;
    using MultibodyTetrahedralMeshSystemType = MultibodyTetrahedralMeshSystem<IndexType>;
    pyb::class_<MultibodyTetrahedralMeshSystemType>(m, "MultibodyTetrahedralMeshSystem")
        .def(pyb::init<>())
        .def(
            "Construct",
            [](MultibodyTetrahedralMeshSystemType& self,
               Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic>> X,
               Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T) { self.Construct(X, T); },
            pyb::arg("X").noconvert(),
            pyb::arg("T").noconvert(),
            "Construct a new multibody tetrahedral mesh system from vertex positions and "
            "tetrahedral connectivity.\n\n"
            "This method sorts the input mesh vertex positions and element indices by body.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): `3 x |# mesh vertices|` matrix of vertex positions.\n"
            "    T (numpy.ndarray): `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity.\n"
            "Returns:\n"
            "    None\n")
        .def_property_readonly("n_bodies", &MultibodyTetrahedralMeshSystemType::NumberOfBodies)
        .def_readonly("V", &MultibodyTetrahedralMeshSystemType::V)
        .def_readonly("E", &MultibodyTetrahedralMeshSystemType::E)
        .def_readonly("F", &MultibodyTetrahedralMeshSystemType::F)
        .def_readonly("VP", &MultibodyTetrahedralMeshSystemType::VP)
        .def_readonly("EP", &MultibodyTetrahedralMeshSystemType::EP)
        .def_readonly("FP", &MultibodyTetrahedralMeshSystemType::FP)
        .def_readonly("TP", &MultibodyTetrahedralMeshSystemType::TP)
        .def_readonly("CC", &MultibodyTetrahedralMeshSystemType::CC)
        .def(
            "vertices_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.VerticesOf(body).eval();
            },
            pyb::arg("body"),
            "Get vertices of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "Returns:\n"
            "    numpy.ndarray: `|# contact vertices of body body| x 1` indices into mesh "
            "vertices.\n")
        .def(
            "edges_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.EdgesOf(body).eval();
            },
            pyb::arg("body"),
            "Get edges of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "Returns:\n"
            "    numpy.ndarray: `2 x |# contact edges of body body|` edges into mesh vertices.\n")
        .def(
            "triangles_of",
            [](MultibodyTetrahedralMeshSystemType& self, IndexType body) {
                return self.TrianglesOf(body).eval();
            },
            pyb::arg("body"),
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
               pyb::EigenDRef<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T) {
                return self.TetrahedraOf(body, T).eval();
            },
            pyb::arg("body"),
            pyb::arg("T"),
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
               pyb::EigenDRef<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X) {
                return self.VertexPositionsOf(body, X).eval();
            },
            pyb::arg("body"),
            pyb::arg("X"),
            "Get vertex positions of body `body`.\n\n"
            "Args:\n"
            "    body (int): Index of the body.\n"
            "    X (numpy.ndarray): `3 x |# mesh vertices|` matrix of vertex positions.\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x |# contact vertices of body body|` vertex positions into "
            "input vertex position matrix `X`.\n");
}

} // namespace pbat::py::sim::contact