#include "HalfEdges.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <pbat/geometry/HalfEdges.h>

namespace pbat::py::geometry {

void BindHalfEdges(nanobind::module_& m)
{
    namespace nb    = nanobind;
    using IndexType = Index;

    m.def(
        "vertex_half_edge_adjacency",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F, IndexType n) {
            auto [GVHEp, GVHEadj] = pbat::geometry::VertexHalfEdgeAdjacency(F, n);
            return std::make_tuple(GVHEp, GVHEadj);
        },
        nb::arg("F"),
        nb::arg("n_vertices") = IndexType(-1),
        "Build the vertex-to-half-edge adjacency (CSR) for a triangle mesh.\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices\n"
        "    n_vertices (int): Number of vertices (optional, -1 infers from F)\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, numpy.ndarray]: (GVHEp, GVHEadj) where GVHEp is `|#v+1|` and\n"
        "    GVHEadj is `|3*#triangles|`.\n");

    m.def(
        "half_edge_face_adjacency",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F) {
            return pbat::geometry::HalfEdgeFaceAdjacency(F);
        },
        nb::arg("F"),
        "Map half-edges to their two adjacent faces (pair per half-edge).\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices\n"
        "Returns:\n"
        "    numpy.ndarray: `2 x |3*#triangles|` mapping half-edges to adjacent faces (-1 if "
        "none).\n");

    m.def(
        "edge_half_edge_adjacency",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
           nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF) {
            return pbat::geometry::EdgeHalfEdgeAdjacency(F, GHEF);
        },
        nb::arg("F"),
        nb::arg("GHEF"),
        "Map undirected edges to their two half-edges.\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices\n"
        "    GHEF (numpy.ndarray): `2 x |3*#triangles|` half-edge to face adjacency\n"
        "Returns:\n"
        "    numpy.ndarray: `2 x |# edges|` edge-to-half-edge adjacency (-1 if boundary).\n");

    m.def(
        "edges",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
           nb::DRef<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE) {
            return pbat::geometry::Edges(F, EHE);
        },
        nb::arg("F"),
        nb::arg("EHE"),
        "Build the undirected edge list from a triangle mesh's half-edge representation.\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices\n"
        "    EHE (numpy.ndarray): `2 x |# edges|` edge-to-half-edge adjacency\n"
        "Returns:\n"
        "    numpy.ndarray: `2 x |# edges|` undirected edges.\n");

    m.def(
        "half_edges",
        [](nb::DRef<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F) {
            return pbat::geometry::HalfEdges(F);
        },
        nb::arg("F"),
        "Build the half-edge list for a triangle mesh.\n\n"
        "Args:\n"
        "    F (numpy.ndarray): `3 x |# triangles|` triangle vertex indices\n"
        "Returns:\n"
        "    numpy.ndarray: `2 x |3*#triangles|` half-edges.\n");
}

} // namespace pbat::py::geometry
