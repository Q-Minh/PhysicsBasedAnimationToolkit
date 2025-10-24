#include "Mesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <pbat/graph/Mesh.h>
#include <tuple>

namespace pbat {
namespace py {
namespace graph {

void BindMesh(nanobind::module_& m)
{
    namespace nb = nanobind;
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C,
           Eigen::Ref<IndexVectorX const> const& w,
           Index nNodes) { return pbat::graph::MeshAdjacencyMatrix(C, w, nNodes); },
        nb::arg("C"),
        nb::arg("w"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "w: (np.ndarray): |#nodes per element|x|#elements| array of edge weights w(c,v)\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C,
           Eigen::Ref<VectorX const> const& w,
           Index nNodes) { return pbat::graph::MeshAdjacencyMatrix(C, w, nNodes); },
        nb::arg("C"),
        nb::arg("w"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "w: (np.ndarray): |#nodes per element|x|#elements| array of edge weights w(c,v)\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshAdjacencyMatrix(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshAdjacencyMatrix(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_primal_graph",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshPrimalGraph(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh primal graph of adjacent vertices (u,v) where u,v are mesh vertices.\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_dual_graph",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes, std::int32_t flags) {
            return pbat::graph::MeshDualGraph(
                C,
                nNodes,
                static_cast<pbat::graph::EMeshDualGraphOptions>(flags));
        },
        nb::arg("C"),
        nb::arg("n")     = Index(-1),
        nb::arg("flags") = Index(0b111),
        "Compute the mesh dual graph of adjacency elements (ci,cj) where ci,cj are mesh elements.\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh\n"
        "flags (int): VertexAdjacency (0b001) | EdgeAdjacency (0b010) | FaceAdjacency (0b100) | "
        "All (0b111)\n");
    m.def(
        "reindex_mesh_by_connected_components",
        [](nb::DRef<MatrixX const> const& X, nb::DRef<IndexMatrixX const> const& E) {
            MatrixX Xcopy      = X;
            IndexMatrixX Ecopy = E;
            IndexVectorX XCC;
            IndexVectorX ECC;
            Eigen::Index nComponents =
                pbat::graph::ReindexMeshByConnectedComponents(Xcopy, Ecopy, XCC, ECC);
            return std::make_tuple(Xcopy, Ecopy, XCC, ECC, nComponents);
        },
        nb::arg("X"),
        nb::arg("E"),
        "Re-index mesh vertices and elements by connected components.\n"
        "Args:\n"
        "    X (np.ndarray): |# dims| x |# nodes| node position matrix\n"
        "    E (np.ndarray): |# nodes per element| x |# elements| element index matrix\n"
        "Returns:\n"
        "    X_reindexed (np.ndarray): `|# dims| x |# nodes|` Re-indexed node position matrix\n"
        "    E_reindexed (np.ndarray): `|# nodes per element| x |# elements|` Re-indexed element "
        "index matrix\n"
        "    XCC (np.ndarray): `|# nodes| x 1` node connected component index vector\n"
        "    ECC (np.ndarray): `|# elements| x 1` element connected component index vector\n"
        "    n_components (int): Number of connected components in the mesh");
}

} // namespace graph
} // namespace py
} // namespace pbat