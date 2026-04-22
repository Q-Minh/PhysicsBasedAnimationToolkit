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
        "    C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "    w: (np.ndarray): |#nodes per element|x|#elements| array of edge weights w(c,v)\n"
        "    n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshAdjacencyMatrix(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "    C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "    n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshAdjacencyMatrix(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "    C (np.ndarray): |# nodes per element|x|# elements| array of mesh elements\n"
        "    n (int): Number of nodes in the mesh");
    m.def(
        "mesh_primal_graph",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshPrimalGraph(C, nNodes);
        },
        nb::arg("C"),
        nb::arg("n") = Index(-1),
        "Compute the mesh primal graph of adjacent vertices (u,v) where u,v are mesh vertices.\n"
        "Args:\n"
        "    C (np.ndarray): |# nodes per element|x|# elements| array of mesh elements\n"
        "    n (int): Number of nodes in the mesh"
        "Returns:\n"
        "    scipy.sparse.csr_matrix: The |# nodes| x |# nodes| sparse adjacency matrix");
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
        "    C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "    n (int): Number of nodes in the mesh\n"
        "    flags (int): VertexAdjacency (0b001) | EdgeAdjacency (0b010) | FaceAdjacency (0b100) "
        "| All (0b111)\n"
        "Returns:\n"
        "    scipy.sparse.csr_matrix: The |# elements| x |# elements| sparse adjacency matrix");
    m.def(
        "sorted_connected_component_ordering",
        [](nb::DRef<MatrixX const> const& X, nb::DRef<IndexMatrixX const> const& E) {
            Eigen::Vector<Index, Eigen::Dynamic> XCC(X.cols());
            Eigen::Vector<Index, Eigen::Dynamic> ECC(E.cols());
            Eigen::Vector<Index, Eigen::Dynamic> Xordering(X.cols());
            Eigen::Vector<Index, Eigen::Dynamic> Eordering(E.cols());
            Eigen::Index nComponents =
                pbat::graph::SortedConnectedComponentOrdering(X, E, XCC, ECC, Xordering, Eordering);
            return std::make_tuple(Xordering, Eordering, XCC, ECC, nComponents);
        },
        nb::arg("X"),
        nb::arg("E"),
        "Compute the sorted connected component ordering of the mesh. The ordering is such that "
        "all nodes and elements belonging to the same connected component are grouped together, "
        "and such connected component groups are sorted, i.e. all `i1` and `e1` with the same "
        "component index `c` appear before all `i2` and `e2` with connected component index `d > "
        "c`.\n"
        "Args:\n"
        "    X (np.ndarray): |# dims| x |# nodes| node position matrix\n"
        "    E (np.ndarray): |# nodes per element| x |# elements| element index matrix\n"
        "Returns:\n"
        "    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: The tuple `(Xordering, "
        "Eordering, XCC, ECC, n_components)` where `Xordering` is a `|# nodes| x 1` node ordering "
        "vector s.t. `Xordering[i]` gives the new index of node `i` in the re-indexed mesh, "
        "`Eordering` is an `|# elements| x 1` element ordering vector s.t. `Eordering[e]` gives "
        "the new index of element `e` in the re-indexed mesh, `XCC` is an `|# nodes| x 1` node "
        "connected component index vector s.t. `XCC[i]` gives the connected component index of "
        "node `i` in the input mesh, `ECC` is an `|# elements| x 1` element connected component "
        "index vector s.t. `ECC[e]` gives the connected component index of element `e` in the "
        "input mesh, and `n_components` is the number of connected components in the mesh");
    m.def(
        "reindex_mesh_by_connected_components",
        [](nb::DRef<MatrixX const> const& X,
           nb::DRef<IndexMatrixX const> const& E,
           nb::DRef<IndexVectorX const> const& XCC,
           nb::DRef<IndexVectorX const> const& ECC,
           nb::DRef<IndexVectorX> Xordering,
           nb::DRef<IndexVectorX> Eordering) {
            MatrixX Xcopy        = X;
            IndexMatrixX Ecopy   = E;
            IndexVectorX XCCcopy = XCC;
            IndexVectorX ECCcopy = ECC;
            pbat::graph::ReindexMeshByConnectedComponents(
                Xcopy,
                Ecopy,
                XCCcopy,
                ECCcopy,
                Xordering,
                Eordering);
            return std::make_tuple(Xcopy, Ecopy, XCCcopy, ECCcopy);
        },
        nb::arg("X"),
        nb::arg("E"),
        nb::arg("XCC"),
        nb::arg("ECC"),
        nb::arg("Xordering"),
        nb::arg("Eordering"),
        "Re-index mesh vertices and elements by connected components. The input mesh X and E are "
        "re-indexed in-place such that nodes and elements belonging to the same connected "
        "component are grouped together, and such connected component groups are sorted, i.e. all "
        "`i1` and `e1` with the same component index `c` appear before all `i2` and `e2` with "
        "connected component index `d > c`.\n"
        "Args:\n"
        "    X (np.ndarray): |# dims| x |# nodes| node position matrix\n"
        "    E (np.ndarray): |# nodes per element| x |# elements| element index matrix\n"
        "    XCC (np.ndarray): |# nodes| x 1 node connected component index vector s.t. `XCC[i]` "
        "gives the connected component index of node `i` in the input mesh.\n"
        "    ECC (np.ndarray): |# elements| x 1 element connected component index vector s.t. "
        "`ECC[e]` gives the connected component index of element `e` in the input mesh\n"
        "    Xordering (np.ndarray): |# nodes| x 1 node ordering vector s.t. `Xordering[i]` gives "
        "the new "
        "index of node `i` in the re-indexed mesh\n"
        "    Eordering (np.ndarray): |# elements| x 1 element ordering vector s.t. `Eordering[e]` "
        "gives the new "
        "index of element `e` in the re-indexed mesh\n"
        "Returns:\n"
        "    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The tuple `(X_reindexed, "
        "E_reindexed, XCC_reindexed, ECC_reindexed)` where `X_reindexed` is a `|# dims| x |# "
        "nodes|` re-indexed node position matrix, `E_reindexed` is a `|# nodes per element| x |# "
        "elements|` re-indexed element index matrix, `XCC_reindexed` is a `|# nodes| x 1` "
        "re-indexed node connected component index vector, and `ECC_reindexed` is a `|# elements| "
        "x 1` re-indexed element connected component index vector\n");
    m.def(
        "reindex_mesh_by_connected_components",
        [](nb::DRef<MatrixX const> const& X, nb::DRef<IndexMatrixX const> const& E) {
            MatrixX Xcopy            = X;
            IndexMatrixX Ecopy       = E;
            Eigen::Index nComponents = pbat::graph::ReindexMeshByConnectedComponents(Xcopy, Ecopy);
            return std::make_tuple(Xcopy, Ecopy, nComponents);
        },
        nb::arg("X"),
        nb::arg("E"),
        "Re-index mesh vertices and elements by connected components. The input mesh X and E are "
        "re-indexed in-place such that nodes and elements belonging to the same connected "
        "component are grouped together, and such connected component groups are sorted, i.e. all "
        "`i1` and `e1` with the same component index `c` appear before all `i2` and `e2` with "
        "connected component index `d > c`.\n"
        "Args:\n"
        "    X (np.ndarray): |# dims| x |# nodes| node position matrix\n"
        "    E (np.ndarray): |# nodes per element| x |# elements| element index matrix\n"
        "Returns:\n"
        "    Tuple[np.ndarray, np.ndarray, int]: The tuple `(X_reindexed, E_reindexed, "
        "n_components)` where `X_reindexed` is a `|# dims| x |# nodes|` re-indexed node position "
        "matrix, `E_reindexed` is a `|# nodes per element| x |# elements|` re-indexed element "
        "index matrix, and `n_components` (int) is the number of connected components in the mesh");
}

} // namespace graph
} // namespace py
} // namespace pbat