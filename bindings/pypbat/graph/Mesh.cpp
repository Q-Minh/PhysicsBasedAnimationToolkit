#include "Mesh.h"

#include <pbat/graph/Mesh.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace graph {

void BindMesh(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C,
           Eigen::Ref<IndexVectorX const> const& w,
           Index nNodes) { return pbat::graph::MeshAdjacencyMatrix(C, w, nNodes); },
        pyb::arg("C"),
        pyb::arg("w"),
        pyb::arg("n") = Index(-1),
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
        pyb::arg("C"),
        pyb::arg("w"),
        pyb::arg("n") = Index(-1),
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
        pyb::arg("C"),
        pyb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_adjacency_matrix",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshAdjacencyMatrix(C, nNodes);
        },
        pyb::arg("C"),
        pyb::arg("n") = Index(-1),
        "Compute the mesh element to vertex adjacency graph (c,v) for c in C and v in [0,n).\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh");
    m.def(
        "mesh_primal_graph",
        [](Eigen::Ref<IndexMatrixX const> const& C, Index nNodes) {
            return pbat::graph::MeshPrimalGraph(C, nNodes);
        },
        pyb::arg("C"),
        pyb::arg("n") = Index(-1),
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
        pyb::arg("C"),
        pyb::arg("n")     = Index(-1),
        pyb::arg("flags") = Index(0b111),
        "Compute the mesh dual graph of adjacency elements (ci,cj) where ci,cj are mesh elements.\n"
        "Args:\n"
        "C (np.ndarray): |#nodes per element|x|#elements| array of mesh elements\n"
        "n (int): Number of nodes in the mesh\n"
        "flags (int): VertexAdjacency (0b001) | EdgeAdjacency (0b010) | FaceAdjacency (0b100) | "
        "All (0b111)\n");
}

} // namespace graph
} // namespace py
} // namespace pbat