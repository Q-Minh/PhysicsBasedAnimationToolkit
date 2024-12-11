#include "Adjacency.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/graph/Adjacency.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace graph {

void BindAdjacency(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "map_to_adjacency",
        [](Eigen::Ref<IndexVectorX const> const& p, Index n) {
            return pbat::graph::MapToAdjacency(p, n);
        },
        pyb::arg("p"),
        pyb::arg("n") = Index(-1),
        "Computes the adjacency list (ptr, adj) in sparse compressed format for a map p: V -> P "
        "for vertices V and partitions P.\n"
        "Args:\n"
        "p (np.ndarray): A partition/cluster/parent/etc. map that takes vertices to their "
        "corresponding partition/cluster/parent/etc.\n"
        "n (int): The number of partitions/clusters/parents/etc.");

    m.def(
        "lil_to_adjacency",
        [](std::vector<std::vector<Index>> const& lil) {
            return pbat::graph::ListOfListsToAdjacency(lil);
        },
        pyb::arg("lil"),
        "Computes the adjacency list (ptr, adj) in sparse compressed format for vertex partitions "
        "in list of lists format.\n"
        "Args:\n"
        "lil (list[list[int]]): A list of lists of vertices");

    pbat::common::ForTypes<
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor>,
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor>,
        Eigen::SparseMatrix<Index, Eigen::ColMajor>,
        Eigen::SparseMatrix<Index, Eigen::RowMajor>,
        Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>,
        Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Index>,
        Eigen::SparseMatrix<Index, Eigen::ColMajor, Index>,
        Eigen::SparseMatrix<Index, Eigen::RowMajor, Index>>([&]<class SparseMatrixType>() {
        m.def(
            "matrix_to_adjacency",
            [](SparseMatrixType const& A) { return pbat::graph::MatrixToAdjacency(A); },
            pyb::arg("A"));
    });
}

} // namespace graph
} // namespace py
} // namespace pbat