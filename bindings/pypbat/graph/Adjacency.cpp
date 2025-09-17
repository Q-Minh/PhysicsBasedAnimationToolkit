#include "Adjacency.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/graph/Adjacency.h>

namespace pbat {
namespace py {
namespace graph {

void BindAdjacency(nanobind::module_& m)
{
    namespace nb = nanobind;
    m.def(
        "map_to_adjacency",
        [](Eigen::Ref<IndexVectorX const> const& p, Index n) {
            return pbat::graph::MapToAdjacency(p, n);
        },
        nb::arg("p"),
        nb::arg("n") = Index(-1),
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
        nb::arg("lil"),
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
            nb::arg("A"));
    });
}

} // namespace graph
} // namespace py
} // namespace pbat