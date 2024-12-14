#ifndef PBAT_GRAPH_MESH_H
#define PBAT_GRAPH_MESH_H

#include "pbat/Aliases.h"

#include <concepts>

namespace pbat {
namespace graph {

template <class TDerivedE, class TDerivedW, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedW> const& w,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false)
{
    if (nNodes < 0)
        nNodes = E.maxCoeff() + TIndex(1);

    using AdjacencyMatrix = Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>;
    AdjacencyMatrix G(nNodes, E.cols());
    using IndexVectorType = Eigen::Vector<TIndex, Eigen::Dynamic>;
    G.reserve(IndexVectorType::Constant(E.cols(), static_cast<TIndex>(E.rows())));
    for (auto e = 0; e < E.cols(); ++e)
        for (auto i = 0; i < E.rows(); ++i)
            G.insert(E(i, e), e) = w(i, e);
    if (bVertexToElement)
        G = G.transpose();
    return G;
}

template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex> MeshAdjacencyMatrix(
    Eigen::DenseBase<TDerivedE> const& E,
    TIndex nNodes         = TIndex(-1),
    bool bVertexToElement = false)
{
    using WeightMatrixType = Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>;
    return MeshAdjacencyMatrix(
        E,
        WeightMatrixType::Ones(E.rows(), E.cols()),
        nNodes,
        bVertexToElement);
}

template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
MeshPrimalGraph(Eigen::DenseBase<TDerivedE> const& E, TIndex nNodes = TIndex(-1))
{
    auto const G = MeshAdjacencyMatrix(E, nNodes);
    return G * G.transpose();
}

template <class TDerivedE, std::integral TIndex = typename TDerivedE::Scalar>
Eigen::SparseMatrix<TIndex, Eigen::ColMajor, TIndex>
MeshDualGraph(Eigen::DenseBase<TDerivedE> const& E, TIndex nNodes = TIndex(-1))
{
    auto const G = MeshAdjacencyMatrix(E, nNodes);
    return G.transpose() * G;
}

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_MESH_H