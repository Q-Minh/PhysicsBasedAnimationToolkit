#ifndef PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
#define PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Permute.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/graph/BreadthFirstSearch.h"
#include "pbat/graph/ConnectedComponents.h"
#include "pbat/graph/Mesh.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>
#include <ranges>

namespace pbat::sim::contact {

/**
 * @brief Multibody Tetrahedral Mesh System
 *
 * Holds a common representation of a multibody system composed of tetrahedral meshes.
 */
template <common::CIndex TIndex = Index>
struct MultibodyTetrahedralMeshSystem
{
    using IndexType = TIndex; ///< Index type used in the multibody system

    MultibodyTetrahedralMeshSystem() = default;
    /**
     * @brief Construct a new Multibody Tetrahedral Mesh System object
     * @tparam TDerivedE Eigen type of the input tetrahedral element matrix
     * @param X `3 x |# mesh vertices|` matrix of vertex positions
     * @param T `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity
     * @post The input mesh vertex positions and element indices will be sorted by body.
     */
    template <common::CArithmetic TScalar = Scalar>
    void Construct(
        Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic>> X,
        Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T);
    /**
     * @brief Get the number of bodies in the multibody system
     * @return The number of bodies
     */
    Eigen::Index NumberOfBodies() const { return VP.size() - 1; }
    /**
     * @brief Get vertices of body `o`
     * @param o Index of the body
     * @return `|# contact vertices of body o| x 1` indices into mesh vertices
     */
    auto VerticesOf(IndexType o) const { return V.segment(VP[o], VP[o + 1] - VP[o]); }
    /**
     * @brief Get edges of body `o`
     * @param o Index of the body
     * @return `2 x |# contact edges of body o|` edges into mesh vertices
     */
    auto EdgesOf(IndexType o) const { return E.middleCols(EP[o], EP[o + 1] - EP[o]); }
    /**
     * @brief Get triangles of body `o`
     * @param o Index of the body
     * @return `3 x |# contact triangles of body o|` triangles into mesh vertices
     */
    auto TrianglesOf(IndexType o) const { return F.middleCols(FP[o], FP[o + 1] - FP[o]); }
    /**
     * @brief Get tetrahedra of body `o`
     * @tparam TDerivedT Eigen type of the input tetrahedral mesh
     * @param o Index of the body
     * @param T `4 x |# tetrahedra|` tetrahedral mesh elements/connectivity
     * @return `4 x |# contact tetrahedra of body o|` tetrahedra into input tetrahedral mesh `T`
     */
    template <class TDerivedT>
    auto TetrahedraOf(IndexType o, Eigen::DenseBase<TDerivedT> const& T) const
    {
        return T.middleCols(TP[o], TP[o + 1] - TP[o]);
    }
    /**
     * @brief Get vertex positions of body `o`
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @param o Index of the body
     * @param X `3 x |# mesh vertices|` matrix of vertex positions
     * @return `3 x |# contact vertices of body o|` matrix of vertex positions
     */
    template <class TDerivedX>
    auto VertexPositionsOf(IndexType o, Eigen::DenseBase<TDerivedX> const& X) const
    {
        return X(Eigen::placeholders::all, V.segment(VP[o], VP[o + 1] - VP[o]));
    }
    /**
     * @brief Get the body associated with vertex `v`
     * @param v Index of the vertex
     * @return Body index of vertex `v`
     */
    auto BodyOfVertex(IndexType v) const { return CC[V[v]]; }
    /**
     * @brief Get the body associated with edge `e`
     * @param e Index of the edge
     * @return Body index of edge `e`
     */
    auto BodyOfEdge(IndexType e) const { return CC[E(0, e)]; }
    /**
     * @brief Get the body associated with triangle `f`
     * @param f Index of the triangle
     * @return Body index of triangle `f`
     */
    auto BodyOfTriangle(IndexType f) const { return CC[F(0, f)]; }
    /**
     * @brief Get the body associated with tetrahedron `t`
     * @param t Index of the tetrahedron
     * @return Body index of tetrahedron `t`
     */
    template <class TDerivedT>
    auto BodyOfTetrahedron(IndexType t, Eigen::DenseBase<TDerivedT> const& T) const
    {
        return CC[T(0, t)];
    }

    Eigen::Vector<TIndex, Eigen::Dynamic>
        V; ///< `|# contact vertices| x 1` indices into mesh vertices
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
        E; ///< `2 x |# contact edges|` edges into mesh vertices
    Eigen::Matrix<TIndex, 3, Eigen::Dynamic>
        F; ///< `3 x |# contact triangles|` triangles into mesh vertices

    Eigen::Vector<TIndex, Eigen::Dynamic>
        VP; ///< `|# bodies + 1| x 1` prefix sum of vertex pointers into `V`
    Eigen::Vector<TIndex, Eigen::Dynamic>
        EP; ///< `|# bodies + 1| x 1` prefix sum of edge pointers into `E`
    Eigen::Vector<TIndex, Eigen::Dynamic>
        FP; ///< `|# bodies + 1| x 1` prefix sum of triangle pointers into `F`
    Eigen::Vector<TIndex, Eigen::Dynamic> TP; ///< `|# bodies + 1| x 1` prefix sum of tetrahedron
                                              ///< pointers into input tetrahedral mesh `T`

    Eigen::Vector<TIndex, Eigen::Dynamic> CC; ///< `|# mesh vertices| x 1` connected component map
};

template <common::CIndex TIndex>
template <common::CArithmetic TScalar>
inline void MultibodyTetrahedralMeshSystem<TIndex>::Construct(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic>> X,
    Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyTetrahedralMeshSystem.Construct");
    IndexType const nNodes    = static_cast<IndexType>(X.cols());
    IndexType const nElements = static_cast<IndexType>(T.cols());
    // 1. Compute the mesh's dual graph over tets
    Eigen::SparseMatrix<IndexType, Eigen::ColMajor, IndexType> const EG =
        graph::MeshDualGraph(T, nNodes, graph::EMeshDualGraphOptions::All);
    // 2. Compute the connected components of the mesh
    graph::BreadthFirstSearch<IndexType> bfs(nElements);
    Eigen::Vector<IndexType, Eigen::Dynamic> ECC(nElements);
    ECC.setConstant(IndexType(-1));
    IndexType const nComponents = graph::ConnectedComponents<IndexType>(
        Eigen::Map<Eigen::Vector<IndexType, Eigen::Dynamic> const>(
            EG.outerIndexPtr(),
            EG.outerSize() + 1),
        Eigen::Map<Eigen::Vector<IndexType, Eigen::Dynamic> const>(
            EG.innerIndexPtr(),
            EG.nonZeros()),
        ECC,
        bfs);
    // 3. Transfer the element connected components to the mesh vertex connected component map
    CC.setConstant(nNodes, IndexType(-1));
    auto verticesToElements =
        Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(nElements, 0, nElements - 1)
            .replicate<1, 4>()
            .transpose()
            .reshaped(); // `4 x |# elements|` matrix `[[0,0,0,0], [1,1,1,1], ...,
                         // [nElements-1,nElements-1,nElements-1,nElements-1]]`
    CC(T.reshaped()) = ECC(verticesToElements);
    // 4. Sort the tets by connected component
    Eigen::Vector<IndexType, Eigen::Dynamic> Eordering =
        common::ArgSort<IndexType>(nElements, [&](IndexType ei, IndexType ej) {
            return ECC[ei] < ECC[ej];
        });
    for (auto r = 0; r < T.rows(); ++r)
        common::Permute(T.row(r).begin(), T.row(r).end(), Eordering.begin());
    common::Permute(ECC.begin(), ECC.end(), Eordering.begin());
    // 5. Sort vertices by connected component
    Eigen::Vector<IndexType, Eigen::Dynamic> Xordering =
        common::ArgSort<IndexType>(nNodes, [&](IndexType i, IndexType j) { return CC[i] < CC[j]; });
    for (auto d = 0; d < X.rows(); ++d)
        common::Permute(X.row(d).begin(), X.row(d).end(), Xordering.begin());
    common::Permute(CC.begin(), CC.end(), Xordering.begin());
    // 6. Re-index tet vertices to match the sorted order
    T.reshaped() = Xordering(T.reshaped());
    // 7. Compute boundary mesh, and note that V and F will already be sorted by connected
    // component, because we have re-indexed T and X
    std::tie(V, F) = geometry::SimplexMeshBoundary<IndexType>(T, nNodes);
    // 8. Compute edges from triangles (edges are also sorted by connected component, since
    // triangles are)
    auto const nEdges =
        F.size() / 2; // Boundary (triangle) mesh of tetrahedral mesh must be manifold+watertight
    E.resize(2, nEdges);
    auto const nTriangles = F.cols();
    for (auto f = 0, e = 0; f < nTriangles; ++f)
    {
        for (auto k = 0; k < 3; ++k)
        {
            auto i = F(k, f);
            auto j = F((k + 1) % 3, f);
            // De-duplicate since every edge is counted twice in the loop
            if (i < j)
            {
                E(0, e) = i;
                E(1, e) = j;
                ++e;
            }
        }
    }
    // 9. Compute the prefix sums for vertices, edges, triangles, and tetrahedra
    VP.setZero(nComponents + 1);
    EP.setZero(nComponents + 1);
    FP.setZero(nComponents + 1);
    TP.setZero(nComponents + 1);
    IndexType const nContactVertices = static_cast<IndexType>(V.size());
    IndexType const nContactEdges    = static_cast<IndexType>(E.cols());
    IndexType const nContactFaces    = static_cast<IndexType>(F.cols());
    for (IndexType o = 0; o < nComponents; ++o)
    {
        // Count vertices of body o
        auto& vosum = VP(o + 1);
        vosum       = VP(o);
        while (vosum < nContactVertices and CC(V(vosum)) == o)
            ++vosum;
        // Count edges of body o
        auto& eosum = EP(o + 1);
        eosum       = EP(o);
        while (eosum < nContactEdges and CC(E(0, eosum)) == o)
            ++eosum;
        // Count faces of body o
        auto& fosum = FP(o + 1);
        fosum       = FP(o);
        while (fosum < nContactFaces and CC(F(0, fosum)) == o)
            ++fosum;
        // Count tetrahedra of body o
        auto& tosum = TP(o + 1);
        tosum       = TP(o);
        while (tosum < nElements and CC(T(0, tosum)) == o)
            ++tosum;
    }
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H