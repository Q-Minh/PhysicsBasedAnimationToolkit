/**
 * @file HalfEdges.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Utilities for half-edge style adjacency on triangle meshes.
 * @version 0.1
 * @date 2025-11-04
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_HALFEDGES_H
#define PBAT_GEOMETRY_HALFEDGES_H

#include "pbat/Aliases.h"
#include "pbat/common/ArgSort.h"
#include "pbat/common/Concepts.h"

#include <algorithm>
#include <numeric>
#include <utility>

namespace pbat {
namespace geometry {

/**
 * @brief Get the incoming vertex of a half-edge in a triangle mesh.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param he Half-edge index
 * @return The incoming vertex index `v` of half-edge `he`
 */
template <class TDerivedF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline TIndex IncomingVertex(Eigen::DenseBase<TDerivedF> const& F, TIndex he)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    return F(he % 3, he / 3);
}

/**
 * @brief Get the outgoing vertex of a half-edge in a triangle mesh.
 *
 * @note Particularly useful for iterating over adjacent vertices of a given vertex, using the
 * `VertexHalfEdgeAdjacency` structure.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param he Half-edge index
 * @param step Number of steps to advance around the face before getting the outgoing vertex
 * @return The outgoing vertex index `v` of half-edge `he`
 */
template <class TDerivedF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline TIndex OutgoingVertex(Eigen::DenseBase<TDerivedF> const& F, TIndex he, int step = 0)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    return F((he + step + 1) % 3, he / 3);
}

/**
 * @brief Get the next half-edge in the same triangle.
 *
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param he Half-edge index
 * @return The next half-edge index in the same triangle
 */
template <common::CIndex TIndex = Index>
inline TIndex NextHalfEdge(TIndex he)
{
    return (he / 3) * 3 + (he + 1) % 3;
}

/**
 * @brief Get the first half-edge of a face.
 *
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param f Face index
 * @return First half-edge index of face `f`
 */
template <common::CIndex TIndex = Index>
inline TIndex FirstHalfEdgeOfFace(TIndex f)
{
    return f * 3;
}

/**
 * @brief Get the face index of a half-edge.
 *
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param he Half-edge index
 * @return Face index of half-edge `he`
 */
template <common::CIndex TIndex = Index>
inline TIndex FaceOfHalfEdge(TIndex he)
{
    return he / 3;
}

/**
 * @brief Tests if two half-edges are oppositely oriented edges of the same undirected edge.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param hei Half-edge index i
 * @param hej Half-edge index j
 * @return true if half-edges `hei` and `hej` are opposite half-edges
 * @return false otherwise
 */
template <class TDerivedF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline bool AreOppositeHalfEdges(Eigen::DenseBase<TDerivedF> const& F, TIndex hei, TIndex hej)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    TIndex const fi  = hei / 3;
    TIndex const fj  = hej / 3;
    TIndex const via = F(hei % 3, fi);
    TIndex const vib = F((hei + 1) % 3, fi);
    TIndex const vja = F(hej % 3, fj);
    TIndex const vjb = F((hej + 1) % 3, fj);
    return (via == vjb) and (vib == vja);
}

/**
 * @brief Build the (incoming-)vertex-to-half-edge adjacency for a triangle mesh.
 *
 * A half-edge is represented implicitly by the pair `(F(ei,f), F((ei+1)%3, f))` for `ei` in
 * `{0,1,2}` and a triangle `f`. This constructs a CSR-like representation (prefix, adjacency) that
 * groups half-edges by their incoming vertex (the first endpoint of the directed half-edge).
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param n Number of vertices in the mesh
 * @return `(GVHEp, GVHEadj)` where
 *  - `GVHEp`: `|n+1|` prefix such that half-edges incoming to vertex `v` are in
 *           `GVHEadj[GVHEp[v]..GVHEp[v+1])`
 *  - `GVHEadj`: `|# half edges|` adjacencies
 */
template <class TDerivedF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline auto VertexHalfEdgeAdjacency(Eigen::DenseBase<TDerivedF> const& F, TIndex n = TIndex(-1))
    -> std::pair<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    Eigen::Index const nFacets    = F.cols();
    Eigen::Index const nHalfEdges = 3 * nFacets;
    if (n < 0)
        n = F.maxCoeff() + 1;
    Eigen::Vector<TIndex, Eigen::Dynamic> GVHEp(n + 1);
    GVHEp.setZero();
    // Count incoming half-edges per vertex (F stores incoming vertex of each half-edge) and store
    // in GVHEp starting from index 1
    GVHEp(F.reshaped().array() /* vertex indices */ + 1).array() += TIndex(1);
    // Prefix sum to get offsets
    std::inclusive_scan(GVHEp.data() + 1, GVHEp.data() + GVHEp.size(), GVHEp.data() + 1);
    // Compute adjacency by essentially grouping (i.e. sorting) half-edges by incoming vertex
    Eigen::Vector<TIndex, Eigen::Dynamic> GVHEadj =
        common::ArgSort<TIndex>(static_cast<TIndex>(nHalfEdges), [&F](TIndex hei, TIndex hej) {
            return IncomingVertex(F, hei) < IncomingVertex(F, hej);
        });
    return {GVHEp, GVHEadj};
}

/**
 * @brief Half-edge to adjacent faces mapping for a triangle mesh.
 *
 * Builds a `2 x |3*# half edges|` array `GFHE` such that for half-edge `he=(i,j)`, `GFHE(0,he)` and
 * `GFHE(1,he)` are the two face indices incident to the directed edge `{i,j}`. `GFHE(1,he) == -1`
 * is used to indicate no adjacent face for a boundary edge.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @return `2 x |3*# triangles|` matrix mapping half-edges (columns) to their two adjacent faces
 * (rows)
 * @pre `F` is edge-manifold
 */
template <class TDerivedF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline auto HalfEdgeFaceAdjacency(Eigen::DenseBase<TDerivedF> const& F)
    -> Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    Eigen::Index const nFacets    = F.cols();
    Eigen::Index const nHalfEdges = 3 * nFacets;
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> GHEF(2, nHalfEdges);
    GHEF.setConstant(static_cast<TIndex>(-1));
    // Sort half-edges by undirected edge key (min(i,j), max(i,j))
    auto order =
        common::ArgSort<TIndex>(static_cast<TIndex>(nHalfEdges), [&F](TIndex hei, TIndex hej) {
            TIndex fi  = FaceOfHalfEdge(hei);
            TIndex fj  = FaceOfHalfEdge(hej);
            TIndex via = IncomingVertex(F, hei);
            TIndex vib = OutgoingVertex(F, hei);
            TIndex vja = IncomingVertex(F, hej);
            TIndex vjb = OutgoingVertex(F, hej);
            return std::make_pair(std::min(via, vib), std::max(via, vib)) <
                   std::make_pair(std::min(vja, vjb), std::max(vja, vjb));
        });
    // Pair consecutive entries
    TIndex k;
    for (k = 0; k < nHalfEdges - 1;)
    {
        TIndex const hei = order(k);
        TIndex const hej = order(k + 1);
        if (AreOppositeHalfEdges(F, hei, hej))
        {
            // Interior edge
            TIndex const fi = FaceOfHalfEdge(hei);
            TIndex const fj = FaceOfHalfEdge(hej);
            GHEF(0, hei)    = fi;
            GHEF(1, hei)    = fj;
            GHEF(0, hej)    = fj;
            GHEF(1, hej)    = fi;
            k += 2;
        }
        else
        {
            // Boundary edge
            TIndex const fi = FaceOfHalfEdge(hei);
            GHEF(0, hei)    = fi;
            GHEF(1, hei)    = -1;
            ++k;
        }
    }
    if (k < nHalfEdges)
    {
        // Boundary edge
        TIndex const hei = order(k);
        TIndex const fi  = FaceOfHalfEdge(hei);
        GHEF(0, hei)     = fi;
        GHEF(1, hei)     = -1;
    }
    return GHEF;
}

/**
 * @brief Get the opposite half-edge of a given half-edge in a triangle mesh.
 *
 * If the half-edge is a boundary edge, returns -1.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TDerivedGHEF Derived type of GHEF
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param hei Half-edge index
 * @param GHEF `2 x |3*# half edges|` half-edge to face adjacency matrix
 * @return Opposite half-edge index of half-edge `hei`, or -1 if none exists
 */
template <class TDerivedF, class TDerivedGHEF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline TIndex OppositeHalfEdge(
    Eigen::DenseBase<TDerivedF> const& F,
    TIndex hei,
    Eigen::DenseBase<TDerivedGHEF> const& GHEF)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    static_assert(
        TDerivedGHEF::RowsAtCompileTime == 2,
        "GHEF must have 2 rows representing half-edge to face adjacency.");
    TIndex const fi = GHEF(0, hei);
    TIndex const fj = GHEF(1, hei);
    if (fj == -1)
        return TIndex(-1); // No opposite half-edge (boundary edge)
    TIndex const via = IncomingVertex(F, hei);
    TIndex const vib = OutgoingVertex(F, hei);
    // Find half-edge in face fj that goes from vib to via, i.e. whose incoming vertex is vib
    TIndex ej  = /*(F(0, fj) == vib)*0 + */ (F(1, fj) == vib) * 1 + (F(2, fj) == vib) * 2;
    TIndex hej = fj * 3 + ej;
    return hej;
}

/**
 * @brief Build edge to half-edge adjacency for a triangle mesh.
 * @note This representation is useful for undirected edge processing, e.g. by looping over columns
 * of the output adjacency matrix.
 * @tparam TDerivedF Derived type of F
 * @tparam TDerivedGHEF Derived type of GHEF
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GHEF `2 x |3*# half edges|` half-edge to face adjacency matrix
 * @return `2 x |# edges|` matrix mapping edges (columns) to their two adjacent half-edges (rows,
 * where -1 indicates no opposite half-edge)
 */
template <class TDerivedF, class TDerivedGHEF, common::CIndex TIndex = typename TDerivedF::Scalar>
inline auto EdgeHalfEdgeAdjacency(
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedGHEF> const& GHEF) -> Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    static_assert(
        TDerivedGHEF::RowsAtCompileTime == 2,
        "GHEF must have 2 rows representing half-edge to face adjacency.");
    Eigen::Index nEdges{0};
    for (auto he = 0; he < GHEF.cols(); ++he)
    {
        bool bIsBoundaryEdge                   = (GHEF(1, he) == -1);
        bool bIsLexicographicallyFirstHalfEdge = IncomingVertex(F, he) < OutgoingVertex(F, he);
        nEdges += (bIsBoundaryEdge or bIsLexicographicallyFirstHalfEdge);
    }
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> EHE(2, nEdges);
    for (auto he = 0, e = 0; he < GHEF.cols(); ++he)
    {
        bool bIsBoundaryEdge                   = (GHEF(1, he) == -1);
        bool bIsLexicographicallyFirstHalfEdge = IncomingVertex(F, he) < OutgoingVertex(F, he);
        if (bIsBoundaryEdge)
        {
            EHE(0, e) = he;
            EHE(1, e) = -1;
            ++e;
        }
        else if (bIsLexicographicallyFirstHalfEdge)
        {
            EHE(0, e) = he;
            EHE(1, e) = OppositeHalfEdge(F, he, GHEF);
            ++e;
        }
    }
    return EHE;
}

/**
 * @brief Build the undirected edge list for a triangle mesh from its half-edge representation.
 *
 * @tparam TDerivedF Derived type of F
 * @tparam TDerivedEHE Derived type of EHE
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param EHE `2 x |# edges|` edge to half-edge adjacency matrix
 * @return `2 x |# edges|` matrix of undirected edge vertex indices
 */
template <class TDerivedF, class TDerivedEHE, common::CIndex TIndex = typename TDerivedF::Scalar>
inline auto Edges(Eigen::DenseBase<TDerivedF> const& F, Eigen::DenseBase<TDerivedEHE> const& EHE)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    static_assert(
        TDerivedEHE::RowsAtCompileTime == 2,
        "EHE must have 2 rows representing edge to half-edge adjacency.");
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> E(2, EHE.cols());
    for (Eigen::Index e = 0; e < EHE.cols(); ++e)
    {
        TIndex hei = EHE(0, e);
        E(0, e)    = IncomingVertex(F, hei);
        E(1, e)    = OutgoingVertex(F, hei);
    }
    return E;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_HALFEDGES_H
