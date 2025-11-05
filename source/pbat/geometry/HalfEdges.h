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
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param he Half-edge index
 * @return The incoming vertex index `v` of half-edge `he`
 */
template <common::CIndex TIndex = Index>
inline TIndex
IncomingVertex(Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F, TIndex he)
{
    return F(he % 3, he / 3);
}

/**
 * @brief Get the outgoing vertex of a half-edge in a triangle mesh.
 *
 * @note Particularly useful for iterating over adjacent vertices of a given vertex, using the
 * `VertexHalfEdgeAdjacency` structure.
 *
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param he Half-edge index
 * @param step Number of steps to advance around the face before getting the outgoing vertex
 * @return The outgoing vertex index `v` of half-edge `he`
 */
template <common::CIndex TIndex = Index>
inline TIndex OutgoingVertex(
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    TIndex he,
    int step = 0)
{
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
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param hei Half-edge index i
 * @param hej Half-edge index j
 * @return true if half-edges `hei` and `hej` are opposite half-edges
 * @return false otherwise
 */
template <common::CIndex TIndex = Index>
inline bool AreOppositeHalfEdges(
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    TIndex hei,
    TIndex hej)
{
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
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param n Number of vertices in the mesh
 * @return `(GVHEp, GVHEadj)` where
 *  - `GVHEp`: `|n+1|` prefix such that half-edges incoming to vertex `v` are in
 *           `GVHEadj[GVHEp[v]..GVHEp[v+1])`
 *  - `GVHEadj`: `|# half edges|` adjacencies
 */
template <common::CIndex TIndex = Index>
inline auto VertexHalfEdgeAdjacency(
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    TIndex n = TIndex(-1))
    -> std::pair<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
{
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
 * @brief Half-edge to adjacent faces mapping for a manifold triangle mesh.
 *
 * Builds a `2 x |3*# half edges|` array `GFHE` such that for half-edge `he=(i,j)`, `GFHE(0,he)` and
 * `GFHE(1,he)` are the two face indices incident to the directed edge `{i,j}`. The function assumes
 * a manifold mesh (each edge belongs to exactly two faces). If non-manifold or boundary
 * edges exist, the behavior is undefined.
 *
 * @tparam TIndex Index type (defaults to pbat::Index)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @return `2 x |3*# triangles|` matrix mapping half-edges to their two adjacent faces
 * @pre `F` is edge-manifold
 */
template <common::CIndex TIndex = Index>
inline auto
HalfEdgeFaceAdjacency(Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F)
    -> Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
{
    Eigen::Index const nFacets    = F.cols();
    Eigen::Index const nHalfEdges = 3 * nFacets;
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> GFHE(2, nHalfEdges);
    GFHE.setConstant(static_cast<TIndex>(-1));
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
    // Pair consecutive entries (assumes manifold edges => pairs of twins)
    for (TIndex k = 0; k < nHalfEdges; k += 2)
    {
        TIndex const hei = order(k);
        TIndex const hej = order(k + 1);
        TIndex const fi  = FaceOfHalfEdge(hei);
        TIndex const fj  = FaceOfHalfEdge(hej);
        GFHE(0, hei)     = fi;
        GFHE(1, hei)     = fj;
        GFHE(0, hej)     = fj;
        GFHE(1, hej)     = fi;
    }
    return GFHE;
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_HALFEDGES_H
