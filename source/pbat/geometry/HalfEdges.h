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
 * @brief Build the vertex-to-half-edge adjacency for a triangle mesh.
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
    TIndex n)
    -> std::pair<Eigen::Vector<TIndex, Eigen::Dynamic>, Eigen::Vector<TIndex, Eigen::Dynamic>>
{
    Eigen::Index const nFacets    = F.cols();
    Eigen::Index const nHalfEdges = 3 * nFacets;
    Eigen::Vector<TIndex, Eigen::Dynamic> GVHEp(n + 1);
    GVHEp.setZero();
    // Count incoming half-edges per vertex (F stores incoming vertex of each half-edge)
    GVHEp(F.reshaped().array() + 1).array() += TIndex(1);
    // Prefix sum to get offsets
    std::inclusive_scan(GVHEp.data() + 1, GVHEp.data() + GVHEp.size(), GVHEp.data() + 1);
    // Adjacency: permutation sorting half-edges by incoming vertex
    Eigen::Vector<TIndex, Eigen::Dynamic> GVHEadj =
        common::ArgSort<TIndex>(static_cast<TIndex>(nHalfEdges), [&F](TIndex hei, TIndex hej) {
            return F(hei % 3, hei / 3) < F(hej % 3, hej / 3);
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
            TIndex fi  = hei / 3;
            TIndex fj  = hej / 3;
            TIndex via = F(hei % 3, fi);
            TIndex vib = F((hei + 1) % 3, fi);
            TIndex vja = F(hej % 3, fj);
            TIndex vjb = F((hej + 1) % 3, fj);
            return std::make_pair(std::min(via, vib), std::max(via, vib)) <
                   std::make_pair(std::min(vja, vjb), std::max(vja, vjb));
        });
    // Pair consecutive entries (assumes manifold edges => pairs of twins)
    for (TIndex k = 0; k < static_cast<TIndex>(nHalfEdges); k += 2)
    {
        TIndex const hei = order(k);
        TIndex const hej = order(k + 1);
        TIndex const fi  = hei / 3;
        TIndex const fj  = hej / 3;
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
