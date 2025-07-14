/**
 * @file MeshBoundary.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains functions to compute the boundary of a mesh.
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_MESH_BOUNDARY_H
#define PBAT_GEOMETRY_MESH_BOUNDARY_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Hash.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <tuple>
#include <unordered_map>

namespace pbat {
namespace geometry {

/**
 * @brief Obtains the boundary mesh of a simplex mesh.
 *
 * @note Only works for triangle (`C.rows()==3`) and tetrahedral (`C.rows()==4`) meshes.
 *
 * @tparam TIndex The index type used in the mesh (default: `Index`)
 * @param C The connectivity matrix of the mesh (i.e. the simplices)
 * @param n The number of vertices in the mesh. If -1, the number of vertices is computed from C.
 * @return A tuple containing the boundary vertices and the boundary facets
 */
template <common::CIndex TIndex = Index>
auto SimplexMeshBoundary(
    Eigen::Ref<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> const& C,
    TIndex n)
    -> std::tuple<
        Eigen::Vector<TIndex, Eigen::Dynamic>,
        Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.geometry.SimplexMeshBoundary");
    if (n < 0)
        n = C.maxCoeff() + 1;

    auto nSimplexNodes  = C.rows();
    auto nSimplexFacets = nSimplexNodes;
    if (nSimplexFacets < 3 or nSimplexFacets > 4)
    {
        std::string const what = fmt::format(
            "SimplexMeshBoundary expected triangle (3x|#elems|) or tetrahedral (4x|#elems|) input "
            "mesh, but got {}x{}",
            C.rows(),
            C.cols());
        throw std::invalid_argument(what);
    }
    auto nFacetNodes = nSimplexNodes - 1;
    auto nSimplices  = C.cols();
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> F(
        nFacetNodes,
        nSimplexFacets * nSimplices);
    for (TIndex c = 0; c < nSimplices; ++c)
    {
        // Tetrahedra
        if (nSimplexFacets == 4)
        {
            auto Fc = F.block<3, 4>(0, c * 4);
            Fc.col(0) << C(0, c), C(1, c), C(3, c);
            Fc.col(1) << C(1, c), C(2, c), C(3, c);
            Fc.col(2) << C(2, c), C(0, c), C(3, c);
            Fc.col(3) << C(0, c), C(2, c), C(1, c);
        }
        // Triangles
        if (nSimplexFacets == 3)
        {
            auto Fc = F.block<2, 3>(0, c * 3);
            Fc.col(0) << C(0, c), C(1, c);
            Fc.col(1) << C(1, c), C(2, c);
            Fc.col(2) << C(2, c), C(0, c);
        }
    }
    // Sort face indices to identify duplicates next
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> FS = F;
    for (TIndex f = 0; f < FS.cols(); ++f)
    {
        std::sort(FS.col(f).begin(), FS.col(f).end());
    }
    // Count face occurrences and pick out boundary facets
    auto fExtractBoundary = [&](auto const& FU) {
        TIndex nFacets{0};
        for (auto f = 0; f < FS.cols(); ++f)
            nFacets += (FU.at(FS.col(f)) == 1);
        Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> B(nFacetNodes, nFacets);
        for (auto f = 0, b = 0; f < F.cols(); ++f)
            if (FU.at(FS.col(f)) == 1)
                B.col(b++) = F.col(f);
        return B;
    };
    auto fExtractVertices = [&](Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> B) {
        auto begin = B.data();
        auto end   = B.data() + B.size();
        std::sort(begin, end);
        auto it                = std::unique(begin, end);
        auto nBoundaryVertices = std::distance(begin, it);
        Eigen::Vector<TIndex, Eigen::Dynamic> V(nBoundaryVertices);
        std::copy(begin, it, V.data());
        return V;
    };
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> B{};
    if (nSimplexFacets == 4)
    {
        std::unordered_map<IndexVector<3>, TIndex> FU{};
        FU.reserve(static_cast<std::size_t>(FS.cols()));
        for (TIndex f = 0; f < FS.cols(); ++f)
            ++FU[FS.col(f)];
        B = fExtractBoundary(FU);
    }
    if (nSimplexFacets == 3)
    {
        std::unordered_map<IndexVector<2>, TIndex> FU{};
        FU.reserve(static_cast<std::size_t>(FS.cols()));
        for (TIndex f = 0; f < FS.cols(); ++f)
            ++FU[FS.col(f)];
        B = fExtractBoundary(FU);
    }
    Eigen::Vector<TIndex, Eigen::Dynamic> V = fExtractVertices(B);
    return {V, B};
}

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_MESH_BOUNDARY_H
