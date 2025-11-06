/**
 * @file MultiMesh.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for multi-mesh contact representation.
 * @version 0.1
 * @date 2025-11-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_SIM_CONTACT_MULTIMESH_H
#define PBAT_SIM_CONTACT_MULTIMESH_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/geometry/MeshBoundary.h"

#include <Eigen/Core>
#include <numeric>
#include <utility>

namespace pbat::sim::contact {

/**
 * @brief Compute boundary triangulation of multi-(tetrahedral-)mesh, and preserve connected
 * component labeling.
 *
 * @tparam TDerivedT Type of tetrahedral element indices
 * @tparam TDerivedXCC Type of node connected component labels
 * @tparam TDerivedV Type of output vertex indices
 * @tparam TDerivedF Type of output face indices
 * @tparam TDerivedVP Type of vertex prefix
 * @tparam TDerivedFP Type of face prefix
 * @tparam TIndex Type of indices
 * @param T `4 x |# tetrahedra|` array of tetrahedral element indices
 * @param XCC `|# nodes|` array of node connected component labels
 * @param V `|# vertices|` array of output vertex indices
 * @param F `3 x |# faces|` array of output face indices
 * @param VP `|# connected components + 1| x 1` vertex prefix
 * @param FP `|# connected components + 1| x 1` face prefix
 */
template <
    class TDerivedT,
    class TDerivedXCC,
    class TDerivedV,
    class TDerivedF,
    class TDerivedVP,
    class TDerivedFP,
    common::CIndex TIndex = typename TDerivedT::Scalar>
void BoundaryTriangulation(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedV>& V,
    Eigen::DenseBase<TDerivedF>& F,
    Eigen::DenseBase<TDerivedVP>& VP,
    Eigen::DenseBase<TDerivedFP>& FP);

/**
 * @brief Compute boundary triangulation edges of multi-(tetrahedral-)mesh with adjacency
 * information, and preserve connected component labeling.
 *
 * @tparam TDerivedXCC Type of node connected component labels
 * @tparam TDerivedF::Scalar Type of indices
 * @param F `3 x |# faces|` array of boundary face indices
 * @param XCC `|# nodes|` array of node connected component labels
 * @param E `2 x |# edges|` array of output edge indices
 * @param EP `|# connected components + 1|` edge prefix
 * @param GVHEp `|# points + 1|` point to half-edge prefix
 * @param GVHEadj `|# half edges|` point to half-edge adjacency
 * @param GHEF `2 x |# half edges|` half-edge to face adjacency
 * @param EHE `2 x |# edges|` edge to half-edge adjacency
 */
template <
    class TDerivedF,
    class TDerivedXCC,
    class TDerivedE,
    class TDerivedEP,
    class TDerivedGVHEp,
    class TDerivedGVHEadj,
    class TDerivedGHEF,
    class TDerivedEHE,
    common::CIndex TIndex = typename TDerivedE::Scalar>
void BoundaryTriangulationEdges(
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedE>& E,
    Eigen::DenseBase<TDerivedEP>& EP,
    Eigen::DenseBase<TDerivedGVHEp>& GVHEp,
    Eigen::DenseBase<TDerivedGVHEadj>& GVHEadj,
    Eigen::DenseBase<TDerivedGHEF>& GHEF,
    Eigen::DenseBase<TDerivedEHE>& EHE);

template <
    class TDerivedT,
    class TDerivedXCC,
    class TDerivedV,
    class TDerivedF,
    class TDerivedVP,
    class TDerivedFP,
    common::CIndex TIndex>
void BoundaryTriangulation(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedV>& V,
    Eigen::DenseBase<TDerivedF>& F,
    Eigen::DenseBase<TDerivedVP>& VP,
    Eigen::DenseBase<TDerivedFP>& FP)
{
    static_assert(
        TDerivedT::RowsAtCompileTime == 4,
        "Element type must be tetrahedral (4 nodes per element).");
    auto const nNodes = XCC.size();
    std::tie(V, F)    = geometry::SimplexMeshBoundary(T, static_cast<TIndex>(nNodes));
    VP.setZero();
    FP.setZero();
    // Count connected component occurrences in VP[1:] and FP[1:]
    VP(XCC(V.reshaped()).array() + 1).array() += TIndex(1);
    FP(XCC(F.row(0)).array() + 1).array() += TIndex(1);
    // Compute prefix sums
    std::inclusive_scan(VP.begin() + 1, VP.end(), VP.begin() + 1);
    std::inclusive_scan(FP.begin() + 1, FP.end(), FP.begin() + 1);
}

template <
    class TDerivedF,
    class TDerivedXCC,
    class TDerivedE,
    class TDerivedEP,
    class TDerivedGVHEp,
    class TDerivedGVHEadj,
    class TDerivedGHEF,
    class TDerivedEHE,
    common::CIndex TIndex>
void BoundaryTriangulationEdges(
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedE>& E,
    Eigen::DenseBase<TDerivedEP>& EP,
    Eigen::DenseBase<TDerivedGVHEp>& GVHEp,
    Eigen::DenseBase<TDerivedGVHEadj>& GVHEadj,
    Eigen::DenseBase<TDerivedGHEF>& GHEF,
    Eigen::DenseBase<TDerivedEHE>& EHE)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "Face type must be triangular (3 nodes per face).");
    TIndex const nPoints     = static_cast<TIndex>(XCC.size());
    std::tie(GVHEp, GVHEadj) = geometry::VertexHalfEdgeAdjacency(F, nPoints);
    GHEF                     = geometry::HalfEdgeFaceAdjacency(F);
    EHE                      = geometry::EdgeHalfEdgeAdjacency(F, GHEF.bottomRows<2>());
    E                        = geometry::Edges(F, EHE.bottomRows<2>());
    EP.setZero();
    // Count connected component occurrences in EP[1:]
    EP(XCC(E.row(0)).array() + 1).array() += TIndex(1);
    // Compute prefix sums
    std::inclusive_scan(EP.begin() + 1, EP.end(), EP.begin() + 1);
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIMESH_H
