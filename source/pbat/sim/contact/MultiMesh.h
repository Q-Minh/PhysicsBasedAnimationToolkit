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
#include "pbat/io/Archive.h"

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
 * @param GXV `|# points| x 1` point to vertex mapping
 */
template <
    class TDerivedT,
    class TDerivedXCC,
    class TDerivedV,
    class TDerivedF,
    class TDerivedVP,
    class TDerivedFP,
    class TDerivedGXV,
    common::CIndex TIndex = typename TDerivedT::Scalar>
void BoundaryTriangulation(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedV>& V,
    Eigen::DenseBase<TDerivedF>& F,
    Eigen::DenseBase<TDerivedVP>& VP,
    Eigen::DenseBase<TDerivedFP>& FP,
    Eigen::DenseBase<TDerivedGXV>& GXV);

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

/**
 * @brief Storage type for a multi-(triangle-)mesh contact simulation representation.
 *
 * @tparam TIndex Type of indices
 */
template <common::CIndex TIndex>
struct MultiMesh
{
    Eigen::Vector<TIndex, Eigen::Dynamic> V;    ///< `|# vertices| x 1` (surface) vertex indices
    Eigen::Matrix<TIndex, 3, Eigen::Dynamic> F; ///< `3 x |# faces|` face indices
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> E; ///< `2 x |# edges|` edge indices
    Eigen::Vector<TIndex, Eigen::Dynamic> VP; ///< `|# connected components + 1| x 1` vertex prefix
    Eigen::Vector<TIndex, Eigen::Dynamic> FP; ///< `|# connected components + 1| x 1` face prefix
    Eigen::Vector<TIndex, Eigen::Dynamic> EP; ///< `|# connected components + 1| x 1` edge prefix
    Eigen::Vector<TIndex, Eigen::Dynamic> GVHEp; ///< `|# points + 1| x 1` point to half-edge prefix
    Eigen::Vector<TIndex, Eigen::Dynamic>
        GVHEadj; ///< `|# half edges| x 1` point to half-edge adjacency
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic>
        GHEF; ///< `2 x |# half edges|` half-edge to face adjacency
    Eigen::Matrix<TIndex, 2, Eigen::Dynamic> EHE; ///< `2 x |# edges|` edge to half-edge adjacency
    Eigen::Vector<TIndex, Eigen::Dynamic>
        GXV; ///< `|# points| x 1` point to vertex mapping, with `-1` for non-vertices
    /**
     * @brief Default construct a new Multi Mesh object
     */
    MultiMesh() = default;
    /**
     * @brief Construct a new Multi Mesh object from a tetrahedral mesh `T` with node connected
     * component labels `XCC`.
     *
     * @tparam TDerivedT Tetrahedral matrix type
     * @tparam TDerivedXCC Connected component label vector type
     * @param T `4 x |# tetrahedra|` array of tetrahedral element indices
     * @param XCC `|# nodes| x 1` array of node connected component labels
     * @param nComponents Number of connected components (optional)
     */
    template <class TDerivedT, class TDerivedXCC>
    MultiMesh(
        Eigen::DenseBase<TDerivedT> const& T,
        Eigen::DenseBase<TDerivedXCC> const& XCC,
        Eigen::Index nComponents = -1);
    /**
     * @brief Construct a new Multi Mesh object from a tetrahedral mesh `T` with node connected
     * component labels `XCC`.
     *
     * @tparam TDerivedT Tetrahedral matrix type
     * @tparam TDerivedXCC Connected component label vector type
     * @param T `4 x |# tetrahedra|` array of tetrahedral element indices
     * @param XCC `|# nodes| x 1` array of node connected component labels
     * @param nComponents Number of connected components (optional)
     */
    template <class TDerivedT, class TDerivedXCC>
    void ConstructFromTetrahedralMesh(
        Eigen::DenseBase<TDerivedT> const& T,
        Eigen::DenseBase<TDerivedXCC> const& XCC,
        Eigen::Index nComponents = -1);
    /**
     * @brief Construct a new Multi Mesh object from a triangle mesh `F` with node connected
     * component labels `XCC`.
     *
     * @tparam TDerivedF Triangle matrix type
     * @tparam TDerivedXCC Connected component label vector type
     * @param F `3 x |# triangles|` array of triangle element indices
     * @param XCC `|# nodes| x 1` array of node connected component labels
     * @param nComponents Number of connected components (optional)
     */
    template <class TDerivedF, class TDerivedXCC>
    void ConstructFromTriangleMesh(
        Eigen::DenseBase<TDerivedF> const& F,
        Eigen::DenseBase<TDerivedXCC> const& XCC,
        Eigen::Index nComponents = -1);
    /**
     * @brief Serialize this MultiMesh to an archive.
     * @param archive Archive to serialize to.
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize this MultiMesh from an archive.
     * @param archive Archive to deserialize from.
     */
    void Deserialize(io::Archive const& archive);
};

template <
    class TDerivedT,
    class TDerivedXCC,
    class TDerivedV,
    class TDerivedF,
    class TDerivedVP,
    class TDerivedFP,
    class TDerivedGXV,
    common::CIndex TIndex>
void BoundaryTriangulation(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::DenseBase<TDerivedV>& V,
    Eigen::DenseBase<TDerivedF>& F,
    Eigen::DenseBase<TDerivedVP>& VP,
    Eigen::DenseBase<TDerivedFP>& FP,
    Eigen::DenseBase<TDerivedGXV>& GXV)
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
    // Compute point to vertex mapping
    auto const nVertices = static_cast<TIndex>(V.size());
    GXV                  = Eigen::Vector<TIndex, Eigen::Dynamic>::Constant(nNodes, TIndex(-1));
    GXV(V.reshaped())    = Eigen::Vector<TIndex, Eigen::Dynamic>::LinSpaced(
        nVertices,
        TIndex(0),
        nVertices - TIndex(1));
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
    EHE                      = geometry::EdgeHalfEdgeAdjacency(F, GHEF.template bottomRows<2>());
    E                        = geometry::Edges(F, EHE.template bottomRows<2>());
    EP.setZero();
    // Count connected component occurrences in EP[1:]
    EP(XCC(E.row(0)).array() + 1).array() += TIndex(1);
    // Compute prefix sums
    std::inclusive_scan(EP.begin() + 1, EP.end(), EP.begin() + 1);
}

template <common::CIndex TIndex>
template <class TDerivedT, class TDerivedXCC>
inline MultiMesh<TIndex>::MultiMesh(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::Index nComponents)
{
    ConstructFromTetrahedralMesh(T, XCC, nComponents);
}

template <common::CIndex TIndex>
template <class TDerivedT, class TDerivedXCC>
inline void MultiMesh<TIndex>::ConstructFromTetrahedralMesh(
    Eigen::DenseBase<TDerivedT> const& T,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::Index nComponents)
{
    static_assert(
        TDerivedT::RowsAtCompileTime == 4,
        "Element type must be tetrahedral (4 nodes per element).");
    if (nComponents < 0)
        nComponents = XCC.maxCoeff() + 1;
    VP.resize(nComponents + 1);
    FP.resize(nComponents + 1);
    EP.resize(nComponents + 1);
    BoundaryTriangulation(T, XCC, V, F, VP, FP, GXV);
    BoundaryTriangulationEdges(F, XCC, E, EP, GVHEp, GVHEadj, GHEF, EHE);
}

template <common::CIndex TIndex>
template <class TDerivedF, class TDerivedXCC>
inline void MultiMesh<TIndex>::ConstructFromTriangleMesh(
    Eigen::DenseBase<TDerivedF> const& Fin,
    Eigen::DenseBase<TDerivedXCC> const& XCC,
    Eigen::Index nComponents)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "Face type must be triangular (3 nodes per face).");
    if (nComponents < 0)
        nComponents = XCC.maxCoeff() + 1;
    VP.resize(nComponents + 1);
    FP.resize(nComponents + 1);
    EP.resize(nComponents + 1);
    F = Fin;
    V = Eigen::Vector<TIndex, Eigen::Dynamic>::LinSpaced(
        static_cast<TIndex>(XCC.size()),
        TIndex(0),
        static_cast<TIndex>(XCC.size()) - TIndex(1));
    VP.setZero();
    FP.setZero();
    // Count connected component occurrences in VP[1:] and FP[1:]
    VP(XCC(V.reshaped()).array() + 1).array() += TIndex(1);
    FP(XCC(F.row(0)).array() + 1).array() += TIndex(1);
    // Compute prefix sums
    std::inclusive_scan(VP.begin() + 1, VP.end(), VP.begin() + 1);
    std::inclusive_scan(FP.begin() + 1, FP.end(), FP.begin() + 1);
    BoundaryTriangulationEdges(F, XCC, E, EP, GVHEp, GVHEadj, GHEF, EHE);
    // Compute point to vertex mapping
    GXV = V;
}

template <common::CIndex TIndex>
inline void MultiMesh<TIndex>::Serialize(io::Archive& archive) const
{
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.MultiMesh");
    grp.WriteData("V", V);
    grp.WriteData("F", F);
    grp.WriteData("E", E);
    grp.WriteData("VP", VP);
    grp.WriteData("FP", FP);
    grp.WriteData("EP", EP);
    grp.WriteData("GVHEp", GVHEp);
    grp.WriteData("GVHEadj", GVHEadj);
    grp.WriteData("GHEF", GHEF);
    grp.WriteData("EHE", EHE);
    grp.WriteData("GXV", GXV);
}

template <common::CIndex TIndex>
inline void MultiMesh<TIndex>::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MultiMesh"];
    V        = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("V");
    F        = grp.ReadData<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>("F");
    E        = grp.ReadData<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>("E");
    VP       = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("VP");
    FP       = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("FP");
    EP       = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("EP");
    GVHEp    = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("GVHEp");
    GVHEadj  = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("GVHEadj");
    GHEF     = grp.ReadData<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>("GHEF");
    EHE      = grp.ReadData<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>("EHE");
    GXV      = grp.ReadData<Eigen::Vector<TIndex, Eigen::Dynamic>>("GXV");
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIMESH_H
