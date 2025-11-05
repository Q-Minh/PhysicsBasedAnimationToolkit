/**
 * @file OffsetGeometryContact.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for offset geometric contact model.
 * @version 0.1
 * @date 2025-11-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
#define PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/io/Archive.h"

#include <embree4/rtcore.h>
#include <type_traits>
#include <utility>

namespace pbat::sim::contact {

/**
 * @brief Parameters for Offset Geometry Contact (OGC).
 *
 * This bundles algorithm configuration knobs and capacities. Mesh data and device are
 * intentionally not stored here per design; this type is for configurable parameters only.
 */
struct OgcParams
{
    /**
     * @brief BVH build quality options.
     */
    enum class EBuildQuality {
        Low,    ///< Low build quality (fast build time)
        Medium, ///< Medium build quality (balanced)
        High    ///< High build quality (slow build time)
    };
    /**
     * @brief Scene construction features.
     */
    enum class ESceneFeatures {
        None,    ///< No special features
        Dynamic, ///< Dynamic scene
        Compact, ///< Compact representation
        Robust   ///< Robust representation
    };

    Scalar r{Scalar(0)};  ///< Contact radius
    Scalar rq{Scalar(0)}; ///< Query inflation radius

    ESceneFeatures eSceneFeatures{ESceneFeatures::None}; ///< Scene features
    EBuildQuality eSceneBvhQuality{EBuildQuality::Low};  ///< Scene BVH build quality
    EBuildQuality eMeshBvhQuality{EBuildQuality::Low};   ///< Mesh BVH build quality

    int nMaxVertexFacetContacts{16}; ///< Max vertex-facet contacts
    int nMaxFacetVertexContacts{16}; ///< Max facet-vertex contacts
    int nMaxEdgeFacetContacts{16};   ///< Max edge-facet contacts

  public:
    /**
     * @brief Set contact and query radii.
     * @param _r Contact radius
     * @param _rq Query radius
     * @return Reference to this
     */
    PBAT_API OgcParams& WithRadii(Scalar _r, Scalar _rq);
    /**
     * @brief Set scene features.
     * @param features Scene features
     * @return Reference to this
     */
    PBAT_API OgcParams& WithSceneFeatures(ESceneFeatures features);
    /**
     * @brief Set both scene and mesh BVH build quality.
     * @param scene Build quality for scene
     * @param mesh Build quality for mesh
     * @return Reference to this
     */
    PBAT_API OgcParams& WithBuildQuality(EBuildQuality scene, EBuildQuality mesh);
    /**
     * @brief Set maximum number of contacts.
     * @param nvf Max vertex-facet contacts
     * @param nfv Max facet-vertex contacts
     * @param nef Max edge-facet contacts
     * @return Reference to this
     */
    PBAT_API OgcParams& WithMaxContacts(int nvf, int nfv, int nef);
    /**
     * @brief Validate and construct the parameters.
     * @param bValidate Whether to validate parameters
     * @return Reference to this
     */
    PBAT_API OgcParams& Construct(bool bValidate = true);
    /**
     * @brief Serialize the parameters to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the parameters from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);
};

/**
 * @brief API of Offset Geometric Contact (OGC) algorithm \cite chen_offset_2025 for multi-body
 * triangle mesh scene.
 *
 * This class does not own mesh topology or vertex positions. It only owns the acceleration
 * structures created within Embree. All API functions accept topology (triangles, edges) and
 * connected-component labels as parameters. Vertex positions must be supplied where needed.
 */
class OffsetGeometryContact
{
  public:
    using ScalarType = Scalar; ///< Type for vertex coordinates
    using IndexType  = Index;  ///< Type for indices into vertex arrays
    static_assert(std::is_signed_v<IndexType>, "IndexType must be a signed integer type");

    /**
     * @brief Default constructor
     */
    PBAT_API OffsetGeometryContact() = default;
    /**
     * @brief Construct an empty OGC object with given device.
     * @param device Spatial acceleration device
     */
    PBAT_API OffsetGeometryContact(geometry::Device device);
    /**
     * @brief Deleted copy constructor and copy assignment operator.
     */
    OffsetGeometryContact(OffsetGeometryContact const&)            = delete;
    OffsetGeometryContact& operator=(OffsetGeometryContact const&) = delete;
    /**
     * @brief Defaulted move constructor and move assignment operator.
     */
    OffsetGeometryContact(OffsetGeometryContact&&)            = default;
    OffsetGeometryContact& operator=(OffsetGeometryContact&&) = default;
    /**
     * @brief Construct and build the BVH scene from shared buffers.
     *
     * @param device Embree device wrapper
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param params OGC parameters
     */
    PBAT_API OffsetGeometryContact(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        OgcParams const& params);
    /**
     * @brief Initialize OGC, i.e. build its spatial acceleration data structures.
     *
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param params OGC parameters
     */
    PBAT_API void Initialize(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        OgcParams const& params);
    /**
     * @brief Prepare for contact iteration.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param params OGC parameters
     */
    PBAT_API void PrepareIteration(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        OgcParams const& params);
    /**
     * @brief Compute vertex-facet and face-facet contact sets.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param GVHEp `|# points + 1|` point to half-edge prefix
     * @param GVHEadj `|# half edges|` half-edge adjacency
     * @param GHEF `2 x |# half edges|` half-edge to face adjacency
     * @param params OGC parameters
     */
    PBAT_API void VertexFacetContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
        OgcParams const& params);
    /**
     * @brief Compute edge-facet contact sets.
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param GVHEp `|# points + 1|` point to half-edge prefix
     * @param GVHEadj `|# half edges|` half-edge adjacency
     * @param params OGC parameters
     */
    PBAT_API void EdgeEdgeContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
        OgcParams const& params);
    /**
     * @brief Scene axis-aligned bounding box.
     */
    PBAT_API auto Bounds() const
        -> std::pair<Eigen::Vector<ScalarType, 3>, Eigen::Vector<ScalarType, 3>>;
    /**
     * @brief Get the vertex displacement bound of vertex `v`.
     * @param v Vertex index
     * @return Displacement bound guaranteeing penetration-free motion
     */
    PBAT_API ScalarType VertexDisplacementBound(IndexType v) const;
    /**
     * @brief Serialize the OGC to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the OGC from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive& archive);
    /**
     * @brief Destructor
     */
    PBAT_API ~OffsetGeometryContact();

  private:
    /**
     * @brief Destroy the underlying Embree scene.
     */
    void Destroy() noexcept;

  public:
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        FOGC; ///< `|# max vertex-facet contacts + 1| x 3*|# vertices|` array of per-vertex contact
              ///< facet sets, where `FOGC.col(3*v + 0)`, `FOGC.col(3*v + 1)`, `FOGC.col(3*v + 2)`
              ///< are respectively the triangle, edge and vertex indices of the contact facets for
              ///< vertex `v`, except for the first coefficient, which indicates the number of
              ///< contact facets in that column.
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        VOGC; ///< `|# max facet-vertex contacts + 1| x |# triangles|` array of per-triangle contact
              ///< facet sets, where `FOGC.col(f)` are vertex indices of the contact vertices for
              ///< triangle `f`, except for the first coefficient, which indicates the number of
              ///< contact vertices in that column.
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        EOGC; ///< `|# max edge-facet contacts + 1| x 2*|# edges|` array of per-edge contact facet
              ///< sets, where `EOGC.col(2*e + 0)`, `EOGC.col(2*e + 1)` are respectively the edge
              ///< and vertex indices of the contact facets for edge `e`, except for the first
              ///< coefficient, which indicates the number of contact facets in that column.
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminv; ///< `|# vertices|` array of vertex displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminf; ///< `|# faces|` array of face displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dmine; ///< `|# edges|` array of edge displacement bounds

  private:
    RTCScene mVertexScene{nullptr}; ///< Opaque RTCScene
    RTCScene mEdgeScene{nullptr};   ///< Opaque RTCScene
    RTCScene mFaceScene{nullptr};   ///< Opaque RTCScene

    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexLocks; ///< `|# verts|` array of locks for synchronized access
                      ///< to per-vertex contact facet sets.
    Eigen::Vector<bool, Eigen::Dynamic> mEdgeLocks; ///< `|# edges|` array of locks for synchronized
                                                    ///< access to per-edge contact facet sets.
};

/**
 * @brief Computes the triangle face (vertex, edge or triangle) nearest to a point's projection on
 * a triangle.
 * @param u First barycentric coordinate of the closest point on the triangle
 * @param v Second barycentric coordinate of the closest point on the triangle
 * @param w Third barycentric coordinate of the closest point on the triangle
 * @return The pair (a, eFace), where a is either a local vertex index or edge index, and eFace
 * indicates the type of face, i.e. (0 | 1 | 2) -> (triangle | edge | vertex)
 */
template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w);

/**
 * @brief Determines if point x is in the vertex feasible region of vertex i.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GVHEp `|# vertices + 1|` vertex to half-edge prefix
 * @param GVHEadj `|# half edges|` vertex to half-edge adjacency
 * @param x `3 x 1` query point
 * @param i Point (global) index corresponding to vertex
 * @return true if in vertex feasible region; false otherwise
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsVertexFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i);

/**
 * @brief Determines if point x is in the edge feasible region of half-edge he of face fi.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GHEF `2 x |# half edges|` half-edge to (adjacent face, opposite face)
 * @param x `3 x 1` query point
 * @param fi Face index of half-edge he
 * @param he Half-edge index
 * @return true if in edge feasible region; false otherwise
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsEdgeFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he);

template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w)
{
    int const nZeros     = (u == TScalar(0)) + (v == TScalar(0)) + (w == TScalar(0));
    bool const bIsVertex = (nZeros == 2);
    bool const bIsEdge   = (nZeros == 1);
    int eFace            = (bIsVertex * 2) + (bIsEdge * 1);
    int a                = bIsVertex * ((v == TScalar(1)) * 1 + (w == TScalar(1)) * 2) +
            bIsEdge * ((u == TScalar(0)) * 1 + (v == TScalar(0)) * 2);
    return {a, eFace};
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsVertexFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i)
{
    bool bInVertexFeasibleRegion{true};
    TIndex const hebegin               = GVHEp(i);
    TIndex const heend                 = GVHEp(i + 1);
    Eigen::Vector<TScalar, 3> const xv = X.col(i);
    for (TIndex he : GVHEadj(Eigen::seq(hebegin, heend - 1)))
    {
        TIndex const vp = geometry::OutgoingVertex<TIndex>(F, he);
        auto const xvp  = X.col(vp);
        bInVertexFeasibleRegion &= ((x - xv).dot(xv - xvp) >= TScalar(0));
    }
    return bInVertexFeasibleRegion;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsEdgeFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he)
{
    TIndex fj = GHEF(1, he);
    TIndex i  = geometry::IncomingVertex(F, he);
    TIndex j  = geometry::OutgoingVertex(F, he);
    TIndex k  = geometry::OutgoingVertex(F, he, 1 /* step */);
    // Get the third vertex l of triangle fj that is not part of undirected edge (i,j)
    TIndex l = (F(0, fj) != i and F(0, fj) != j) * F(0, fj) +
               (F(1, fj) != i and F(1, fj) != j) * F(1, fj) +
               (F(2, fj) != i and F(2, fj) != j) * F(2, fj);
    Eigen::Vector<TScalar, 3> xi  = X.col(i);
    Eigen::Vector<TScalar, 3> xj  = X.col(j);
    Eigen::Vector<TScalar, 3> xk  = X.col(k);
    Eigen::Vector<TScalar, 3> xl  = X.col(l);
    Eigen::Vector<TScalar, 3> xij = xj - xi;
    TScalar xijn2                 = xij.squaredNorm();
    // Tangent to the plane spanned by triangle fi, perpendicular to edge (i,j)
    Eigen::Vector<TScalar, 3> pin = (xi - xk) + (xk - xi).dot(xij) / xijn2 * xij;
    // Tangent to the plane spanned by triangle fj, perpendicular to edge (i,j)
    Eigen::Vector<TScalar, 3> pjn = (xi - xl) + (xl - xi).dot(xij) / xijn2 * xij;
    bool bInEdgeFeasibleRegion =
        ((x - xi).dot(xj - xi) >= TScalar(0)) and // within half-plane of vertex i
        ((x - xj).dot(xi - xj) >= TScalar(0)) and // within half-plane of vertex j
        ((x - xi).dot(pin) >= TScalar(0)) and     // within half-plane perpendicular to fi
        ((x - xi).dot(pjn) >= TScalar(0));        // within half-plane perpendicular to fj
    return bInEdgeFeasibleRegion;
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
