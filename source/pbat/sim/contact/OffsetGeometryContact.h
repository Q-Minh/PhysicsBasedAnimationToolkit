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
#include "pbat/io/Archive.h"

#include <embree4/rtcore.h>
#include <type_traits>
#include <utility>

namespace pbat::sim::contact {

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
     * @brief Construct and build the BVH scene from shared buffers.
     *
     * @param device Embree device wrapper
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param nMaxVertexFacetContacts Maximum number of vertex-facet contacts per vertex
     * @param nMaxFacetVertexContacts Maximum number of facet-vertex contacts per triangle
     * @param nMaxEdgeFacetContacts Maximum number of edge-facet contacts per edge
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API OffsetGeometryContact(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        int nMaxVertexFacetContacts    = 16,
        int nMaxFacetVertexContacts    = 16,
        int nMaxEdgeFacetContacts      = 16,
        ESceneFeatures eSceneFeatures  = ESceneFeatures::Dynamic,
        EBuildQuality eSceneBvhQuality = EBuildQuality::Low,
        EBuildQuality eMeshBvhQuality  = EBuildQuality::Low);
    /**
     * @brief Initialize OGC, i.e. build its spatial acceleration data structures.
     *
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param nMaxVertexFacetContacts Maximum number of vertex-facet contacts per vertex
     * @param nMaxFacetVertexContacts Maximum number of facet-vertex contacts per triangle
     * @param nMaxEdgeFacetContacts Maximum number of edge-facet contacts per edge
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API void Initialize(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        int nMaxVertexFacetContacts    = 16,
        int nMaxFacetVertexContacts    = 16,
        int nMaxEdgeFacetContacts      = 16,
        ESceneFeatures eSceneFeatures  = ESceneFeatures::None,
        EBuildQuality eSceneBvhQuality = EBuildQuality::Low,
        EBuildQuality eMeshBvhQuality  = EBuildQuality::Low);
    /**
     * @brief Prepare for contact iteration.
     */
    PBAT_API void PrepareIteration();
    /**
     * @brief Compute vertex-facet and face-facet contact sets.
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param r Contact radius
     * @param rq Query radius
     */
    PBAT_API void VertexFacetContactDetection(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        ScalarType r,
        ScalarType rq);
    /**
     * @brief Compute edge-facet contact sets.
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param r Contact radius
     * @param rq Query radius
     */
    PBAT_API void EdgeEdgeContactDetection(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        ScalarType r,
        ScalarType rq);
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
              ///< sets, where `EOGC.col(2*e + 0)`, `EOGC.col(2*e + 1)` are respectively the vertex
              ///< and edge indices of the contact facets for edge `e`, except for the first
              ///< coefficient, which indicates the number of contact facets in that column.

  private:
    RTCScene mVertexScene{nullptr}; ///< Opaque RTCScene
    RTCScene mEdgeScene{nullptr};   ///< Opaque RTCScene
    RTCScene mFaceScene{nullptr};   ///< Opaque RTCScene

    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexLocks; ///< `|3 # verts)|` array of locks for synchronized access
                      ///< to per-vertex contact facet sets.
    Eigen::Vector<bool, Eigen::Dynamic>
        mEdgeLocks; ///< `|2 # edges|` array of locks for synchronized access
                    ///< to per-edge contact facet sets.
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminv; ///< `|# vertices|` array of vertex displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminf; ///< `|# faces|` array of face displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dmine; ///< `|# edges|` array of edge displacement bounds
    Eigen::Vector<IndexType, Eigen::Dynamic>
        GVHEp; ///< `|# vertices + 1|` vertex to half-edge prefix
    Eigen::Vector<IndexType, Eigen::Dynamic>
        GVHEadj; ///< `|# half edges|` vertex to half-edge adjacency
    Eigen::Matrix<IndexType, 2, Eigen::Dynamic>
        GHEF; ///< `2 x |# half edges|` half-edge to face adjacency
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
