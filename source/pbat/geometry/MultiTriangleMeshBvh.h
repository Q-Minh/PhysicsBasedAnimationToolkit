/**
 * @file MultiTriangleMeshBvh.h
 * @brief Multi-mesh BVH for triangle meshes.
 */

#ifndef PBAT_GEOMETRY_MULTITRIANGLEMESHBVH_H
#define PBAT_GEOMETRY_MULTITRIANGLEMESHBVH_H

#include "Device.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/AxisAlignedBoundingBox.h"

namespace pbat::geometry {

/**
 * @brief RAII wrapper over an Embree RTCScene representing a multi-body triangle-mesh scene.
 *
 * This class does not own mesh topology or vertex positions. It only owns the acceleration
 * structures created within Embree. All API functions accept topology (triangles, edges) and
 * connected-component labels as parameters. Vertex positions must be supplied where needed.
 */
class MultiTriangleMeshBvh
{
  public:
    using ScalarType = Scalar; ///< Type for vertex coordinates
    using IndexType  = Index;  ///< Type for indices into vertex arrays

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
    MultiTriangleMeshBvh() = default;
    /**
     * @brief Construct and build the BVH scene from shared buffers.
     *
     * @param device Embree device wrapper
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API MultiTriangleMeshBvh(
        Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        ESceneFeatures eSceneFeatures  = ESceneFeatures::None,
        EBuildQuality eSceneBvhQuality = EBuildQuality::Low,
        EBuildQuality eMeshBvhQuality  = EBuildQuality::Low);
    /**
     * @brief Copy constructor
     * @param other The other MultiTriangleMeshBvh to copy from
     */
    MultiTriangleMeshBvh(MultiTriangleMeshBvh const& other);
    /**
     * @brief Copy assignment operator
     * @param other The other MultiTriangleMeshBvh to copy from
     * @return MultiTriangleMeshBvh& Reference to this MultiTriangleMeshBvh
     */
    MultiTriangleMeshBvh& operator=(MultiTriangleMeshBvh const& other);
    /**
     * @brief Move constructor
     * @param other The other MultiTriangleMeshBvh to move from
     */
    PBAT_API MultiTriangleMeshBvh(MultiTriangleMeshBvh&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other The other MultiTriangleMeshBvh to move from
     * @return Reference to this MultiTriangleMeshBvh
     */
    PBAT_API MultiTriangleMeshBvh& operator=(MultiTriangleMeshBvh&& other) noexcept;
    /**
     * @brief Construct the BVH scene from shared buffers.
     *
     * @param device Spatial acceleration device
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API void Construct(
        Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        ESceneFeatures eSceneFeatures  = ESceneFeatures::None,
        EBuildQuality eSceneBvhQuality = EBuildQuality::Low,
        EBuildQuality eMeshBvhQuality  = EBuildQuality::Low);
    /**
     * @brief Update the BVH geometry (but not its topology).
     *
     * @param device Spatial acceleration device
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     */
    PBAT_API void UpdateGeometry(
        Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP);
    /**
     * @brief Find nearest faces to each vertex.
     *
     * @param device Spatial acceleration device
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param QR `|# vertices| x 1` query radii
     * @param NF `|# vertices| x 1` output nearest face indices, NF(i) < 0 indicates no face found
     */
    PBAT_API void NearestFacesToVertices(
        Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& QR,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic>> NF) const;
    /**
     * @brief Scene axis-aligned bounding box.
     */
    PBAT_API AxisAlignedBoundingBox<3> Bounds() const;
    /**
     * @brief Get the Device object associated with this BVH
     * @return Device
     */
    PBAT_API Device GetDevice() const;
    /**
     * @brief Destructor
     */
    PBAT_API ~MultiTriangleMeshBvh();

  private:
    /**
     * @brief Destroy the underlying Embree scene.
     */
    void Destroy() noexcept;

    void* mVertexScene{nullptr}; ///< Opaque RTCScene*
    void* mEdgeScene{nullptr};   ///< Opaque RTCScene*
    void* mFaceScene{nullptr};   ///< Opaque RTCScene*
};

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_MULTITRIANGLEMESHBVH_H
