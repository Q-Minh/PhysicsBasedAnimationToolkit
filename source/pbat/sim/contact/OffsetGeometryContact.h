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
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <embree4/rtcore.h>
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
    OffsetGeometryContact() = default;
    /**
     * @brief Construct and build the BVH scene from shared buffers.
     *
     * @param device Embree device wrapper
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param E `2 x |# edges|` edge vertex indices (global indices into X)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API OffsetGeometryContact(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
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
     * @param other The other OffsetGeometryContact to copy from
     */
    PBAT_API OffsetGeometryContact(OffsetGeometryContact const& other);
    /**
     * @brief Copy assignment operator
     * @param other The other OffsetGeometryContact to copy from
     * @return OffsetGeometryContact& Reference to this OffsetGeometryContact
     */
    PBAT_API OffsetGeometryContact& operator=(OffsetGeometryContact const& other);
    /**
     * @brief Move constructor
     * @param other The other OffsetGeometryContact to move from
     */
    PBAT_API OffsetGeometryContact(OffsetGeometryContact&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other The other OffsetGeometryContact to move from
     * @return Reference to this OffsetGeometryContact
     */
    PBAT_API OffsetGeometryContact& operator=(OffsetGeometryContact&& other) noexcept;
    /**
     * @brief Initialize OGC, i.e. build its spatial acceleration data structures.
     *
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param eSceneFeatures Scene features
     * @param eSceneBvhQuality Scene build quality
     * @param eMeshBvhQuality Geometry build quality
     */
    PBAT_API void Initialize(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
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
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     */
    PBAT_API void UpdateGeometry(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP);
    /**
     * @brief Find nearest faces to each vertex.
     *
     * @tparam FOnVertexFacePairFound Callable with signature `void(IndexType vertexIndex, IndexType
     * vertexComponent, IndexType faceIndex, IndexType faceComponent, ScalarType d, size_t
     * nCollisions)`
     * @param device Spatial acceleration device
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param QR `|# vertices| x 1` query radii
     * @param fOnVertexFacePairFound Callable invoked when a vertex-face pair is found within the
     * query radius
     */
    template <class FOnVertexFacePairFound>
    void VertexFacePairsWithinDistance(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& QR,
        FOnVertexFacePairFound&& onVertexFacePairFound) const;
    /**
     * @brief Scene axis-aligned bounding box.
     */
    PBAT_API auto Bounds() const
        -> std::pair<Eigen::Vector<ScalarType, 3>, Eigen::Vector<ScalarType, 3>>;
    /**
     * @brief Destructor
     */
    PBAT_API ~OffsetGeometryContact();

  private:
    /**
     * @brief Destroy the underlying Embree scene.
     */
    void Destroy() noexcept;

    RTCScene mVertexScene{nullptr}; ///< Opaque RTCScene
    RTCScene mEdgeScene{nullptr};   ///< Opaque RTCScene
    RTCScene mFaceScene{nullptr};   ///< Opaque RTCScene

    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        VOGC; ///< `|# max vertex-facet contacts| x 3*|# vertices|` array of per-vertex contact
              ///< facet sets, where `VOGC.col(3*v + 0)`, `VOGC.col(3*v + 1)`, `VOGC.col(3*v + 2)`
              ///< are respectively the vertex, edge and triangle indices of the contact facets for
              ///< vertex `v`.
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        FOGC; ///< `|# max face-facet contacts| x 3*|# triangles|` array of per-triangle contact
              ///< facet sets, where `FOGC.col(3*f + 0)`, `FOGC.col(3*f + 1)`, `FOGC.col(3*f + 2)`
              ///< are respectively the vertex, edge and triangle indices of the contact facets for
              ///< triangle `f`.
    Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic>
        EOGC; ///< `|# max edge-facet contacts| x 3*|# edges|` array of per-edge contact facet
              ///< sets, where `EOGC.col(3*e + 0)`, `EOGC.col(3*e + 1)`, `EOGC.col(3*e + 2)` are
              ///< respectively the vertex, edge and triangle indices of the contact facets for
              ///< edge `e`.
};

namespace detail {

static RTCSceneFlags toRtc(OffsetGeometryContact::ESceneFeatures flags) noexcept
{
    using ESceneFeatures   = OffsetGeometryContact::ESceneFeatures;
    RTCSceneFlags rtcFlags = RTC_SCENE_FLAG_NONE;
    if (flags == ESceneFeatures::Dynamic)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_DYNAMIC);
    }
    if (flags == ESceneFeatures::Compact)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_COMPACT);
    }
    if (flags == ESceneFeatures::Robust)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_ROBUST);
    }
    return rtcFlags;
}

static RTCBuildQuality toRtc(OffsetGeometryContact::EBuildQuality q) noexcept
{
    using Q = OffsetGeometryContact::EBuildQuality;
    switch (q)
    {
        case Q::Low: return RTC_BUILD_QUALITY_LOW;
        case Q::Medium: return RTC_BUILD_QUALITY_MEDIUM;
        case Q::High: [[fallthrough]];
        default: return RTC_BUILD_QUALITY_HIGH;
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct UserData
{
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& V;
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& E;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& VP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& FP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& EP;
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void TriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    Eigen::Index f                  = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<TScalar, 3, 3> xf = userData->X(Eigen::placeholders::all, userData->F.col(f));
    args->bounds_o->lower_x         = std::min({xf(0, 0), xf(0, 1), xf(0, 2)});
    args->bounds_o->lower_y         = std::min({xf(1, 0), xf(1, 1), xf(1, 2)});
    args->bounds_o->lower_z         = std::min({xf(2, 0), xf(2, 1), xf(2, 2)});
    args->bounds_o->upper_x         = std::max({xf(0, 0), xf(0, 1), xf(0, 2)});
    args->bounds_o->upper_y         = std::max({xf(1, 0), xf(1, 1), xf(1, 2)});
    args->bounds_o->upper_z         = std::max({xf(2, 0), xf(2, 1), xf(2, 2)});
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    Eigen::Index e                  = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<TScalar, 3, 2> xe = userData->X(Eigen::placeholders::all, userData->E.col(e));
    args->bounds_o->lower_x         = std::min(xe(0, 0), xe(0, 1));
    args->bounds_o->lower_y         = std::min(xe(1, 0), xe(1, 1));
    args->bounds_o->lower_z         = std::min(xe(2, 0), xe(2, 1));
    args->bounds_o->upper_x         = std::max(xe(0, 0), xe(0, 1));
    args->bounds_o->upper_y         = std::max(xe(1, 0), xe(1, 1));
    args->bounds_o->upper_z         = std::max(xe(2, 0), xe(2, 1));
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void PointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    Eigen::Index v          = static_cast<Eigen::Index>(args->primID);
    auto xv                 = userData->X.col(userData->V(v));
    args->bounds_o->lower_x = xv(0);
    args->bounds_o->lower_y = xv(1);
    args->bounds_o->lower_z = xv(2);
    args->bounds_o->upper_x = xv(0);
    args->bounds_o->upper_y = xv(1);
    args->bounds_o->upper_z = xv(2);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex, class FOnVertexFacePairFound>
struct VertexFacePairsWithinDistanceUserPtr
{
    UserData<TScalar, TIndex>* userData;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& QR;
    FOnVertexFacePairFound fOnVertexFacePairFound;
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex, class FOnVertexFacePairFound>
void VertexFacePairsWithinDistanceRTCCollideFunc(
    void* userPtr,
    RTCCollision* collision,
    [[maybe_unused]] size_t nCollisions)
{
    using CallbackDataType =
        VertexFacePairsWithinDistanceUserPtr<TScalar, TIndex, FOnVertexFacePairFound>;
    CallbackDataType* callbackData      = static_cast<CallbackDataType*>(userPtr);
    UserData<TScalar, TIndex>* userData = callbackData->userData;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& QR = callbackData->QR;
    TIndex v                        = static_cast<TIndex>(collision->primID0);
    TIndex cv                       = static_cast<TIndex>(collision->geomID0);
    TIndex f                        = static_cast<TIndex>(collision->primID1);
    TIndex cf                       = static_cast<TIndex>(collision->geomID1);
    Eigen::Vector<TScalar, 3> xv    = userData->V.col(v);
    Eigen::Matrix<TScalar, 3, 3> xf = userData->V(Eigen::placeholders::all, userData->F.col(f));
    TScalar d2                      = geometry::DistanceQueries::PointTriangle(
        math::linalg::mini::FromEigen(xv),
        math::linalg::mini::FromEigen(xf.col(0)),
        math::linalg::mini::FromEigen(xf.col(1)),
        math::linalg::mini::FromEigen(xf.col(2)));
    if (d2 < QR(v) * QR(v))
    {
        callbackData->fOnVertexFacePairFound(v, cv, f, cf, d2);
    }
}

} // namespace detail

template <class FOnVertexFacePairFound>
inline void OffsetGeometryContact::VertexFacePairsWithinDistance(
    geometry::Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& QR,
    FOnVertexFacePairFound&& fOnVertexFacePairFound) const
{
    detail::UserData<ScalarType, IndexType> userData{V, F, E, VP, FP, EP};
    detail::VertexFacePairsWithinDistanceUserPtr<ScalarType, IndexType, FOnVertexFacePairFound>
        userPtr{&userData, QR, std::forward<FOnVertexFacePairFound>(fOnVertexFacePairFound)};
    rtcCollide(
        static_cast<RTCScene>(mVertexScene),
        static_cast<RTCScene>(mFaceScene),
        &detail::VertexFacePairsWithinDistanceRTCCollideFunc<
            ScalarType,
            IndexType,
            FOnVertexFacePairFound>,
        &userData);
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
