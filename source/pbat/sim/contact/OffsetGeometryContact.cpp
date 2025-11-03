#include "OffsetGeometryContact.h"

#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace pbat::sim::contact {

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
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const* X;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* V;
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const* F;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* E;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* VP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* FP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* EP;
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>* FOGC;
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>* VOGC;
    Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>* EOGC;
    Eigen::Vector<bool, Eigen::Dynamic>* mLocks;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminv;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminf;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dmine;
    TScalar r;
    TScalar rq;
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void TriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    auto& X                         = *(userData->X);
    auto& F                         = *(userData->F);
    Eigen::Index f                  = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<TScalar, 3, 3> xf = X(Eigen::placeholders::all, F.col(f));
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
    auto& X                         = *(userData->X);
    auto& E                         = *(userData->E);
    Eigen::Index e                  = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<TScalar, 3, 2> xe = X(Eigen::placeholders::all, E.col(e));
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
    auto& X                 = *(userData->X);
    auto& V                 = *(userData->V);
    Eigen::Index v          = static_cast<Eigen::Index>(args->primID);
    auto xv                 = X.col(V(v));
    args->bounds_o->lower_x = xv(0);
    args->bounds_o->lower_y = xv(1);
    args->bounds_o->lower_z = xv(2);
    args->bounds_o->upper_x = xv(0);
    args->bounds_o->upper_y = xv(1);
    args->bounds_o->upper_z = xv(2);
}

} // namespace detail

OffsetGeometryContact::OffsetGeometryContact(geometry::Device device)
    : FOGC(),
      VOGC(),
      EOGC(),
      mVertexScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mEdgeScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mFaceScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mLocks(),
      dminv(),
      dminf(),
      dmine()
{
}

OffsetGeometryContact::OffsetGeometryContact(
    geometry::Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    int nMaxVertexFacetContacts,
    int nMaxFaceFacetContacts,
    int nMaxEdgeFacetContacts,
    ESceneFeatures eSceneFeatures,
    EBuildQuality eSceneBvhQuality,
    EBuildQuality eMeshBvhQuality)
    : OffsetGeometryContact()
{
    Initialize(
        device,
        X,
        V,
        F,
        E,
        VP,
        FP,
        EP,
        nMaxVertexFacetContacts,
        nMaxFaceFacetContacts,
        nMaxEdgeFacetContacts,
        eSceneFeatures,
        eSceneBvhQuality,
        eMeshBvhQuality);
}

void OffsetGeometryContact::Initialize(
    geometry::Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    int nMaxVertexFacetContacts,
    int nMaxFaceFacetContacts,
    int nMaxEdgeFacetContacts,
    ESceneFeatures eSceneFeatures,
    EBuildQuality eSceneBvhQuality,
    EBuildQuality eMeshBvhQuality)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.Initialize");
    // 1. Allocate contact sets
    FOGC.setConstant(nMaxVertexFacetContacts, 3 * V.size(), IndexType(-1));
    VOGC.setConstant(nMaxFaceFacetContacts, F.cols(), IndexType(-1));
    EOGC.setConstant(nMaxEdgeFacetContacts, 2 * E.cols(), IndexType(-1));
    // 2. Compute BVHs
    rtcSetSceneFlags(mVertexScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(mFaceScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(mEdgeScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneBuildQuality(mVertexScene, detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(mFaceScene, detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(mEdgeScene, detail::toRtc(eSceneBvhQuality));
    // Iterate from last to first so that Embree doesn't need to continuously resize its ID storage.
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X),
        std::addressof(V),
        std::addressof(F),
        std::addressof(E),
        std::addressof(VP),
        std::addressof(FP),
        std::addressof(EP),
        std::addressof(VOGC),
        std::addressof(FOGC),
        std::addressof(EOGC),
        std::addressof(mLocks),
        std::addressof(dminv),
        std::addressof(dminf),
        std::addressof(dmine),
        ScalarType(0),
        ScalarType(0)};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = nComponents - 1; c >= 0; --c)
    {
        IndexType const vertexOffset = VP[c];
        IndexType const nVertices    = VP[c + 1] - vertexOffset;
        IndexType const faceOffset   = FP[c];
        IndexType const nFaces       = FP[c + 1] - faceOffset;
        IndexType const edgeOffset   = EP[c];
        IndexType const nEdges       = EP[c + 1] - edgeOffset;
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(mVertexScene, vertexGeometry, static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(vertexGeometry, static_cast<unsigned int>(nVertices));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            vertexGeometry,
            &detail::PointRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(vertexGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(vertexGeometry);
        rtcReleaseGeometry(vertexGeometry);
        // Triangle geometry
        RTCGeometry triangleGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(mFaceScene, triangleGeometry, static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(triangleGeometry, static_cast<unsigned int>(nFaces));
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            triangleGeometry,
            &detail::TriangleRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(triangleGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(triangleGeometry);
        rtcReleaseGeometry(triangleGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(mEdgeScene, edgeGeometry, static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(edgeGeometry, static_cast<unsigned int>(nEdges));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            edgeGeometry,
            &detail::EdgeRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(edgeGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(edgeGeometry);
        rtcReleaseGeometry(edgeGeometry);
    }
    rtcCommitScene(mVertexScene);
    rtcCommitScene(mFaceScene);
    rtcCommitScene(mEdgeScene);
    // 3. Initialize locks and bounds
    mLocks.resize(std::max({3 * V.size(), F.cols(), 2 * E.cols()}));
    dminv.resize(3 * V.size());
    dminf.resize(F.cols());
    dmine.resize(2 * E.cols());
}

void OffsetGeometryContact::VertexFacetContactDetection(
    geometry::Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    ScalarType r,
    ScalarType rq)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.VertexFacetContactDetection");
    // 0. Reset locks and dmin
    mLocks.setConstant(false);
    dminv.setConstant(rq);
    dminf.setConstant(rq);
    // 1. Recompute BVHs
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X),
        std::addressof(V),
        std::addressof(F),
        nullptr,
        std::addressof(VP),
        std::addressof(FP),
        nullptr,
        std::addressof(VOGC),
        std::addressof(FOGC),
        std::addressof(EOGC),
        std::addressof(mLocks),
        std::addressof(dminv),
        std::addressof(dminf),
        std::addressof(dmine),
        r,
        rq};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = nComponents - 1; c >= 0; --c)
    {
        IndexType const vertexOffset = VP[c];
        IndexType const nVertices    = VP[c + 1] - vertexOffset;
        IndexType const faceOffset   = FP[c];
        IndexType const nFaces       = FP[c + 1] - faceOffset;
        // Vertex geometry
    }
    // 2. Detect contacts
}

void OffsetGeometryContact::EdgeEdgeContactDetection(
    geometry::Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    ScalarType r,
    ScalarType rq)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.EdgeEdgeContactDetection");
}

std::pair<
    Eigen::Vector<OffsetGeometryContact::ScalarType, 3>,
    Eigen::Vector<OffsetGeometryContact::ScalarType, 3>>
OffsetGeometryContact::Bounds() const
{
    RTCBounds bounds;
    rtcGetSceneBounds(mVertexScene, &bounds);
    return {
        Eigen::Vector<ScalarType, 3>(bounds.lower_x, bounds.lower_y, bounds.lower_z),
        Eigen::Vector<ScalarType, 3>(bounds.upper_x, bounds.upper_y, bounds.upper_z)};
}

OffsetGeometryContact::~OffsetGeometryContact()
{
    Destroy();
}

void OffsetGeometryContact::Destroy() noexcept
{
    if (mVertexScene)
    {
        rtcReleaseScene(mVertexScene);
        mVertexScene = nullptr;
    }
    if (mEdgeScene)
    {
        rtcReleaseScene(mEdgeScene);
        mEdgeScene = nullptr;
    }
    if (mFaceScene)
    {
        rtcReleaseScene(mFaceScene);
        mFaceScene = nullptr;
    }
}

} // namespace pbat::sim::contact