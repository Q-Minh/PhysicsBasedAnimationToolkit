/**
 * @file MultiTriangleMeshBvh.cpp
 */

#include "pbat/geometry/MultiTriangleMeshBvh.h"

#include "MultiTriangleMeshBvh.h"

#include <algorithm>
#include <embree4/rtcore.h>
#include <stdexcept>

namespace pbat::geometry {
    
namespace detail {

static RTCSceneFlags toRtc(pbat::geometry::MultiTriangleMeshBvh::ESceneFeatures flags) noexcept
{
    using ESceneFeatures   = pbat::geometry::MultiTriangleMeshBvh::ESceneFeatures;
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

static RTCBuildQuality toRtc(pbat::geometry::MultiTriangleMeshBvh::EBuildQuality q) noexcept
{
    using Q = pbat::geometry::MultiTriangleMeshBvh::EBuildQuality;
    switch (q)
    {
        case Q::Low: return RTC_BUILD_QUALITY_LOW;
        case Q::Medium: return RTC_BUILD_QUALITY_MEDIUM;
        case Q::High: [[fallthrough]];
        default: return RTC_BUILD_QUALITY_HIGH;
    }
}

struct UserData
{
    Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic> const> const& V;
    Eigen::Ref<Eigen::Matrix<Index, 3, Eigen::Dynamic> const> const& F;
    Eigen::Ref<Eigen::Matrix<Index, 2, Eigen::Dynamic> const> const& E;
    Eigen::Ref<IndexVectorX const> const& VP;
    Eigen::Ref<IndexVectorX const> const& FP;
    Eigen::Ref<IndexVectorX const> const& EP;
};

void TriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData* userData            = static_cast<UserData*>(args->geometryUserPtr);
    Eigen::Index f                = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<float, 3, 3> xf = userData->V(Eigen::placeholders::all, userData->F.col(f));
    xf.minCoeff();
    args->bounds_o->lower_x = std::min({xf(0, 0), xf(0, 1), xf(0, 2)});
    args->bounds_o->lower_y = std::min({xf(1, 0), xf(1, 1), xf(1, 2)});
    args->bounds_o->lower_z = std::min({xf(2, 0), xf(2, 1), xf(2, 2)});
    args->bounds_o->upper_x = std::max({xf(0, 0), xf(0, 1), xf(0, 2)});
    args->bounds_o->upper_y = std::max({xf(1, 0), xf(1, 1), xf(1, 2)});
    args->bounds_o->upper_z = std::max({xf(2, 0), xf(2, 1), xf(2, 2)});
}

void EdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData* userData            = static_cast<UserData*>(args->geometryUserPtr);
    Eigen::Index e                = static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<float, 3, 2> xe = userData->V(Eigen::placeholders::all, userData->E.col(e));
    args->bounds_o->lower_x       = std::min(xe(0, 0), xe(0, 1));
    args->bounds_o->lower_y       = std::min(xe(1, 0), xe(1, 1));
    args->bounds_o->lower_z       = std::min(xe(2, 0), xe(2, 1));
    args->bounds_o->upper_x       = std::max(xe(0, 0), xe(0, 1));
    args->bounds_o->upper_y       = std::max(xe(1, 0), xe(1, 1));
    args->bounds_o->upper_z       = std::max(xe(2, 0), xe(2, 1));
}

void PointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData* userData      = static_cast<UserData*>(args->geometryUserPtr);
    Eigen::Index v          = static_cast<Eigen::Index>(args->primID);
    auto xv                 = userData->V(Eigen::placeholders::all, v);
    args->bounds_o->lower_x = xv(0);
    args->bounds_o->lower_y = xv(1);
    args->bounds_o->lower_z = xv(2);
    args->bounds_o->upper_x = xv(0);
    args->bounds_o->upper_y = xv(1);
    args->bounds_o->upper_z = xv(2);
}

} // namespace detail

MultiTriangleMeshBvh::MultiTriangleMeshBvh(
    Device device,
    Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<Index, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<Index, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<IndexVectorX const> const& VP,
    Eigen::Ref<IndexVectorX const> const& FP,
    Eigen::Ref<IndexVectorX const> const& EP,
    ESceneFeatures eSceneFeatures,
    EBuildQuality eSceneBvhQuality,
    EBuildQuality eMeshBvhQuality)
    : mVertexScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mEdgeScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mFaceScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))}
{
    Construct(device, V, F, E, VP, FP, EP, eSceneFeatures, eSceneBvhQuality, eMeshBvhQuality);
}

MultiTriangleMeshBvh::MultiTriangleMeshBvh(MultiTriangleMeshBvh const& other)
    : mVertexScene{other.mVertexScene}, mEdgeScene{other.mEdgeScene}, mFaceScene{other.mFaceScene}
{
    if (mVertexScene)
    {
        rtcRetainScene(static_cast<RTCScene>(mVertexScene));
    }
    if (mEdgeScene)
    {
        rtcRetainScene(static_cast<RTCScene>(mEdgeScene));
    }
    if (mFaceScene)
    {
        rtcRetainScene(static_cast<RTCScene>(mFaceScene));
    }
}

MultiTriangleMeshBvh& MultiTriangleMeshBvh::operator=(MultiTriangleMeshBvh const& other)
{
    if (this != &other)
    {
        Destroy();
        mVertexScene = other.mVertexScene;
        if (mVertexScene)
        {
            rtcRetainScene(static_cast<RTCScene>(mVertexScene));
        }
        mEdgeScene = other.mEdgeScene;
        if (mEdgeScene)
        {
            rtcRetainScene(static_cast<RTCScene>(mEdgeScene));
        }
        mFaceScene = other.mFaceScene;
        if (mFaceScene)
        {
            rtcRetainScene(static_cast<RTCScene>(mFaceScene));
        }
    }
    return *this;
}

MultiTriangleMeshBvh::MultiTriangleMeshBvh(MultiTriangleMeshBvh&& other) noexcept
    : mVertexScene{std::exchange(other.mVertexScene, nullptr)},
      mEdgeScene{std::exchange(other.mEdgeScene, nullptr)},
      mFaceScene{std::exchange(other.mFaceScene, nullptr)}
{
}

MultiTriangleMeshBvh& MultiTriangleMeshBvh::operator=(MultiTriangleMeshBvh&& other) noexcept
{
    if (this != &other)
    {
        Destroy();
        mVertexScene = std::exchange(other.mVertexScene, nullptr);
        mEdgeScene   = std::exchange(other.mEdgeScene, nullptr);
        mFaceScene   = std::exchange(other.mFaceScene, nullptr);
    }
    return *this;
}

void MultiTriangleMeshBvh::Construct(
    Device device,
    Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<Index, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<Index, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<IndexVectorX const> const& VP,
    Eigen::Ref<IndexVectorX const> const& FP,
    Eigen::Ref<IndexVectorX const> const& EP,
    ESceneFeatures eSceneFeatures,
    EBuildQuality eSceneBvhQuality,
    EBuildQuality eMeshBvhQuality)
{
    rtcSetSceneFlags(static_cast<RTCScene>(mVertexScene), detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(static_cast<RTCScene>(mFaceScene), detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(static_cast<RTCScene>(mEdgeScene), detail::toRtc(eSceneFeatures));
    rtcSetSceneBuildQuality(static_cast<RTCScene>(mVertexScene), detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(static_cast<RTCScene>(mFaceScene), detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(static_cast<RTCScene>(mEdgeScene), detail::toRtc(eSceneBvhQuality));
    detail::UserData userData{V, F, E, VP, FP, EP};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = 0; c < nComponents; ++c)
    {
        Index const vertexOffset = VP[c];
        Index const nVertices    = VP[c + 1] - vertexOffset;
        Index const faceOffset   = FP[c];
        Index const nFaces       = FP[c + 1] - faceOffset;
        Index const edgeOffset   = EP[c];
        Index const nEdges       = EP[c + 1] - edgeOffset;
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mVertexScene),
            vertexGeometry,
            static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(vertexGeometry, static_cast<unsigned int>(nVertices));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(vertexGeometry, &detail::PointRTCBoundsFunction, nullptr);
        rtcSetGeometryBuildQuality(vertexGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(vertexGeometry);
        rtcReleaseGeometry(vertexGeometry);
        // Triangle geometry
        RTCGeometry triangleGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mFaceScene),
            triangleGeometry,
            static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(triangleGeometry, static_cast<unsigned int>(nFaces));
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(triangleGeometry, &detail::TriangleRTCBoundsFunction, nullptr);
        rtcSetGeometryBuildQuality(triangleGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(triangleGeometry);
        rtcReleaseGeometry(triangleGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mEdgeScene),
            edgeGeometry,
            static_cast<unsigned int>(c));
        rtcSetGeometryUserPrimitiveCount(edgeGeometry, static_cast<unsigned int>(nEdges));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(edgeGeometry, &detail::EdgeRTCBoundsFunction, nullptr);
        rtcSetGeometryBuildQuality(edgeGeometry, detail::toRtc(eMeshBvhQuality));
        rtcCommitGeometry(edgeGeometry);
        rtcReleaseGeometry(edgeGeometry);
    }
    rtcCommitScene(static_cast<RTCScene>(mVertexScene));
    rtcCommitScene(static_cast<RTCScene>(mFaceScene));
    rtcCommitScene(static_cast<RTCScene>(mEdgeScene));
}

void MultiTriangleMeshBvh::UpdateGeometry(
    Device device,
    Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<Index, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<Index, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<IndexVectorX const> const& VP,
    Eigen::Ref<IndexVectorX const> const& FP,
    Eigen::Ref<IndexVectorX const> const& EP)
{
    detail::UserData userData{V, F, E, VP, FP, EP};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = 0; c < nComponents; ++c)
    {
        // See https://github.com/RenderKit/embree/blob/v4.4.0/tutorials/collide/collide_device.cpp
        rtcUpdateGeometryBuffer(
            rtcGetGeometry(static_cast<RTCScene>(mVertexScene), static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcUpdateGeometryBuffer(
            rtcGetGeometry(static_cast<RTCScene>(mFaceScene), static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcUpdateGeometryBuffer(
            rtcGetGeometry(static_cast<RTCScene>(mEdgeScene), static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcCommitGeometry(
            rtcGetGeometry(static_cast<RTCScene>(mVertexScene), static_cast<unsigned int>(c)));
        rtcCommitGeometry(
            rtcGetGeometry(static_cast<RTCScene>(mFaceScene), static_cast<unsigned int>(c)));
        rtcCommitGeometry(
            rtcGetGeometry(static_cast<RTCScene>(mEdgeScene), static_cast<unsigned int>(c)));
    }
    rtcCommitScene(static_cast<RTCScene>(mVertexScene));
    rtcCommitScene(static_cast<RTCScene>(mFaceScene));
    rtcCommitScene(static_cast<RTCScene>(mEdgeScene));
}

AxisAlignedBoundingBox<3> MultiTriangleMeshBvh::Bounds() const
{
    RTCBounds bounds;
    rtcGetSceneBounds(static_cast<RTCScene>(mVertexScene), &bounds);
    return AxisAlignedBoundingBox<3>(
        Eigen::Vector3f(bounds.lower_x, bounds.lower_y, bounds.lower_z),
        Eigen::Vector3f(bounds.upper_x, bounds.upper_y, bounds.upper_z));
}

Device MultiTriangleMeshBvh::GetDevice() const
{
    return Device(rtcGetSceneDevice(static_cast<RTCScene>(mVertexScene)));
}

MultiTriangleMeshBvh::~MultiTriangleMeshBvh()
{
    Destroy();
}

void MultiTriangleMeshBvh::Destroy() noexcept
{
    if (mVertexScene)
    {
        rtcReleaseScene(static_cast<RTCScene>(mVertexScene));
        mVertexScene = nullptr;
    }
    if (mEdgeScene)
    {
        rtcReleaseScene(static_cast<RTCScene>(mEdgeScene));
        mEdgeScene = nullptr;
    }
    if (mFaceScene)
    {
        rtcReleaseScene(static_cast<RTCScene>(mFaceScene));
        mFaceScene = nullptr;
    }
}

} // namespace pbat::geometry
