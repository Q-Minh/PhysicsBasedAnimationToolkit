/**
 * @file MultiTriangleMeshBvh.cpp
 */

#include "MultiTriangleMeshBvh.h"

#include <algorithm>
#include <stdexcept>

namespace pbat::geometry {

MultiTriangleMeshBvh::MultiTriangleMeshBvh(
    Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    ESceneFeatures eSceneFeatures,
    EBuildQuality eSceneBvhQuality,
    EBuildQuality eMeshBvhQuality)
    : mVertexScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mEdgeScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mFaceScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))}
{
    Construct(device, X, V, F, E, VP, FP, EP, eSceneFeatures, eSceneBvhQuality, eMeshBvhQuality);
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
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
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
    detail::UserData<ScalarType, IndexType> userData{X, V, F, E, VP, FP, EP};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = 0; c < nComponents; ++c)
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
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mVertexScene),
            vertexGeometry,
            static_cast<unsigned int>(c));
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
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mFaceScene),
            triangleGeometry,
            static_cast<unsigned int>(c));
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
        rtcAttachGeometryByID(
            static_cast<RTCScene>(mEdgeScene),
            edgeGeometry,
            static_cast<unsigned int>(c));
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
    rtcCommitScene(static_cast<RTCScene>(mVertexScene));
    rtcCommitScene(static_cast<RTCScene>(mFaceScene));
    rtcCommitScene(static_cast<RTCScene>(mEdgeScene));
}

void MultiTriangleMeshBvh::UpdateGeometry(
    Device device,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP)
{
    detail::UserData userData{X, V, F, E, VP, FP, EP};
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

std::pair<
    Eigen::Vector<MultiTriangleMeshBvh::ScalarType, 3>,
    Eigen::Vector<MultiTriangleMeshBvh::ScalarType, 3>>
MultiTriangleMeshBvh::Bounds() const
{
    RTCBounds bounds;
    rtcGetSceneBounds(static_cast<RTCScene>(mVertexScene), &bounds);
    return {
        Eigen::Vector<ScalarType, 3>(bounds.lower_x, bounds.lower_y, bounds.lower_z),
        Eigen::Vector<ScalarType, 3>(bounds.upper_x, bounds.upper_y, bounds.upper_z)};
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
