#include "OffsetGeometryContact.h"

#include <algorithm>
#include <stdexcept>

namespace pbat::sim::contact {

OffsetGeometryContact::OffsetGeometryContact(
    geometry::Device device,
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
    Initialize(device, X, V, F, E, VP, FP, EP, eSceneFeatures, eSceneBvhQuality, eMeshBvhQuality);
}

OffsetGeometryContact::OffsetGeometryContact(OffsetGeometryContact const& other)
    : mVertexScene{other.mVertexScene}, mEdgeScene{other.mEdgeScene}, mFaceScene{other.mFaceScene}
{
    if (mVertexScene)
    {
        rtcRetainScene(mVertexScene);
    }
    if (mEdgeScene)
    {
        rtcRetainScene(mEdgeScene);
    }
    if (mFaceScene)
    {
        rtcRetainScene(mFaceScene);
    }
}

OffsetGeometryContact& OffsetGeometryContact::operator=(OffsetGeometryContact const& other)
{
    if (this != &other)
    {
        Destroy();
        mVertexScene = other.mVertexScene;
        if (mVertexScene)
        {
            rtcRetainScene(mVertexScene);
        }
        mEdgeScene = other.mEdgeScene;
        if (mEdgeScene)
        {
            rtcRetainScene(mEdgeScene);
        }
        mFaceScene = other.mFaceScene;
        if (mFaceScene)
        {
            rtcRetainScene(mFaceScene);
        }
    }
    return *this;
}

OffsetGeometryContact::OffsetGeometryContact(OffsetGeometryContact&& other) noexcept
    : mVertexScene{std::exchange(other.mVertexScene, nullptr)},
      mEdgeScene{std::exchange(other.mEdgeScene, nullptr)},
      mFaceScene{std::exchange(other.mFaceScene, nullptr)}
{
}

OffsetGeometryContact& OffsetGeometryContact::operator=(OffsetGeometryContact&& other) noexcept
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

void OffsetGeometryContact::Initialize(
    geometry::Device device,
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
    rtcSetSceneFlags(mVertexScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(mFaceScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneFlags(mEdgeScene, detail::toRtc(eSceneFeatures));
    rtcSetSceneBuildQuality(mVertexScene, detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(mFaceScene, detail::toRtc(eSceneBvhQuality));
    rtcSetSceneBuildQuality(mEdgeScene, detail::toRtc(eSceneBvhQuality));
    detail::UserData<ScalarType, IndexType> userData{X, V, F, E, VP, FP, EP};
    Eigen::Index nComponents = VP.size() - 1;
    // Iterate from last to first so that Embree doesn't need to continuously resize its ID storage.
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
}

void OffsetGeometryContact::UpdateGeometry(
    geometry::Device device,
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
            rtcGetGeometry(mVertexScene, static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcUpdateGeometryBuffer(
            rtcGetGeometry(mFaceScene, static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcUpdateGeometryBuffer(
            rtcGetGeometry(mEdgeScene, static_cast<unsigned int>(c)),
            RTC_BUFFER_TYPE_VERTEX,
            0);
        rtcCommitGeometry(rtcGetGeometry(mVertexScene, static_cast<unsigned int>(c)));
        rtcCommitGeometry(rtcGetGeometry(mFaceScene, static_cast<unsigned int>(c)));
        rtcCommitGeometry(rtcGetGeometry(mEdgeScene, static_cast<unsigned int>(c)));
    }
    rtcCommitScene(mVertexScene);
    rtcCommitScene(mFaceScene);
    rtcCommitScene(mEdgeScene);
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