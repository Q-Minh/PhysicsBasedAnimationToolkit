/**
 * @file State.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief State for Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */
#ifndef PBAT_SIM_CONTACT_OGC_STATE_H
#define PBAT_SIM_CONTACT_OGC_STATE_H

#include "ContactFace.h"
#include "Input.h"
#include "Params.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/Device.h"
#include "pbat/graph/AdjacencySet.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>
#include <embree4/rtcore.h>
#include <tbb/enumerable_thread_specific.h>
#include <vector>

namespace pbat::sim::contact::ogc {

namespace detail {

/**
 * @brief Parameters for RTC bounds function over dynamic geometry
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct DynamicRTCBoundsFunctionParams
{
    Input<TScalar, TIndex> const* input; ///< OGC input for the body
    Params<TScalar> const* params;       ///< OGC parameters
    Eigen::Index b;                      ///< Body index
};

/**
 * @brief Parameters for RTC bounds function over static geometry
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct StaticRTCBoundsFunctionParams
{
    Input<TScalar, TIndex> const* input; ///< OGC input for the body
    Params<TScalar> const* params;       ///< OGC parameters
};

}; // namespace detail

/**
 * @brief Transient OGC algorithm data
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
class State
{
  public:
    using SelfType   = State<TScalar, TIndex>; ///< Self type
    using ScalarType = TScalar;                ///< Scalar type
    using IndexType  = TIndex;                 ///< Index type

    /**
     * @brief Construct a new Ogc State object
     */
    State();
    /**
     * @brief Construct and initialize a new Ogc State object
     * @param device Device to use for acceleration structures
     * @param input OGC's input parameters
     * @param params OGC's parameters
     */
    State(
        geometry::Device device,
        Input<TScalar, TIndex> const& input,
        Params<TScalar> const& params);
    /**
     * @brief Disable copy constructor
     */
    State(State const&) = delete;
    /**
     * @brief Disable copy assignment operator
     */
    State& operator=(State const&) = delete;
    /**
     * @brief Move constructor
     */
    State(State&& other) noexcept;
    /**
     * @brief Move assignment operator
     */
    State& operator=(State&& other) noexcept;
    /**
     * @brief Initialize OGC state from input
     * @param device Device to use for acceleration structures
     * @param input OGC's input parameters
     * @param params OGC's parameters
     */
    void Initialize(
        geometry::Device device,
        Input<TScalar, TIndex> const& input,
        Params<TScalar> const& params);
    /**
     * @brief Prepare for an OGC algorithm execution
     * @param input OGC's input parameters
     * @param params OGC's parameters
     * @pre Initialize() has been called
     */
    void PrepareForExecution(Input<TScalar, TIndex> const& input, Params<TScalar> const& params);
    /**
     * @brief Destroy the State object
     */
    ~State();

  protected:
    /**
     * @brief Destroy acceleration structures
     */
    void DestroyAccelerationStructures();

  public:
    /**
     * @brief Displacement bounds
     */
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        bv; ///< `|# vertices|` array of total vertex displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminv; ///< `|# vertices|` array of vertex local displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminf; ///< `|# facets|` array of face local displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dmine; ///< `|# half-edges|` array of half-edge local displacement bounds

    /**
     * @brief Contact sets (thread-local).
     *
     * API users can use the graph::AdjacencySet::Reduce function to merge these thread-local
     * contact sets into a single set for each contact type after parallel contact generation.
     */
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDDVV; ///< Dynamic-dynamic vertex-vertex contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDDVE; ///< Dynamic-dynamic vertex-(half-)edge contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDDVF; ///< Dynamic-dynamic vertex-triangle contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDDEE; ///< Dynamic-dynamic edge-edge contact pairs.

    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDSVV; ///< Dynamic-static vertex-vertex contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDSVE; ///< Dynamic-static vertex-(half-)edge contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDSVF; ///< Dynamic-static vertex-triangle contact pairs.
    tbb::enumerable_thread_specific<graph::AdjacencySet<void, IndexType>>
        mDSEE; ///< Dynamic-static edge-edge contact pairs.

    /**
     * @brief Acceleration structure for static geometry
     */
    RTCScene mDynamicVertexScene{nullptr}; ///< BVH over dynamic vertices
    RTCScene mDynamicEdgeScene{nullptr};   ///< BVH over dynamic edges
    RTCScene mDynamicFacetScene{nullptr};  ///< BVH over dynamic facets
    RTCScene mStaticVertexScene{nullptr};  ///< BVH over static vertices
    RTCScene mStaticEdgeScene{nullptr};    ///< BVH over static edges
    RTCScene mStaticFacetScene{nullptr};   ///< BVH over static facets

  private:
    std::vector<detail::DynamicRTCBoundsFunctionParams<ScalarType, IndexType>>
        mPerBodyRtcBoundsParams; ///< Per-body transient inputs for embree BVH construction

    /**
     * @brief Prepare per-body RTC bounds function parameters
     * @param input OGC's input parameters
     * @param params OGC's parameters
     */
    void PreparePerBodyRtcBoundsParams(
        Input<TScalar, TIndex> const& input,
        Params<TScalar> const& params);
};

namespace detail {

template <common::CIndex TIndex>
graph::AdjacencySet<ContactFace<TIndex>, TIndex, TIndex> CreateEmptyContactFaceAdjacencySet()
{
    graph::AdjacencySet<ContactFace<TIndex>, TIndex, TIndex> adjSet;
    adjSet.Reserve(4096, 1024);
    return adjSet;
}

} // namespace detail

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline State<TScalar, TIndex>::State()
    : bv(),
      dminv(),
      dminf(),
      dmine(),
      mDDVV(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDDVE(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDDVF(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDDEE(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDSVV(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDSVE(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDSVF(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDSEE(&detail::CreateEmptyContactFaceAdjacencySet<TIndex>),
      mDynamicVertexScene(nullptr),
      mDynamicEdgeScene(nullptr),
      mDynamicFacetScene(nullptr),
      mStaticVertexScene(nullptr),
      mStaticEdgeScene(nullptr),
      mStaticFacetScene(nullptr),
      mPerBodyRtcBoundsParams()
{
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline State<TScalar, TIndex>::State(
    geometry::Device device,
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params)
    : State()
{
    Initialize(device, input, params);
}

namespace detail {

/**
 * @brief RTC bounds function for dynamic points
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicPointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<DynamicRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input = rtcParams->input;
    Params<TScalar> const* params       = rtcParams->params;
    TIndex const b                      = rtcParams->b;
    auto const& X                       = input->X.value();
    auto const& V                       = input->V.value();
    auto const& VP                      = input->VP.value();
    Eigen::Index v                      = input->DynamicVertex(b, args->primID);
    auto xv                             = X.col(V(v));
    TScalar const queryRadius           = params->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = xv(0) - queryRadius;
    args->bounds_o->lower_y             = xv(1) - queryRadius;
    args->bounds_o->lower_z             = xv(2) - queryRadius;
    args->bounds_o->upper_x             = xv(0) + queryRadius;
    args->bounds_o->upper_y             = xv(1) + queryRadius;
    args->bounds_o->upper_z             = xv(2) + queryRadius;
}

/**
 * @brief RTC bounds function for dynamic edges
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicEdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<DynamicRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input   = rtcParams->input;
    Params<TScalar> const* params         = rtcParams->params;
    TIndex const b                        = rtcParams->b;
    auto const& X                         = input->X.value();
    auto const& E                         = input->E.value();
    auto const& EP                        = input->EP.value();
    TIndex e                              = input->DynamicEdge(b, args->primID);
    Eigen::Matrix<TScalar, 3, 2> const xe = X(Eigen::placeholders::all, E.col(e));
    Eigen::Vector<TScalar, 3> const xmid  = TScalar(0.5) * (xe.col(0) + xe.col(1));
    TScalar const halfLength              = TScalar(0.5) * (xe.col(0) - xe.col(1)).norm();
    TScalar const queryRadius = halfLength + params->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x   = xmid(0) - queryRadius;
    args->bounds_o->lower_y   = xmid(1) - queryRadius;
    args->bounds_o->lower_z   = xmid(2) - queryRadius;
    args->bounds_o->upper_x   = xmid(0) + queryRadius;
    args->bounds_o->upper_y   = xmid(1) + queryRadius;
    args->bounds_o->upper_z   = xmid(2) + queryRadius;
}

/**
 * @brief RTC bounds function for dynamic edges
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicTriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<DynamicRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input = rtcParams->input;
    Params<TScalar> const* params       = rtcParams->params;
    TIndex const b                      = rtcParams->b;
    auto const& X                       = input->X.value();
    auto const& F                       = input->F.value();
    auto const& FP                      = input->FP.value();
    Eigen::Index f                      = input->DynamicFacet(b, args->primID);
    Eigen::Matrix<TScalar, 3, 3> xf     = X(Eigen::placeholders::all, F.col(f));
    TScalar constexpr eps               = std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = std::min({xf(0, 0), xf(0, 1), xf(0, 2)}) - eps;
    args->bounds_o->lower_y             = std::min({xf(1, 0), xf(1, 1), xf(1, 2)}) - eps;
    args->bounds_o->lower_z             = std::min({xf(2, 0), xf(2, 1), xf(2, 2)}) - eps;
    args->bounds_o->upper_x             = std::max({xf(0, 0), xf(0, 1), xf(0, 2)}) + eps;
    args->bounds_o->upper_y             = std::max({xf(1, 0), xf(1, 1), xf(1, 2)}) + eps;
    args->bounds_o->upper_z             = std::max({xf(2, 0), xf(2, 1), xf(2, 2)}) + eps;
}

/**
 * @brief RTC bounds function for static points
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void StaticPointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<StaticRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input = rtcParams->input;
    Params<TScalar> const* params       = rtcParams->params;
    auto const& X                       = input->Venv.value();
    TIndex v                            = static_cast<TIndex>(args->primID);
    auto xv                             = X.col(v);
    TScalar const queryRadius           = params->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = xv(0) - queryRadius;
    args->bounds_o->lower_y             = xv(1) - queryRadius;
    args->bounds_o->lower_z             = xv(2) - queryRadius;
    args->bounds_o->upper_x             = xv(0) + queryRadius;
    args->bounds_o->upper_y             = xv(1) + queryRadius;
    args->bounds_o->upper_z             = xv(2) + queryRadius;
}

/**
 * @brief RTC bounds function for static edges
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void StaticEdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<StaticRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input   = rtcParams->input;
    Params<TScalar> const* params         = rtcParams->params;
    auto const& E                         = input->Eenv.value();
    auto const& X                         = input->Venv.value();
    TIndex e                              = static_cast<TIndex>(args->primID);
    Eigen::Matrix<TScalar, 3, 2> const xe = X(Eigen::placeholders::all, E.col(e));
    Eigen::Vector<TScalar, 3> const xmid  = TScalar(0.5) * (xe.col(0) + xe.col(1));
    TScalar const halfLength              = TScalar(0.5) * (xe.col(0) - xe.col(1)).norm();
    TScalar const queryRadius = halfLength + params->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x   = xmid(0) - queryRadius;
    args->bounds_o->lower_y   = xmid(1) - queryRadius;
    args->bounds_o->lower_z   = xmid(2) - queryRadius;
    args->bounds_o->upper_x   = xmid(0) + queryRadius;
    args->bounds_o->upper_y   = xmid(1) + queryRadius;
    args->bounds_o->upper_z   = xmid(2) + queryRadius;
}

/**
 * @brief RTC bounds function for static triangles
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param args RTC bounds function arguments
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void StaticTriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    auto* rtcParams =
        static_cast<StaticRTCBoundsFunctionParams<TScalar, TIndex>*>(args->geometryUserPtr);
    Input<TScalar, TIndex> const* input = rtcParams->input;
    auto const& X                       = input->Venv.value();
    auto const& F                       = input->Fenv.value();
    TIndex f                            = static_cast<TIndex>(args->primID);
    Eigen::Matrix<TScalar, 3, 3> xf     = X(Eigen::placeholders::all, F.col(f));
    TScalar constexpr eps               = std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = std::min({xf(0, 0), xf(0, 1), xf(0, 2)}) - eps;
    args->bounds_o->lower_y             = std::min({xf(1, 0), xf(1, 1), xf(1, 2)}) - eps;
    args->bounds_o->lower_z             = std::min({xf(2, 0), xf(2, 1), xf(2, 2)}) - eps;
    args->bounds_o->upper_x             = std::max({xf(0, 0), xf(0, 1), xf(0, 2)}) + eps;
    args->bounds_o->upper_y             = std::max({xf(1, 0), xf(1, 1), xf(1, 2)}) + eps;
    args->bounds_o->upper_z             = std::max({xf(2, 0), xf(2, 1), xf(2, 2)}) + eps;
}

} // namespace detail

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline State<TScalar, TIndex>::State(State&& other) noexcept : State()
{
    *this = std::move(other);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline State<TScalar, TIndex>& State<TScalar, TIndex>::operator=(State&& other) noexcept
{
    DestroyAccelerationStructures();
    bv                      = std::move(other.bv);
    dminv                   = std::move(other.dminv);
    dminf                   = std::move(other.dminf);
    dmine                   = std::move(other.dmine);
    mDDVV                   = std::move(other.mDDVV);
    mDDVE                   = std::move(other.mDDVE);
    mDDVF                   = std::move(other.mDDFE);
    mDDEE                   = std::move(other.mDDEE);
    mDSVV                   = std::move(other.mDSVV);
    mDSVE                   = std::move(other.mDSVE);
    mDSVF                   = std::move(other.mDSVF);
    mDSEE                   = std::move(other.mDSEE);
    mDynamicVertexScene     = std::exchange(other.mDynamicVertexScene, nullptr);
    mDynamicEdgeScene       = std::exchange(other.mDynamicEdgeScene, nullptr);
    mDynamicFacetScene      = std::exchange(other.mDynamicFacetScene, nullptr);
    mStaticVertexScene      = std::exchange(other.mStaticVertexScene, nullptr);
    mStaticEdgeScene        = std::exchange(other.mStaticEdgeScene, nullptr);
    mStaticFacetScene       = std::exchange(other.mStaticFacetScene, nullptr);
    mPerBodyRtcBoundsParams = std::move(other.mPerBodyRtcBoundsParams);
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void State<TScalar, TIndex>::Initialize(
    geometry::Device device,
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.ogc.State.Initialize");
    // 1. Allocate contact sets and bounds
    auto const nDynamicVertices  = input.V->size();
    auto const nDynamicEdges     = input.E->cols();
    auto const nDynamicFacets    = input.F->cols();
    auto const nDynamicHalfEdges = 3 * nDynamicFacets;
    // TODO:
    // Implement mechanism to reserve memory for thread-local contact sets up-front.
    dminv.resize(nDynamicVertices);
    dminf.resize(nDynamicFacets);
    dmine.resize(nDynamicHalfEdges);
    bv.resize(nDynamicVertices);
    // 2. Compute BVHs
    DestroyAccelerationStructures();
    mDynamicVertexScene = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
    mDynamicEdgeScene   = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
    mDynamicFacetScene  = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
    // 2.a. Dynamic geometry
    rtcSetSceneFlags(mDynamicVertexScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
    rtcSetSceneFlags(mDynamicEdgeScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
    rtcSetSceneFlags(mDynamicFacetScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
    rtcSetSceneBuildQuality(
        mDynamicVertexScene,
        static_cast<RTCBuildQuality>(params.eDynamicSceneBvhQuality));
    rtcSetSceneBuildQuality(
        mDynamicEdgeScene,
        static_cast<RTCBuildQuality>(params.eDynamicSceneBvhQuality));
    rtcSetSceneBuildQuality(
        mDynamicFacetScene,
        static_cast<RTCBuildQuality>(params.eDynamicSceneBvhQuality));
    PreparePerBodyRtcBoundsParams(input, params);
    Eigen::Index nBodies = input.NumBodies();
    for (Eigen::Index b = 0; b < nBodies; ++b)
    {
        Eigen::Index nBodyVerts  = input.NumVertices(b);
        Eigen::Index nBodyEdges  = input.NumEdges(b);
        Eigen::Index nBodyFacets = input.NumFacets(b);
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(vertexGeometry, static_cast<unsigned int>(nBodyVerts));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcSetGeometryBoundsFunction(
            vertexGeometry,
            &detail::DynamicPointRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            vertexGeometry,
            static_cast<RTCBuildQuality>(params.eDynamicMeshBvhQuality));
        rtcCommitGeometry(vertexGeometry);
        rtcAttachGeometryByID(mDynamicVertexScene, vertexGeometry, static_cast<unsigned int>(b));
        rtcReleaseGeometry(vertexGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(edgeGeometry, static_cast<unsigned int>(nBodyEdges));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcSetGeometryBoundsFunction(
            edgeGeometry,
            &detail::DynamicEdgeRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            edgeGeometry,
            static_cast<RTCBuildQuality>(params.eDynamicMeshBvhQuality));
        rtcCommitGeometry(edgeGeometry);
        rtcAttachGeometryByID(mDynamicEdgeScene, edgeGeometry, static_cast<unsigned int>(b));
        rtcReleaseGeometry(edgeGeometry);
        // Facet geometry
        RTCGeometry triangleGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(triangleGeometry, static_cast<unsigned int>(nBodyFacets));
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcSetGeometryBoundsFunction(
            triangleGeometry,
            &detail::DynamicTriangleRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            triangleGeometry,
            static_cast<RTCBuildQuality>(params.eDynamicMeshBvhQuality));
        rtcCommitGeometry(triangleGeometry);
        rtcAttachGeometryByID(mDynamicFacetScene, triangleGeometry, static_cast<unsigned int>(b));
        rtcReleaseGeometry(triangleGeometry);
    }
    rtcCommitScene(mDynamicVertexScene);
    rtcCommitScene(mDynamicEdgeScene);
    rtcCommitScene(mDynamicFacetScene);
    // 2.b. Static geometry
    if (input.HasStaticGeometry())
    {
        mStaticVertexScene = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
        mStaticEdgeScene   = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
        mStaticFacetScene  = rtcNewScene(static_cast<RTCDevice>(device.Raw()));
        rtcSetSceneFlags(mStaticVertexScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
        rtcSetSceneBuildQuality(
            mStaticVertexScene,
            static_cast<RTCBuildQuality>(params.eStaticSceneBvhQuality));
        rtcSetSceneFlags(mStaticEdgeScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
        rtcSetSceneBuildQuality(
            mStaticEdgeScene,
            static_cast<RTCBuildQuality>(params.eStaticSceneBvhQuality));
        rtcSetSceneFlags(mStaticFacetScene, static_cast<RTCSceneFlags>(params.eSceneFeatures));
        rtcSetSceneBuildQuality(
            mStaticFacetScene,
            static_cast<RTCBuildQuality>(params.eStaticSceneBvhQuality));
        detail::StaticRTCBoundsFunctionParams<ScalarType, IndexType> staticRtcBoundsFunctionParams{
            std::addressof(input),
            std::addressof(params)};
        Eigen::Index nStaticVerts  = input.NumStaticVertices();
        Eigen::Index nStaticEdges  = input.NumStaticEdges();
        Eigen::Index nStaticFacets = input.NumStaticFacets();
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(vertexGeometry, static_cast<unsigned int>(nStaticVerts));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&staticRtcBoundsFunctionParams));
        rtcSetGeometryBoundsFunction(
            vertexGeometry,
            &detail::StaticPointRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            vertexGeometry,
            static_cast<RTCBuildQuality>(params.eStaticMeshBvhQuality));
        rtcCommitGeometry(vertexGeometry);
        rtcAttachGeometry(mStaticVertexScene, vertexGeometry);
        rtcReleaseGeometry(vertexGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(edgeGeometry, static_cast<unsigned int>(nStaticEdges));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&staticRtcBoundsFunctionParams));
        rtcSetGeometryBoundsFunction(
            edgeGeometry,
            &detail::StaticEdgeRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            edgeGeometry,
            static_cast<RTCBuildQuality>(params.eStaticMeshBvhQuality));
        rtcCommitGeometry(edgeGeometry);
        rtcAttachGeometry(mStaticEdgeScene, edgeGeometry);
        rtcReleaseGeometry(edgeGeometry);
        // Facet geometry
        RTCGeometry triangleGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(
            triangleGeometry,
            static_cast<unsigned int>(nStaticFacets));
        rtcSetGeometryUserData(
            triangleGeometry,
            static_cast<void*>(&staticRtcBoundsFunctionParams));
        rtcSetGeometryBoundsFunction(
            triangleGeometry,
            &detail::StaticTriangleRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(
            triangleGeometry,
            static_cast<RTCBuildQuality>(params.eStaticMeshBvhQuality));
        rtcCommitGeometry(triangleGeometry);
        rtcAttachGeometry(mStaticFacetScene, triangleGeometry);
        rtcReleaseGeometry(triangleGeometry);
        rtcCommitScene(mStaticVertexScene);
        rtcCommitScene(mStaticEdgeScene);
        rtcCommitScene(mStaticFacetScene);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void State<TScalar, TIndex>::PrepareForExecution(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.ogc.State.PrepareForExecution");
    // 1. Clear contact sets
    for (graph::AdjacencySet<void, TIndex>& vv : mDDVV)
        vv.Clear();
    for (graph::AdjacencySet<void, TIndex>& ve : mDDVE)
        ve.Clear();
    for (graph::AdjacencySet<void, TIndex>& vf : mDDVF)
        vf.Clear();
    for (graph::AdjacencySet<void, TIndex>& ee : mDDEE)
        ee.Clear();
    for (graph::AdjacencySet<void, TIndex>& vv : mDSVV)
        vv.Clear();
    for (graph::AdjacencySet<void, TIndex>& ve : mDSVE)
        ve.Clear();
    for (graph::AdjacencySet<void, TIndex>& vf : mDSVF)
        vf.Clear();
    for (graph::AdjacencySet<void, TIndex>& ee : mDSEE)
        ee.Clear();
    // 2. Reset bounds
    dminv.setConstant(params.rq * params.rq);
    dminf.setConstant(params.rq * params.rq);
    dmine.setConstant(params.rq * params.rq);
    // 3. Recompute dynamic BVHs
    PreparePerBodyRtcBoundsParams(input, params);
    Eigen::Index nBodies = input.NumBodies();
    for (Eigen::Index b = 0; b < nBodies; ++b)
    {
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcGetGeometry(mDynamicVertexScene, static_cast<unsigned int>(b));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcCommitGeometry(vertexGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry = rtcGetGeometry(mDynamicEdgeScene, static_cast<unsigned int>(b));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcCommitGeometry(edgeGeometry);
        // Facet geometry
        RTCGeometry triangleGeometry =
            rtcGetGeometry(mDynamicFacetScene, static_cast<unsigned int>(b));
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&mPerBodyRtcBoundsParams[b]));
        rtcCommitGeometry(triangleGeometry);
    }
    rtcCommitScene(mDynamicVertexScene);
    rtcCommitScene(mDynamicEdgeScene);
    rtcCommitScene(mDynamicFacetScene);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void State<TScalar, TIndex>::DestroyAccelerationStructures()
{
    if (mDynamicVertexScene)
    {
        rtcReleaseScene(mDynamicVertexScene);
        mDynamicVertexScene = nullptr;
    }
    if (mDynamicEdgeScene)
    {
        rtcReleaseScene(mDynamicEdgeScene);
        mDynamicEdgeScene = nullptr;
    }
    if (mDynamicFacetScene)
    {
        rtcReleaseScene(mDynamicFacetScene);
        mDynamicFacetScene = nullptr;
    }
    if (mStaticVertexScene)
    {
        rtcReleaseScene(mStaticVertexScene);
        mStaticVertexScene = nullptr;
    }
    if (mStaticEdgeScene)
    {
        rtcReleaseScene(mStaticEdgeScene);
        mStaticEdgeScene = nullptr;
    }
    if (mStaticFacetScene)
    {
        rtcReleaseScene(mStaticFacetScene);
        mStaticFacetScene = nullptr;
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
State<TScalar, TIndex>::~State()
{
    DestroyAccelerationStructures();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void State<TScalar, TIndex>::PreparePerBodyRtcBoundsParams(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params)
{
    Eigen::Index nBodies = input.NumBodies();
    mPerBodyRtcBoundsParams.resize(nBodies);
    for (Eigen::Index b = 0; b < nBodies; ++b)
    {
        mPerBodyRtcBoundsParams[b].input  = std::addressof(input);
        mPerBodyRtcBoundsParams[b].params = std::addressof(params);
        mPerBodyRtcBoundsParams[b].b      = b;
    }
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_STATE_H
