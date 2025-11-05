#include "OffsetGeometryContact.h"

#include "pbat/common/Atomic.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Norm.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <stdexcept>

namespace pbat::sim::contact {

OgcParams& OgcParams::WithRadii(Scalar _r, Scalar _rq)
{
    r  = _r;
    rq = _rq;
    return *this;
}

OgcParams& OgcParams::WithSceneFeatures(ESceneFeatures features)
{
    eSceneFeatures = features;
    return *this;
}

OgcParams& OgcParams::WithBuildQuality(EBuildQuality scene, EBuildQuality mesh)
{
    eSceneBvhQuality = scene;
    eMeshBvhQuality  = mesh;
    return *this;
}

OgcParams& OgcParams::WithMaxContacts(int nvf, int nfv, int nef)
{
    nMaxVertexFacetContacts = nvf;
    nMaxFacetVertexContacts = nfv;
    nMaxEdgeFacetContacts   = nef;
    return *this;
}

OgcParams& OgcParams::Construct(bool bValidate)
{
    if (bValidate)
    {
        if (r < Scalar(0) or rq < Scalar(0) or rq < r)
        {
            throw std::invalid_argument("OgcParams: rq >= r >= 0 required");
        }
        if (nMaxVertexFacetContacts < 0 or nMaxFacetVertexContacts < 0 or nMaxEdgeFacetContacts < 0)
        {
            throw std::invalid_argument("OgcParams: contact capacities must be non-negative");
        }
    }
    return *this;
}

void OgcParams::Serialize(io::Archive& archive) const
{
    archive.WriteMetaData("r", r);
    archive.WriteMetaData("rq", rq);
    archive.WriteMetaData("eSceneFeatures", static_cast<int>(eSceneFeatures));
    archive.WriteMetaData("eSceneBvhQuality", static_cast<int>(eSceneBvhQuality));
    archive.WriteMetaData("eMeshBvhQuality", static_cast<int>(eMeshBvhQuality));
    archive.WriteMetaData("nMaxVertexFacetContacts", nMaxVertexFacetContacts);
    archive.WriteMetaData("nMaxFacetVertexContacts", nMaxFacetVertexContacts);
    archive.WriteMetaData("nMaxEdgeFacetContacts", nMaxEdgeFacetContacts);
}

void OgcParams::Deserialize(io::Archive const& archive)
{
    if (archive.HasMetaData("r"))
        r = archive.ReadMetaData<Scalar>("r");
    if (archive.HasMetaData("rq"))
        rq = archive.ReadMetaData<Scalar>("rq");
    if (archive.HasMetaData("eSceneFeatures"))
        eSceneFeatures = static_cast<ESceneFeatures>(archive.ReadMetaData<int>("eSceneFeatures"));
    if (archive.HasMetaData("eSceneBvhQuality"))
        eSceneBvhQuality =
            static_cast<EBuildQuality>(archive.ReadMetaData<int>("eSceneBvhQuality"));
    if (archive.HasMetaData("eMeshBvhQuality"))
        eMeshBvhQuality = static_cast<EBuildQuality>(archive.ReadMetaData<int>("eMeshBvhQuality"));
    if (archive.HasMetaData("nMaxVertexFacetContacts"))
        nMaxVertexFacetContacts = archive.ReadMetaData<int>("nMaxVertexFacetContacts");
    if (archive.HasMetaData("nMaxFacetVertexContacts"))
        nMaxFacetVertexContacts = archive.ReadMetaData<int>("nMaxFacetVertexContacts");
    if (archive.HasMetaData("nMaxEdgeFacetContacts"))
        nMaxEdgeFacetContacts = archive.ReadMetaData<int>("nMaxEdgeFacetContacts");
}

namespace detail {

static RTCSceneFlags toRtc(OgcParams::ESceneFeatures flags) noexcept
{
    using ESceneFeatures   = OgcParams::ESceneFeatures;
    RTCSceneFlags rtcFlags = RTC_SCENE_FLAG_NONE;
    if (flags == OgcParams::ESceneFeatures::Dynamic)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_DYNAMIC);
    }
    if (flags == OgcParams::ESceneFeatures::Compact)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_COMPACT);
    }
    if (flags == OgcParams::ESceneFeatures::Robust)
    {
        rtcFlags = static_cast<RTCSceneFlags>(rtcFlags | RTC_SCENE_FLAG_ROBUST);
    }
    return rtcFlags;
}

static RTCBuildQuality toRtc(OgcParams::EBuildQuality q) noexcept
{
    using Q = OgcParams::EBuildQuality;
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
    Eigen::Vector<bool, Eigen::Dynamic>* mVertexLocks;
    Eigen::Vector<bool, Eigen::Dynamic>* mEdgeLocks;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminv;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminf;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dmine;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* GVHEp;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* GVHEadj;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* GHEF;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* EHE;
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
void PointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    auto& X                 = *(userData->X);
    auto& V                 = *(userData->V);
    Eigen::Index v          = static_cast<Eigen::Index>(args->primID);
    auto xv                 = X.col(V(v));
    args->bounds_o->lower_x = xv(0) - userData->rq;
    args->bounds_o->lower_y = xv(1) - userData->rq;
    args->bounds_o->lower_z = xv(2) - userData->rq;
    args->bounds_o->upper_x = xv(0) + userData->rq;
    args->bounds_o->upper_y = xv(1) + userData->rq;
    args->bounds_o->upper_z = xv(2) + userData->rq;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserData<TScalar, TIndex>* userData =
        static_cast<UserData<TScalar, TIndex>*>(args->geometryUserPtr);
    auto& X                         = *(userData->X);
    auto& E                         = *(userData->E);
    TIndex e                        = static_cast<TIndex>(args->primID);
    Eigen::Matrix<TScalar, 3, 2> xe = X(Eigen::placeholders::all, E.col(e));
    Eigen::Vector<TScalar, 3> xmid  = TScalar(0.5) * (xe.col(0) + xe.col(1));
    TScalar halfLength              = TScalar(0.5) * (xe.col(0) - xe.col(1)).norm();
    TScalar queryRadius             = halfLength + userData->rq;
    args->bounds_o->lower_x         = xmid(0) - queryRadius;
    args->bounds_o->lower_y         = xmid(1) - queryRadius;
    args->bounds_o->lower_z         = xmid(2) - queryRadius;
    args->bounds_o->upper_x         = xmid(0) + queryRadius;
    args->bounds_o->upper_y         = xmid(1) + queryRadius;
    args->bounds_o->upper_z         = xmid(2) + queryRadius;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void VertexFacetRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    UserData<TScalar, TIndex>* userData = static_cast<UserData<TScalar, TIndex>*>(userPtr);
    auto& X                             = *(userData->X);
    auto& V                             = *(userData->V);
    auto& F                             = *(userData->F);
    auto& FOGC                          = *(userData->FOGC);
    auto& VOGC                          = *(userData->VOGC);
    auto& mVertexLocks                  = *(userData->mVertexLocks);
    auto& dminv                         = *(userData->dminv);
    auto& GVHEp                         = *(userData->GVHEp);
    auto& GVHEadj                       = *(userData->GVHEadj);
    auto& GHEF                          = *(userData->GHEF);
    auto& dminf                         = *(userData->dminf);
    TScalar const r                     = userData->r;
    // For each potential contact pair (v,f)
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get vertex-facet pair (iv, f) where ix is the point index of vertex iv
        RTCCollision const& collision        = collisions[ci];
        TIndex const iv                      = static_cast<TIndex>(collision.primID0);
        TIndex const ix                      = V(iv);
        TIndex const f                       = static_cast<TIndex>(collision.primID1);
        Eigen::Vector<TIndex, 3> const finds = F.col(f);
        // Avoid contact with adjacent triangle
        if ((finds.array() == ix).any())
            continue;
        // Compute distance from vertex i to triangle f via closest point projection
        Eigen::Vector<TScalar, 3> const xi    = X.col(ix);
        Eigen::Matrix<TScalar, 3, 3> const xf = X(Eigen::placeholders::all, finds);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        SVector<TScalar, 3> uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
            FromEigen(xi),
            FromEigen(xf.col(0)),
            FromEigen(xf.col(1)),
            FromEigen(xf.col(2)));
        Eigen::Vector<TScalar, 3> dx2f =
            (xi - (uvw(0) * xf.col(0) + uvw(1) * xf.col(1) + uvw(2) * xf.col(2)));
        TScalar d2 = dx2f.squaredNorm();
        // Update triangle and vertex displacement bounds
        common::AtomicMin(dminv(iv), d2);
        common::AtomicMin(dminf(f), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine face (vertex, edge, triangle) closest to vertex iv
        auto const [alocal, eFace] = ClosestFaceFacetToVertex(uvw(0), uvw(1), uvw(2));
        // Get contact face set column index for vertex iv and face type eFace
        TIndex sj = 3 * iv + eFace;
        // Vectorize contact face index a based on its type (triangle | edge | vertex)
        TIndex a = (eFace == 0) * f + (eFace == 1) * (3 * f + alocal /* he */) +
                   (eFace == 2) * finds(alocal) /* v */;
        // Synchronize reads/writes to vertex iv's contact sets
        common::AtomicExecute(mVertexLocks(iv), [&]() {
            int const nContactFacets = FOGC(0, sj);
            // Avoid duplicated contact with `a` detected from a neighbour facet. Brute force search
            // the list of contact faces for `a`. Is there a better way?
            bool const bExcessContact = (nContactFacets == FOGC.rows() - 1);
            int k;
            if (not bExcessContact)
                for (k = 0; k < nContactFacets; ++k)
                    if (FOGC(k, sj) == a)
                        break;
            bool const bDuplicateContact = (k < nContactFacets);
            if (bDuplicateContact or bExcessContact)
                return;
            // Update contact face sets
            auto const fUpdateContactFaceSets = [&]() {
                FOGC(FOGC(0, sj)++, sj) = a;
                TIndex& counter         = VOGC(0, f);
                TIndex fk               = common::AtomicAdd<TIndex>(counter, 1);
                VOGC(fk, f)             = iv;
            };
            switch (eFace)
            {
                case 2 /* vertex */: {
                    if (IsVertexFeasible<TScalar, TIndex>(X, F, GVHEp, GVHEadj, xi, a))
                        fUpdateContactFaceSets();
                    break;
                }
                case 1 /* edge */: {
                    if (IsEdgeFeasible<TScalar, TIndex>(X, F, GHEF, xi, f, a))
                        fUpdateContactFaceSets();
                    break;
                }
                default /* triangle */: {
                    fUpdateContactFaceSets();
                    break;
                }
            }
        });
    }
}

} // namespace detail

OffsetGeometryContact::OffsetGeometryContact(geometry::Device device)
    : FOGC(),
      VOGC(),
      EOGC(),
      dminv(),
      dminf(),
      dmine(),
      mVertexScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mEdgeScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mFaceScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mVertexLocks(),
      mEdgeLocks()
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
    OgcParams const& params)
    : OffsetGeometryContact()
{
    Initialize(device, X, V, F, E, VP, FP, EP, params);
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
    OgcParams const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.Initialize");
    // 1. Allocate contact sets and bounds
    auto const nVertices  = V.size();
    auto const nFacets    = F.cols();
    auto const nEdges     = E.cols();
    auto const nHalfEdges = 3 * nFacets;
    FOGC.setConstant(params.nMaxVertexFacetContacts + 1, 3 * nVertices, IndexType(-1));
    VOGC.setConstant(params.nMaxFacetVertexContacts + 1, nFacets, IndexType(-1));
    EOGC.setConstant(params.nMaxEdgeFacetContacts + 1, 2 * nHalfEdges, IndexType(-1));
    dminv.resize(nVertices);
    dminf.resize(nFacets);
    dmine.resize(nEdges);
    // 2. Compute BVHs
    rtcSetSceneFlags(mVertexScene, detail::toRtc(params.eSceneFeatures));
    rtcSetSceneFlags(mFaceScene, detail::toRtc(params.eSceneFeatures));
    rtcSetSceneFlags(mEdgeScene, detail::toRtc(params.eSceneFeatures));
    rtcSetSceneBuildQuality(mVertexScene, detail::toRtc(params.eSceneBvhQuality));
    rtcSetSceneBuildQuality(mFaceScene, detail::toRtc(params.eSceneBvhQuality));
    rtcSetSceneBuildQuality(mEdgeScene, detail::toRtc(params.eSceneBvhQuality));
    // Iterate from last to first so that Embree doesn't need to continuously resize its ID storage.
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X) /*X*/,
        std::addressof(V) /*V*/,
        std::addressof(F) /*F*/,
        std::addressof(E) /*E*/,
        std::addressof(VP) /*VP*/,
        std::addressof(FP) /*FP*/,
        std::addressof(EP) /*EP*/,
        std::addressof(FOGC) /*FOGC*/,
        std::addressof(VOGC) /*VOGC*/,
        std::addressof(EOGC) /*EOGC*/,
        std::addressof(mVertexLocks) /*mVertexLocks*/,
        std::addressof(mEdgeLocks) /*mEdgeLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        nullptr /*GVHEp*/,
        nullptr /*GVHEadj*/,
        nullptr /*GHEF*/,
        nullptr /*EHE*/,
        ScalarType(0) /*r*/,
        ScalarType(0) /*rq*/};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = nComponents - 1; c >= 0; --c)
    {
        IndexType const vertexOffset       = VP[c];
        IndexType const nComponentVertices = VP[c + 1] - vertexOffset;
        IndexType const faceOffset         = FP[c];
        IndexType const nComponentFaces    = FP[c + 1] - faceOffset;
        IndexType const edgeOffset         = EP[c];
        IndexType const nComponentEdges    = EP[c + 1] - edgeOffset;
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(
            vertexGeometry,
            static_cast<unsigned int>(nComponentVertices));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            vertexGeometry,
            &detail::PointRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(vertexGeometry, detail::toRtc(params.eMeshBvhQuality));
        rtcCommitGeometry(vertexGeometry);
        rtcAttachGeometryByID(mVertexScene, vertexGeometry, static_cast<unsigned int>(c));
        rtcReleaseGeometry(vertexGeometry);
        // Triangle geometry
        RTCGeometry triangleGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(
            triangleGeometry,
            static_cast<unsigned int>(nComponentFaces));
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            triangleGeometry,
            &detail::TriangleRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(triangleGeometry, detail::toRtc(params.eMeshBvhQuality));
        rtcCommitGeometry(triangleGeometry);
        rtcAttachGeometryByID(mFaceScene, triangleGeometry, static_cast<unsigned int>(c));
        rtcReleaseGeometry(triangleGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(edgeGeometry, static_cast<unsigned int>(nComponentEdges));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&userData));
        rtcSetGeometryBoundsFunction(
            edgeGeometry,
            &detail::EdgeRTCBoundsFunction<ScalarType, IndexType>,
            nullptr);
        rtcSetGeometryBuildQuality(edgeGeometry, detail::toRtc(params.eMeshBvhQuality));
        rtcCommitGeometry(edgeGeometry);
        rtcAttachGeometryByID(mEdgeScene, edgeGeometry, static_cast<unsigned int>(c));
        rtcReleaseGeometry(edgeGeometry);
    }
    rtcCommitScene(mVertexScene);
    rtcCommitScene(mFaceScene);
    rtcCommitScene(mEdgeScene);
    // 3. Allocate locks
    mVertexLocks.resize(nVertices);
    mEdgeLocks.resize(nEdges);
}

void OffsetGeometryContact::PrepareIteration(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
    OgcParams const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.PrepareIteration");
    // 0. Reset locks, contact sets and displacement bounds
    mVertexLocks.setConstant(false);
    mEdgeLocks.setConstant(false);
    VOGC.row(0).setZero();
    FOGC.row(0).setZero();
    EOGC.row(0).setZero();
    dminv.setConstant(params.rq * params.rq);
    dminf.setConstant(params.rq * params.rq);
    dmine.setConstant(params.rq * params.rq);
    // 1. Recompute BVHs
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X) /*X*/,
        std::addressof(V) /*V*/,
        std::addressof(F) /*F*/,
        std::addressof(E) /*E*/,
        std::addressof(VP) /*VP*/,
        std::addressof(FP) /*FP*/,
        std::addressof(EP) /*EP*/,
        std::addressof(FOGC) /*FOGC*/,
        std::addressof(VOGC) /*VOGC*/,
        std::addressof(EOGC) /*EOGC*/,
        std::addressof(mVertexLocks) /*mVertexLocks*/,
        std::addressof(mEdgeLocks) /*mEdgeLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        nullptr /*GVHEp*/,
        nullptr /*GVHEadj*/,
        nullptr /*GHEF*/,
        nullptr /*EHE*/,
        params.r /*r*/,
        params.rq /*rq*/};
    Eigen::Index nComponents = VP.size() - 1;
    for (Eigen::Index c = 0; c < nComponents; ++c)
    {
        IndexType const vertexOffset       = VP[c];
        IndexType const nComponentVertices = VP[c + 1] - vertexOffset;
        IndexType const faceOffset         = FP[c];
        IndexType const nComponentFaces    = FP[c + 1] - faceOffset;
        IndexType const edgeOffset         = EP[c];
        IndexType const nComponentEdges    = EP[c + 1] - edgeOffset;
        // Vertex geometry
        RTCGeometry vertexGeometry = rtcGetGeometry(mVertexScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&userData));
        rtcCommitGeometry(vertexGeometry);
        rtcAttachGeometryByID(mVertexScene, vertexGeometry, static_cast<unsigned int>(c));
        // Face geometry
        RTCGeometry faceGeometry = rtcGetGeometry(mFaceScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(faceGeometry, static_cast<void*>(&userData));
        rtcCommitGeometry(faceGeometry);
        rtcAttachGeometryByID(mFaceScene, faceGeometry, static_cast<unsigned int>(c));
        // Edge geometry
        RTCGeometry edgeGeometry = rtcGetGeometry(mEdgeScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&userData));
        rtcCommitGeometry(edgeGeometry);
        rtcAttachGeometryByID(mEdgeScene, edgeGeometry, static_cast<unsigned int>(c));
    }
    rtcCommitScene(mVertexScene);
    rtcCommitScene(mFaceScene);
    rtcCommitScene(mEdgeScene);
}

void OffsetGeometryContact::VertexFacetContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
    OgcParams const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.VertexFacetContactDetection");
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X) /*X*/,
        std::addressof(V) /*V*/,
        std::addressof(F) /*F*/,
        nullptr /*E*/,
        std::addressof(VP) /*VP*/,
        std::addressof(FP) /*FP*/,
        nullptr /*EP*/,
        std::addressof(FOGC) /*FOGC*/,
        std::addressof(VOGC) /*VOGC*/,
        std::addressof(EOGC) /*EOGC*/,
        std::addressof(mVertexLocks) /*mVertexLocks*/,
        std::addressof(mEdgeLocks) /*mEdgeLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        std::addressof(GVHEp) /*GVHEp*/,
        std::addressof(GVHEadj) /*GVHEadj*/,
        std::addressof(GHEF) /*GHEF*/,
        nullptr /*EHE*/,
        params.r /*r*/,
        params.rq /*rq*/};
    rtcCollide(
        mVertexScene,
        mFaceScene,
        detail::VertexFacetRTCCollideFunc<ScalarType, IndexType>,
        static_cast<void*>(&userData));
}

void OffsetGeometryContact::EdgeEdgeContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE,
    OgcParams const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.EdgeEdgeContactDetection");
    detail::UserData<ScalarType, IndexType> userData{
        std::addressof(X) /*X*/,
        std::addressof(V) /*V*/,
        std::addressof(F) /*F*/,
        nullptr /*E*/,
        std::addressof(VP) /*VP*/,
        std::addressof(FP) /*FP*/,
        nullptr /*EP*/,
        std::addressof(FOGC) /*FOGC*/,
        std::addressof(VOGC) /*VOGC*/,
        std::addressof(EOGC) /*EOGC*/,
        std::addressof(mVertexLocks) /*mVertexLocks*/,
        std::addressof(mEdgeLocks) /*mEdgeLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        std::addressof(GVHEp) /*GVHEp*/,
        std::addressof(GVHEadj) /*GVHEadj*/,
        nullptr /*GHEF*/,
        nullptr /*EHE*/,
        params.r /*r*/,
        params.rq /*rq*/};
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

#include <doctest/doctest.h>

namespace pbat::sim::contact::detail::test {

/**
 * @brief Computes the triangle face (vertex, edge or triangle) nearest to the point xv.
 * @param uvw Barycentric coordinates of the closest point on triangle to xv.
 * @return The pair (a, eFace), where a is either a local vertex index or edge index, and eFace
 * indicates the type of face, i.e. (0 | 1 | 2) -> (triangle | edge | vertex)
 */
template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w)
{
    int const nZeros     = (u == TScalar(0)) + (v == TScalar(0)) + (w == TScalar(0));
    bool const bIsVertex = (nZeros == 2);
    bool const bIsEdge   = (nZeros == 1);
    int eFace;
    if (bIsVertex)
        eFace = 2;
    else if (bIsEdge)
        eFace = 1;
    else // is triangle
        eFace = 0;
    int a;
    if (bIsVertex)
    {
        if (v == float(1))
            a = 1;
        else if (w == float(1))
            a = 2;
        else // uvw(0) == float(1) must be true
            a = 0;
    }
    else if (bIsEdge)
    {
        if (u == float(0))
            a = 1;
        else if (v == float(0))
            a = 2;
        else // uvw(2) == float(0) must be true
            a = 0;
    }
    else // is triangle
    {
        a = 0;
    }
    return {a, eFace};
}

} // namespace pbat::sim::contact::detail::test

TEST_CASE("[sim][contact][detail] ClosestFaceFacetToVertex")
{
    using pbat::math::linalg::mini::SVector;
    using namespace pbat::sim::contact;
    auto fCheck = [&](double u, double v, double w, int aExpected, int eFaceExpected) {
        auto got = ClosestFaceFacetToVertex(u, v, w);
        auto exp = detail::test::ClosestFaceFacetToVertex(u, v, w);
        CHECK_EQ(got.first, aExpected);
        CHECK_EQ(got.second, eFaceExpected);
        CHECK_EQ(got.first, exp.first);
        CHECK_EQ(got.second, exp.second);
    };
    // Vertices
    fCheck(1.0, 0.0, 0.0, 0, 2); // vertex 0 -> a=0, eFace=2
    fCheck(0.0, 1.0, 0.0, 1, 2); // vertex 1 -> a=1, eFace=2
    fCheck(0.0, 0.0, 1.0, 2, 2); // vertex 2 -> a=2, eFace=2

    // Edge interiors (midpoints)
    fCheck(0.0, 0.5, 0.5, 1, 1); // edge opposite vertex 0 -> a=1, eFace=1
    fCheck(0.5, 0.0, 0.5, 2, 1); // edge opposite vertex 1 -> a=2, eFace=1
    fCheck(0.5, 0.5, 0.0, 0, 1); // edge opposite vertex 2 -> a=0, eFace=1

    // Triangle interior (no zeros)
    fCheck(0.2, 0.3, 0.5, 0, 0);    // face -> a=0, eFace=0
    fCheck(0.1, 0.1, 0.8, 0, 0);    // face -> a=0, eFace=0
    fCheck(0.34, 0.33, 0.33, 0, 0); // face -> a=0, eFace=0
}