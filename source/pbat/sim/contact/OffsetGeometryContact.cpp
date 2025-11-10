#include "OffsetGeometryContact.h"

#include "pbat/common/Atomic.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Norm.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <stdexcept>
#include <tbb/parallel_for.h>

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

OgcParams& OgcParams::WithMaxContactEstimates(int nvf, int nfv, int nef)
{
    nMaxVertexFaceContactsEstimate = nvf;
    nMaxFaceVertexContactsEstimate = nfv;
    nMaxEdgeFaceContactsEstimate   = nef;
    return *this;
}

OgcParams& OgcParams::WithDisplacementBoundConfig(Scalar _gammap, Scalar _gammae)
{
    gammap = _gammap;
    gammae = _gammae;
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
        if (nMaxVertexFaceContactsEstimate < 0 or nMaxFaceVertexContactsEstimate < 0 or
            nMaxEdgeFaceContactsEstimate < 0)
        {
            throw std::invalid_argument("OgcParams: contact capacities must be non-negative");
        }
        if (gammap <= Scalar(0) or gammap >= Scalar(0.5))
        {
            throw std::invalid_argument(
                "OgcParams: 0 < gammap < 0.5 required for displacement bound config");
        }
        if (gammae < Scalar(0) or gammae >= Scalar(1))
        {
            throw std::invalid_argument(
                "OgcParams: 0 <= gammae < 1 required for displacement bound config");
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
    archive.WriteMetaData("nMaxVertexFaceContactsEstimate", nMaxVertexFaceContactsEstimate);
    archive.WriteMetaData("nMaxFaceVertexContactsEstimate", nMaxFaceVertexContactsEstimate);
    archive.WriteMetaData("nMaxEdgeFaceContactsEstimate", nMaxEdgeFaceContactsEstimate);
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
    if (archive.HasMetaData("nMaxVertexFaceContactsEstimate"))
        nMaxVertexFaceContactsEstimate =
            archive.ReadMetaData<int>("nMaxVertexFaceContactsEstimate");
    if (archive.HasMetaData("nMaxFaceVertexContactsEstimate"))
        nMaxFaceVertexContactsEstimate =
            archive.ReadMetaData<int>("nMaxFaceVertexContactsEstimate");
    if (archive.HasMetaData("nMaxEdgeFaceContactsEstimate"))
        nMaxEdgeFaceContactsEstimate = archive.ReadMetaData<int>("nMaxEdgeFaceContactsEstimate");
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
void TriangleRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserDataByComponent<TScalar, TIndex>* userDataByComp =
        static_cast<UserDataByComponent<TScalar, TIndex>*>(args->geometryUserPtr);
    UserData<TScalar, TIndex>* userData = userDataByComp->userData;
    TIndex const component              = userDataByComp->component;
    auto& X                             = *(userData->X);
    auto& F                             = *(userData->F);
    auto& FP                            = *(userData->FP);
    Eigen::Index f                      = FP(component) + static_cast<Eigen::Index>(args->primID);
    Eigen::Matrix<TScalar, 3, 3> xf     = X(Eigen::placeholders::all, F.col(f));
    TScalar constexpr eps               = std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = std::min({xf(0, 0), xf(0, 1), xf(0, 2)}) - eps;
    args->bounds_o->lower_y             = std::min({xf(1, 0), xf(1, 1), xf(1, 2)}) - eps;
    args->bounds_o->lower_z             = std::min({xf(2, 0), xf(2, 1), xf(2, 2)}) - eps;
    args->bounds_o->upper_x             = std::max({xf(0, 0), xf(0, 1), xf(0, 2)}) + eps;
    args->bounds_o->upper_y             = std::max({xf(1, 0), xf(1, 1), xf(1, 2)}) + eps;
    args->bounds_o->upper_z             = std::max({xf(2, 0), xf(2, 1), xf(2, 2)}) + eps;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void PointRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserDataByComponent<TScalar, TIndex>* userDataByComp =
        static_cast<UserDataByComponent<TScalar, TIndex>*>(args->geometryUserPtr);
    UserData<TScalar, TIndex>* userData = userDataByComp->userData;
    TIndex const component              = userDataByComp->component;
    auto& X                             = *(userData->X);
    auto& V                             = *(userData->V);
    auto& VP                            = *(userData->VP);
    Eigen::Index v                      = VP(component) + static_cast<Eigen::Index>(args->primID);
    auto xv                             = X.col(V(v));
    TScalar const rq                    = userData->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x             = xv(0) - rq;
    args->bounds_o->lower_y             = xv(1) - rq;
    args->bounds_o->lower_z             = xv(2) - rq;
    args->bounds_o->upper_x             = xv(0) + rq;
    args->bounds_o->upper_y             = xv(1) + rq;
    args->bounds_o->upper_z             = xv(2) + rq;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeRTCBoundsFunction(const struct RTCBoundsFunctionArguments* args)
{
    UserDataByComponent<TScalar, TIndex>* userDataByComp =
        static_cast<UserDataByComponent<TScalar, TIndex>*>(args->geometryUserPtr);
    UserData<TScalar, TIndex>* userData   = userDataByComp->userData;
    TIndex const component                = userDataByComp->component;
    auto& X                               = *(userData->X);
    auto& E                               = *(userData->E);
    auto& EP                              = *(userData->EP);
    TIndex e                              = EP(component) + static_cast<TIndex>(args->primID);
    Eigen::Matrix<TScalar, 3, 2> const xe = X(Eigen::placeholders::all, E.col(e));
    Eigen::Vector<TScalar, 3> const xmid  = TScalar(0.5) * (xe.col(0) + xe.col(1));
    TScalar const halfLength              = TScalar(0.5) * (xe.col(0) - xe.col(1)).norm();
    TScalar const queryRadius = halfLength + userData->rq + std::numeric_limits<TScalar>::epsilon();
    args->bounds_o->lower_x   = xmid(0) - queryRadius;
    args->bounds_o->lower_y   = xmid(1) - queryRadius;
    args->bounds_o->lower_z   = xmid(2) - queryRadius;
    args->bounds_o->upper_x   = xmid(0) + queryRadius;
    args->bounds_o->upper_y   = xmid(1) + queryRadius;
    args->bounds_o->upper_z   = xmid(2) + queryRadius;
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
    auto& VP                            = *(userData->VP);
    auto& FP                            = *(userData->FP);
    auto& mVertexLocks                  = *(userData->mVertexLocks);
    auto& mFacetLocks                   = *(userData->mFacetLocks);
    auto& dminv                         = *(userData->dminv);
    auto& GVHEp                         = *(userData->GVHEp);
    auto& GVHEadj                       = *(userData->GVHEadj);
    auto& GHEF                          = *(userData->GHEF);
    auto& dminf                         = *(userData->dminf);
    TScalar const r                     = userData->r;
    std::vector<std::vector<OffsetGeometryContact::ContactFace>>& FOGC = *(userData->FOGC);
    std::vector<std::vector<TIndex>>& VOGC                             = *(userData->VOGC);
    // For each potential contact pair (v,f)
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get vertex-facet pair (iv, f) where ix is the point index of vertex iv
        RTCCollision const& collision        = collisions[ci];
        TIndex const bv                      = static_cast<TIndex>(collision.geomID0);
        TIndex const bf                      = static_cast<TIndex>(collision.geomID1);
        TIndex const iv                      = VP(bv) + static_cast<TIndex>(collision.primID0);
        TIndex const ix                      = V(iv);
        TIndex const f                       = FP(bf) + static_cast<TIndex>(collision.primID1);
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
        // Vectorize contact face index a based on its type (triangle | edge | vertex)
        TIndex const a = VertexFacetContactFaceIndex(F, f, alocal, eFace);
        // Synchronize reads/writes to vertex iv's contact sets
        common::AtomicExecute(mVertexLocks(iv), [&]() {
            // Brute-force duplicate search
            bool const bDuplicateContact = std::any_of(
                FOGC[iv].begin(),
                FOGC[iv].end(),
                [&](OffsetGeometryContact::ContactFace const& contactFace) {
                    return contactFace.a == a and contactFace.eFace == eFace;
                });
            if (bDuplicateContact)
                return;
            // Update contact face sets
            auto const fUpdateContactSets = [&]() {
                FOGC[iv].emplace_back(a, eFace);
                common::AtomicExecute(mFacetLocks(f), [&]() { VOGC[f].emplace_back(ix); });
            };
            switch (eFace)
            {
                case 2 /* vertex */: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xi, a))
                        fUpdateContactSets();
                    break;
                }
                case 1 /* edge */: {
                    if (IsEdgeFeasible(X, F, GHEF, xi, f, a))
                        fUpdateContactSets();
                    break;
                }
                default /* triangle */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeEdgeRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    UserData<TScalar, TIndex>* userData = static_cast<UserData<TScalar, TIndex>*>(userPtr);
    auto& X                             = *(userData->X);
    auto& V                             = *(userData->V);
    auto& F                             = *(userData->F);
    auto& E                             = *(userData->E);
    auto& VP                            = *(userData->VP);
    auto& EP                            = *(userData->EP);
    auto& mEdgeLocks                    = *(userData->mEdgeLocks);
    auto& GVHEp                         = *(userData->GVHEp);
    auto& GVHEadj                       = *(userData->GVHEadj);
    auto& EHE                           = *(userData->EHE);
    auto& dmine                         = *(userData->dmine);
    TScalar const r                     = userData->r;
    std::vector<std::vector<OffsetGeometryContact::ContactFace>>& EOGC = *(userData->EOGC);
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get edge-edge pair (e1, e2)
        RTCCollision const& collision      = collisions[ci];
        TIndex const be1                   = static_cast<TIndex>(collision.geomID0);
        TIndex const be2                   = static_cast<TIndex>(collision.geomID1);
        TIndex const e1                    = EP(be1) + static_cast<TIndex>(collision.primID0);
        TIndex const e2                    = EP(be2) + static_cast<TIndex>(collision.primID1);
        Eigen::Vector<TIndex, 2> const e1v = E.col(e1);
        Eigen::Vector<TIndex, 2> const e2v = E.col(e2);
        // Avoid contact with same edge or adjacent edge
        bool const bIsSameEdgeOrAreAdjacent =
            e1v(0) == e2v(0) or e1v(0) == e2v(1) or e1v(1) == e2v(0) or e1v(1) == e2v(1);
        if (bIsSameEdgeOrAreAdjacent)
            continue;
        // Compute distance between edges e1 and e2 via closest point projection
        Eigen::Matrix<TScalar, 3, 2> const xe1 = X(Eigen::placeholders::all, e1v);
        Eigen::Matrix<TScalar, 3, 2> const xe2 = X(Eigen::placeholders::all, e2v);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        // s,t in [0,1] are the barycentric coordinates of the closest points on edges e1 and e2
        SVector<TScalar, 2> const st = geometry::ClosestPointQueries::LineSegments(
            FromEigen(xe1.col(0)),
            FromEigen(xe1.col(1)),
            FromEigen(xe2.col(0)),
            FromEigen(xe2.col(1)));
        // Closest points on edges e1 and e2
        Eigen::Vector<TScalar, 3> const xc1 =
            (TScalar(1) - st(0)) * xe1.col(0) + st(0) * xe1.col(1);
        Eigen::Vector<TScalar, 3> const xc2 =
            (TScalar(1) - st(1)) * xe2.col(0) + st(1) * xe2.col(1);
        TScalar const d2 = (xc1 - xc2).squaredNorm();
        // Update half-edge displacement bounds
        Eigen::Vector<TIndex, 2> const ehe1 = EHE.col(e1);
        Eigen::Vector<TIndex, 2> const ehe2 = EHE.col(e2);
        common::AtomicMin(dmine(ehe1(0)), d2);
        if (ehe1(1) >= 0) // Boundary edge has no 2nd half-edge
            common::AtomicMin(dmine(ehe1(1)), d2);
        common::AtomicMin(dmine(ehe2(0)), d2);
        if (ehe2(1) >= 0) // Boundary edge has no 2nd half-edge
            common::AtomicMin(dmine(ehe2(1)), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine faces (vertex or edge) closest to edges e1 and e2, i.e. faces of xc1 and xc2
        auto const [a1, eFace1, a2, eFace2] =
            ClosestFaceEdgeToEdge(st(0), st(1), e1, e2, {e1v(0), e1v(1)}, {e2v(0), e2v(1)});
        // Synchronized updates to edges e1 and e2's contact sets
        common::AtomicExecute(mEdgeLocks(e1), [&]() {
            auto const fUpdateContactSets = [&]() {
                // If we're contacting the interior of edge e2, store the contact pairs (ehe1(0),
                // ehe2(0)) and (ehe1(1), ehe2(0)). Because we only store vertex-to-half-edge
                // adjacencies, rather than vertex-(undirected-)edge adjacencies, we store 2 contact
                // pairs for edge e1 corresponding to both of its half-edges. This way, both
                // vertices/endpoints of edge e1 can reach half-edge ehe2(0). The vertices do not
                // need to reach ehe2(1), as it is redundant/unnecessary for computing an edge-edge
                // contact potential (we only need the 2 pairs of 2 vertices in no particular
                // order).
                bool const bIsA2Edge = (eFace2 == 0);
                TIndex const a       = bIsA2Edge * ehe2(0) + (not bIsA2Edge) * a2;
                EOGC[ehe1(0)].emplace_back(a, eFace2);
                if (ehe1(1) >= 0)
                    EOGC[ehe1(1)].emplace_back(a, eFace2);
            };
            switch (eFace2)
            {
                case 1 /* vertex */: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xc1, a2))
                        fUpdateContactSets();
                    break;
                }
                default /* edge */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
        common::AtomicExecute(mEdgeLocks(e2), [&]() {
            auto const fUpdateContactSets = [&]() {
                // See comment in edge e1's update above for explanation. We flip the roles of e1
                // and e2 here.
                bool const bIsA1Edge = (eFace1 == 0);
                TIndex const a       = bIsA1Edge * ehe1(0) + (not bIsA1Edge) * a1;
                EOGC[ehe2(0)].emplace_back(a, eFace1);
                if (ehe2(1) >= 0)
                    EOGC[ehe2(1)].emplace_back(a, eFace1);
            };
            switch (eFace1)
            {
                case 1 /* vertex */: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xc2, a1))
                        fUpdateContactSets();
                    break;
                }
                default /* edge */: {
                    fUpdateContactSets();
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
      bv(),
      dminv(),
      dminf(),
      dmine(),
      mVertexScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mEdgeScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mFaceScene{rtcNewScene(static_cast<RTCDevice>(device.Raw()))},
      mVertexLocks(),
      mEdgeLocks(),
      mFacetLocks(),
      mUserDataPerComponent()
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
    : OffsetGeometryContact(device)
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
    FOGC.resize(nVertices);
    for (auto& vfogc : FOGC)
        vfogc.reserve(params.nMaxVertexFaceContactsEstimate);
    VOGC.resize(nFacets);
    for (auto& fvogc : VOGC)
        fvogc.reserve(params.nMaxFaceVertexContactsEstimate);
    EOGC.resize(nHalfEdges);
    for (auto& eogc : EOGC)
        eogc.reserve(params.nMaxEdgeFaceContactsEstimate);
    dminv.resize(nVertices);
    dminf.resize(nFacets);
    dmine.resize(nHalfEdges);
    bv.resize(nVertices);
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
        std::addressof(mFacetLocks) /*mFacetLocks*/,
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
    mUserDataPerComponent.resize(nComponents);
    for (Eigen::Index c = nComponents - 1; c >= 0; --c)
    {
        IndexType const vertexOffset       = VP[c];
        IndexType const nComponentVertices = VP[c + 1] - vertexOffset;
        IndexType const faceOffset         = FP[c];
        IndexType const nComponentFaces    = FP[c + 1] - faceOffset;
        IndexType const edgeOffset         = EP[c];
        IndexType const nComponentEdges    = EP[c + 1] - edgeOffset;
        mUserDataPerComponent[c]           = {&userData, static_cast<IndexType>(c)};
        // Vertex geometry
        RTCGeometry vertexGeometry =
            rtcNewGeometry(static_cast<RTCDevice>(device.Raw()), RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(
            vertexGeometry,
            static_cast<unsigned int>(nComponentVertices));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
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
        rtcSetGeometryUserData(triangleGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
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
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
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
    mFacetLocks.resize(nFacets);
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
    mFacetLocks.setConstant(false);
    for (auto& vfogc : FOGC)
        vfogc.clear();
    for (auto& fvogc : VOGC)
        fvogc.clear();
    for (auto& eogc : EOGC)
        eogc.clear();
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
        std::addressof(mFacetLocks) /*mFacetLocks*/,
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
        mUserDataPerComponent[c].userData  = &userData;
        // Vertex geometry
        RTCGeometry vertexGeometry = rtcGetGeometry(mVertexScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(vertexGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
        rtcCommitGeometry(vertexGeometry);
        // Face geometry
        RTCGeometry faceGeometry = rtcGetGeometry(mFaceScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(faceGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
        rtcCommitGeometry(faceGeometry);
        // Edge geometry
        RTCGeometry edgeGeometry = rtcGetGeometry(mEdgeScene, static_cast<unsigned int>(c));
        rtcSetGeometryUserData(edgeGeometry, static_cast<void*>(&mUserDataPerComponent[c]));
        rtcCommitGeometry(edgeGeometry);
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
        std::addressof(mFacetLocks) /*mFacetLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        std::addressof(GVHEp) /*GVHEp*/,
        std::addressof(GVHEadj) /*GVHEadj*/,
        std::addressof(GHEF) /*GHEF*/,
        nullptr /*EHE*/,
        params.r /*r*/,
        params.rq /*rq*/};
    // Compute FOGC and VOGC (with duplicates)
    rtcCollide(
        mVertexScene,
        mFaceScene,
        detail::VertexFacetRTCCollideFunc<ScalarType, IndexType>,
        static_cast<void*>(&userData));
    // Remove duplicate entries in VOGC
    tbb::parallel_for(IndexType(0), IndexType(F.cols()), [this](IndexType f) {
        std::sort(VOGC[f].begin(), VOGC[f].end());
        VOGC[f].erase(std::unique(VOGC[f].begin(), VOGC[f].end()), VOGC[f].end());
    });
    // Finalize per-vertex displacement bounds
    dminv.noalias() = dminv.cwiseSqrt();
    // Finalize per-triangle displacement bounds
    dminf.noalias() = dminf.cwiseSqrt();
}

void OffsetGeometryContact::EdgeEdgeContactDetection(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
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
        std::addressof(E) /*E*/,
        std::addressof(VP) /*VP*/,
        nullptr /*FP*/,
        std::addressof(EP) /*EP*/,
        std::addressof(FOGC) /*FOGC*/,
        std::addressof(VOGC) /*VOGC*/,
        std::addressof(EOGC) /*EOGC*/,
        std::addressof(mVertexLocks) /*mVertexLocks*/,
        std::addressof(mEdgeLocks) /*mEdgeLocks*/,
        std::addressof(mFacetLocks) /*mFacetLocks*/,
        std::addressof(dminv) /*dminv*/,
        std::addressof(dminf) /*dminf*/,
        std::addressof(dmine) /*dmine*/,
        std::addressof(GVHEp) /*GVHEp*/,
        std::addressof(GVHEadj) /*GVHEadj*/,
        nullptr /*GHEF*/,
        std::addressof(EHE) /*EHE*/,
        params.r /*r*/,
        params.rq /*rq*/};
    // Compute EOGC
    rtcCollide(
        mEdgeScene,
        mEdgeScene,
        detail::EdgeEdgeRTCCollideFunc<ScalarType, IndexType>,
        static_cast<void*>(&userData));
    // De-duplicate EOGC
    auto const nHalfEdges = static_cast<IndexType>(3 * F.cols());
    tbb::parallel_for(IndexType(0), nHalfEdges, [this](IndexType he) {
        std::sort(EOGC[he].begin(), EOGC[he].end());
        EOGC[he].erase(std::unique(EOGC[he].begin(), EOGC[he].end()), EOGC[he].end());
    });
    // Finalize per-edge displacement bounds
    dmine.noalias() = dmine.cwiseSqrt();
}

void OffsetGeometryContact::ComputeDisplacementBounds(
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
    OgcParams const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.OffsetGeometryContact.ComputeDisplacementBounds");
    // dminv, dminf, dmine already computed during contact detection
    IndexType const nVertices = static_cast<IndexType>(V.size());
    for (IndexType v = 0; v < nVertices; ++v)
    {
        IndexType i  = V(v);
        bv(v)        = dminv(v);
        auto hebegin = GVHEp(i);
        auto heend   = GVHEp(i + 1);
        for (IndexType k = hebegin; k < heend; ++k)
        {
            IndexType const he = GVHEadj(k);
            IndexType const f  = geometry::FaceOfHalfEdge(he);
            bv(v)              = std::min({bv(v), dmine(he), dminf(f)});
        }
        bv(v) *= params.gammap;
    }
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

#include "MultiMesh.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/graph/Mesh.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] ClosestFaceFacetToVertex")
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

TEST_CASE("[sim][contact] IsVertexFeasible")
{
    using namespace pbat;

    // Arrange: single tetrahedral cube and its boundary triangulation and half-edge adjacency
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto [Vb, Fb] = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto [GVHEp, GVHEadj] =
        geometry::VertexHalfEdgeAdjacency(Fb.bottomRows<3>(), static_cast<Index>(X.cols()));
    Scalar const eps            = Scalar(1e-3);
    Index const i               = 7;
    Eigen::Vector<Scalar, 3> xv = X.col(i);

    SUBCASE("feasible: move along +(1,1,1)")
    {
        Eigen::Vector<Scalar, 3> x = xv + eps * Eigen::Vector<Scalar, 3>::Ones();
        CHECK(sim::contact::IsVertexFeasible(X, Fb.bottomRows<3>(), GVHEp, GVHEadj, x, i));
    }
    SUBCASE("infeasible: move along +(1,-1,-1) axis")
    {
        Eigen::Vector<Scalar, 3> x =
            xv + eps * Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(-1), Scalar(-1)};
        CHECK_FALSE(sim::contact::IsVertexFeasible(X, Fb.bottomRows<3>(), GVHEp, GVHEadj, x, i));
    }
}

TEST_CASE("[sim][contact] IsEdgeFeasible")
{
    using namespace pbat;

    // Arrange: single tetrahedral cube and its boundary triangulation and half-edge adjacency
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto [Vb, Fb]                = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto GHEF                    = geometry::HalfEdgeFaceAdjacency(Fb.bottomRows<3>());
    auto EHE                     = geometry::EdgeHalfEdgeAdjacency(Fb.bottomRows<3>(), GHEF);
    auto const fGetHalfEdgeIndex = [&](Eigen::Vector<Scalar, 3> xi, Eigen::Vector<Scalar, 3> xj) {
        for (auto f = 0; f < Fb.cols(); ++f)
        {
            Index hei = geometry::FirstHalfEdgeOfFace(f);
            Index hej = geometry::NextHalfEdge(hei);
            Index hek = geometry::NextHalfEdge(hej);
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hei)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hei)).isApprox(xj))
                return hei;
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hej)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hej)).isApprox(xj))
                return hej;
            if (X.col(geometry::IncomingVertex(Fb.bottomRows<3>(), hek)).isApprox(xi) and
                X.col(geometry::OutgoingVertex(Fb.bottomRows<3>(), hek)).isApprox(xj))
                return hek;
        }
        return Index(-1);
    };
    auto const fGetVertexIndex = [&](Eigen::Vector<Scalar, 3> xv) {
        for (Index v = 0; v < X.cols(); ++v)
            if (X.col(v).isApprox(xv))
                return v;
        return Index(-1);
    };
    Scalar const eps = Scalar(1e-3);
    SUBCASE("Half-edge (0,0,0) -> (1,0,0)")
    {
        Index const he =
            fGetHalfEdgeIndex(Eigen::Vector<Scalar, 3>{0, 0, 0}, Eigen::Vector<Scalar, 3>{1, 0, 0});
        Index const i = fGetVertexIndex(Eigen::Vector<Scalar, 3>{0, 0, 0});
        SUBCASE("feasible: move along +(1,-1,-1) axis")
        {
            Eigen::Vector<Scalar, 3> x =
                X.col(i) + eps * Eigen::Vector<Scalar, 3>{Scalar(1), Scalar(-1), Scalar(-1)};
            CHECK(sim::contact::IsEdgeFeasible(X, Fb.bottomRows<3>(), GHEF, x, he, i));
        }
        SUBCASE("infeasible: move along +(-1,0,0) axis")
        {
            Eigen::Vector<Scalar, 3> x =
                X.col(i) + eps * Eigen::Vector<Scalar, 3>{Scalar(-1), Scalar(0), Scalar(0)};
            CHECK_FALSE(sim::contact::IsEdgeFeasible(X, Fb.bottomRows<3>(), GHEF, x, he, i));
        }
    }
}

TEST_CASE("[sim][contact] OffsetGeometryContact")
{
    using namespace pbat;
    using namespace pbat::sim::contact;

    // Arrange
    OgcParams params = OgcParams()
                           .WithRadii(Scalar(0.01), Scalar(0.02))
                           .WithMaxContactEstimates(16, 16, 16)
                           .Construct();

    // single tetrahedral cube
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    MatrixX X1 = X;
    // Stack second cube on top in z direction, then shift in (1,1,1) direction, but stay inside
    // contact radius.
    MatrixX X2 = X.colwise() + (X.row(2).maxCoeff() - X.row(2).minCoeff()) * Vector<3>::UnitZ();
    X2.colwise() += Scalar(0.9) * params.r * Vector<3>::Ones().normalized();
    X.resize(3, X1.cols() + X2.cols());
    X << X1, X2;
    IndexMatrixX T1 = T;
    IndexMatrixX T2 = T.array() + static_cast<Index>(X1.cols());
    T.resize(4, T1.cols() + T2.cols());
    T << T1, T2;

    // Sort by connected components and reindex mesh/labels
    IndexVectorX XCC(X.cols()), TCC(T.cols()), Xord(X.cols()), Tord(T.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(X, T, XCC, TCC, Xord, Tord);
    graph::ReindexMeshByConnectedComponents(X, T, XCC, TCC, Xord, Tord);

    IndexVectorX V;
    IndexMatrixX F;
    IndexVectorX VP(nComponents + 1);
    IndexVectorX FP(nComponents + 1);
    IndexVectorX GXV;
    BoundaryTriangulation(T.bottomRows<4>(), XCC, V, F, VP, FP, GXV);
    IndexMatrixX E;
    IndexVectorX EP(nComponents + 1);
    IndexVectorX GVHEp, GVHEadj;
    IndexMatrixX GHEF, EHE;
    BoundaryTriangulationEdges(F.bottomRows<3>(), XCC, E, EP, GVHEp, GVHEadj, GHEF, EHE);

    geometry::DeviceConfig config;
    config.threads     = 1;
    config.userThreads = 1;
    config.verbose     = 0 /* 3 for debugging */;
    geometry::Device device(config);
    OffsetGeometryContact ogc(device, X, V, F, E, VP, FP, EP, params);
    ogc.PrepareIteration(X, V, F, E, VP, FP, EP, params);

    CHECK(std::all_of(ogc.FOGC.begin(), ogc.FOGC.end(), [](auto const& contacts) {
        return contacts.empty();
    }));
    CHECK(std::all_of(ogc.VOGC.begin(), ogc.VOGC.end(), [](auto const& contacts) {
        return contacts.empty();
    }));
    CHECK(std::all_of(ogc.EOGC.begin(), ogc.EOGC.end(), [](auto const& contacts) {
        return contacts.empty();
    }));

    SUBCASE("Vertex-Facet Contact Detection")
    {
        // Act: prepare iteration and perform vertex-facet contact detection
        ogc.VertexFacetContactDetection(X, V, F, VP, FP, GVHEp, GVHEadj, GHEF, params);

        // Assert: expect contacts between top vertices of bottom cube and bottom faces of top cube
        auto const fContactSetHasTriangles =
            [](std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return std::any_of(
                    contactFaces.begin(),
                    contactFaces.end(),
                    [](OffsetGeometryContact::ContactFace const& contactFace) {
                        return contactFace.IsTriangle();
                    });
            };
        auto const fContactSetHasEdges =
            [](std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return std::any_of(
                    contactFaces.begin(),
                    contactFaces.end(),
                    [](OffsetGeometryContact::ContactFace const& contactFace) {
                        return contactFace.IsEdge();
                    });
            };
        auto const fContactSetHasVertices =
            [](std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return std::any_of(
                    contactFaces.begin(),
                    contactFaces.end(),
                    [](OffsetGeometryContact::ContactFace const& contactFace) {
                        return contactFace.IsVertex();
                    });
            };
        Eigen::Index const nVerticesWithTriangleContacts = std::accumulate(
            ogc.FOGC.begin(),
            ogc.FOGC.end(),
            Eigen::Index(0),
            [&fContactSetHasTriangles](
                Eigen::Index acc,
                std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return acc + fContactSetHasTriangles(contactFaces);
            });
        Eigen::Index const nVerticesWithEdgeContacts = std::accumulate(
            ogc.FOGC.begin(),
            ogc.FOGC.end(),
            Eigen::Index(0),
            [&fContactSetHasEdges](
                Eigen::Index acc,
                std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return acc + fContactSetHasEdges(contactFaces);
            });
        Eigen::Index const nVerticesWithVertexContacts = std::accumulate(
            ogc.FOGC.begin(),
            ogc.FOGC.end(),
            Eigen::Index(0),
            [&fContactSetHasVertices](
                Eigen::Index acc,
                std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return acc + fContactSetHasVertices(contactFaces);
            });
        CHECK_EQ(nVerticesWithTriangleContacts, 1);
        CHECK_EQ(nVerticesWithEdgeContacts, 5);
        CHECK_EQ(nVerticesWithVertexContacts, 2);

        Eigen::Index const nTotalVertexFaceContacts = std::accumulate(
            ogc.FOGC.begin(),
            ogc.FOGC.end(),
            Eigen::Index(0),
            [](Eigen::Index acc,
               std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return acc + static_cast<Eigen::Index>(contactFaces.size());
            });
        Eigen::Index const nTotalFaceVertexContacts = std::accumulate(
            ogc.VOGC.begin(),
            ogc.VOGC.end(),
            Eigen::Index(0),
            [](Eigen::Index acc, std::vector<Index> const& contactVertices) {
                return acc + static_cast<Eigen::Index>(contactVertices.size());
            });
        CHECK_EQ(nTotalVertexFaceContacts, nTotalFaceVertexContacts);
    }
    SUBCASE("Edge-Edge Contact Detection")
    {
        // Act: prepare iteration and perform edge-edge contact detection
        ogc.EdgeEdgeContactDetection(X, V, F, E, VP, EP, GVHEp, GVHEadj, EHE, params);
        // Assert: expect some edge-edge contacts
        Eigen::Index const nHalfEdgeEdgeContacts = std::accumulate(
            ogc.EOGC.begin(),
            ogc.EOGC.end(),
            Eigen::Index(0),
            [](Eigen::Index acc,
               std::vector<OffsetGeometryContact::ContactFace> const& contactFaces) {
                return acc + static_cast<Eigen::Index>(contactFaces.size());
            });
        CHECK_GT(nHalfEdgeEdgeContacts, 0);
    }
    SUBCASE("All contact detection")
    {
        // Act
        ogc.VertexFacetContactDetection(X, V, F, VP, FP, GVHEp, GVHEadj, GHEF, params);
        ogc.EdgeEdgeContactDetection(X, V, F, E, VP, EP, GVHEp, GVHEadj, EHE, params);
        ogc.ComputeDisplacementBounds(V, F, GVHEp, GVHEadj, params);
        // Assert: displacement bounds are less than rq
        // NOTE: This is a weak test, but at least ensures that some plausible computation was done.
        Scalar const minDisplacementBound = ogc.bv.minCoeff();
        CHECK_LT(minDisplacementBound, params.rq);
    }
}

TEST_CASE("[sim][contact] ClosestFaceEdgeToEdge")
{
    using namespace pbat;

    // Setup: a single tetrahedral cube; use its boundary triangle mesh to build edges
    MatrixX X(3, 8);
    IndexMatrixX T(4, 5);
    // clang-format off
    X << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto const [V, F] = geometry::SimplexMeshBoundary(T, static_cast<Index>(X.cols()));
    auto const GHEF   = geometry::HalfEdgeFaceAdjacency(F.bottomRows<3>());
    auto const EHE    = geometry::EdgeHalfEdgeAdjacency(F.bottomRows<3>(), GHEF);
    auto const E = geometry::Edges(F.bottomRows<3>(), EHE); // 2 x |#edges| undirected edge list

    // Pick two concrete edges (by index) and build their endpoint arrays
    REQUIRE(E.cols() >= 2);
    Index const e1 = 0;
    Index const e2 = 1;
    std::array<Index, 2> const e1v{E(0, e1), E(1, e1)};
    std::array<Index, 2> const e2v{E(0, e2), E(1, e2)};

    auto const fCheck =
        [&](Scalar s, Scalar t, int a1Exp, int eFace1Exp, int a2Exp, int eFace2Exp) {
            auto const [a1, eFace1, a2, eFace2] =
                sim::contact::ClosestFaceEdgeToEdge(s, t, e1, e2, e1v, e2v);
            CHECK_EQ(a1, a1Exp);
            CHECK_EQ(eFace1, eFace1Exp);
            CHECK_EQ(a2, a2Exp);
            CHECK_EQ(eFace2, eFace2Exp);
        };

    // Both interior points on edges -> edge-edge contact
    fCheck(Scalar(0.3), Scalar(0.7), static_cast<int>(e1), 0, static_cast<int>(e2), 0);

    // One vertex, one edge
    fCheck(Scalar(0.0), Scalar(0.5), static_cast<int>(e1v[0]), 1, static_cast<int>(e2), 0);
    fCheck(Scalar(1.0), Scalar(0.5), static_cast<int>(e1v[1]), 1, static_cast<int>(e2), 0);
    fCheck(Scalar(0.5), Scalar(0.0), static_cast<int>(e1), 0, static_cast<int>(e2v[0]), 1);
    fCheck(Scalar(0.5), Scalar(1.0), static_cast<int>(e1), 0, static_cast<int>(e2v[1]), 1);

    // Both vertices -> vertex-vertex contact
    fCheck(Scalar(0.0), Scalar(0.0), static_cast<int>(e1v[0]), 1, static_cast<int>(e2v[0]), 1);
    fCheck(Scalar(1.0), Scalar(1.0), static_cast<int>(e1v[1]), 1, static_cast<int>(e2v[1]), 1);
}