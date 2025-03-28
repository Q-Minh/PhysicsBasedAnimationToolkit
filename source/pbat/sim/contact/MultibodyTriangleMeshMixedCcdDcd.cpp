#include "MultibodyTriangleMeshMixedCcdDcd.h"

namespace pbat::sim::contact {

void MultibodyTriangleMeshMixedCcdDcd::UpdateBodyAabbs()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.UpdateBodyAabbs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const& bvh = mVertexBvhs[o];
        auto LO         = mBodyAabbs.col(o).head<kDims>();
        auto UO         = mBodyAabbs.col(o).tail<kDims>();
        auto IBO        = bvh.InternalNodeBoundingBoxes();
        Index root      = bvh.Tree().Root();
        LO              = IBO.col(root).head<kDims>();
        UO              = IBO.col(root).tail<kDims>();
    }
}

void MultibodyTriangleMeshMixedCcdDcd::UpdateMeshVertexBvhs()
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.UpdateMeshVertexBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        mVertexBvhs[o].Update(VB.topRows<kDims>(), VB.bottomRows<kDims>());
    }
}

void MultibodyTriangleMeshMixedCcdDcd::UpdateMeshEdgeBvhs()
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.UpdateMeshEdgeBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        mEdgeBvhs[o].Update(EB.topRows<kDims>(), EB.bottomRows<kDims>());
    }
}

void MultibodyTriangleMeshMixedCcdDcd::UpdateMeshTriangleBvhs(bool bForDcd)
{
    if (bForDcd)
    {
        PBAT_PROFILE_NAMED_SCOPE(
            "pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.UpdateMeshTriangleBvhs.ForDcd");
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
        auto const nDcdBodies = mDcdBodies.size();
        for (auto k = 0; k < nDcdBodies; ++k)
        {
            auto o  = mDcdBodies[k];
            auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
            mTriangleBvhs[o].Update(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        }
#include "pbat/warning/Pop.h"
    }
    else
    {
        PBAT_PROFILE_NAMED_SCOPE(
            "pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.UpdateMeshTriangleBvhs");
        auto const nObjects = mBodyAabbs.cols();
        for (auto o = 0; o < nObjects; ++o)
        {
            auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
            mTriangleBvhs[o].Update(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        }
    }
}

void MultibodyTriangleMeshMixedCcdDcd::RecomputeBodyBvh()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.RecomputeBodyBvh");
    auto L = mBodyAabbs.topRows<kDims>();
    auto U = mBodyAabbs.bottomRows<kDims>();
    mBodyBvh.Construct(L, U);
    mBodyBvh.Update(L, U);
}

void MultibodyTriangleMeshMixedCcdDcd::SortActiveSets()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyTriangleMeshMixedCcdDcd.SortActiveSets");
    auto const nVerts = mVertexTriangleCountingSortRange.size() - 1;
    // Sort vertex-triangle active contacts
    auto const fKeyFromVertexTriangle = [](VertexTriangleContact const& vf) {
        return vf.v;
    };
    mVertexTriangleCountingSortRange.setZero();
    pbat::common::CountingSort(
        mVertexTriangleCountingSortRange.data(),
        mVertexTriangleCountingSortRange.data() + nVerts,
        mActiveVertexTrianglePairs.begin(),
        mActiveVertexTrianglePairs.end(),
        Index(0),
        fKeyFromVertexTriangle);
    mVertexTriangleCountingSortRange[0] = 0;
    pbat::common::PrefixSumFromSortedKeys(
        mActiveVertexTrianglePairs.begin(),
        mActiveVertexTrianglePairs.end(),
        mVertexTriangleCountingSortRange.data() + 1,
        mVertexTriangleCountingSortRange.data() + nVerts + 1,
        fKeyFromVertexTriangle);
    // Sort vertex-edge active contacts
    auto const fKeyFromVertexEdge = [](VertexEdgeContact const& ve) {
        return ve.v;
    };
    mVertexEdgeCountingSortRange.setZero();
    pbat::common::CountingSort(
        mVertexEdgeCountingSortRange.data(),
        mVertexEdgeCountingSortRange.data() + nVerts,
        mActiveEdgeEdgePairs.begin(),
        mActiveEdgeEdgePairs.end(),
        Index(0),
        fKeyFromVertexEdge);
    mVertexEdgeCountingSortRange[0] = 0;
    pbat::common::PrefixSumFromSortedKeys(
        mActiveEdgeEdgePairs.begin(),
        mActiveEdgeEdgePairs.end(),
        mVertexEdgeCountingSortRange.data() + 1,
        mVertexEdgeCountingSortRange.data() + nVerts + 1,
        fKeyFromVertexEdge);
}

} // namespace pbat::sim::contact
