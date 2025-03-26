#include "MultibodyTriangleMeshMixedCcdDcd.h"

namespace pbat::sim::contact {

void MultibodyTriangleMeshMixedCcdDcd::UpdateBodyAabbs()
{
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
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        mVertexBvhs[o].Update(VB.topRows<kDims>(), VB.bottomRows<kDims>());
    }
}

void MultibodyTriangleMeshMixedCcdDcd::UpdateMeshEdgeBvhs()
{
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
    auto L = mBodyAabbs.topRows<kDims>();
    auto U = mBodyAabbs.bottomRows<kDims>();
    mBodyBvh.Construct(L, U);
    mBodyBvh.Update(L, U);
}

void MultibodyTriangleMeshMixedCcdDcd::SortActiveSets()
{
    auto const nVerts                 = mVertexTriangleCountingSortRange.size() - 1;
    auto const fKeyFromVertexTriangle = [](VertexTriangleContact const& vf) {
        return vf.v;
    };
    pbat::common::CountingSort(
        mVertexTriangleCountingSortRange.data(),
        mVertexTriangleCountingSortRange.data() + nVerts,
        mActiveVertexTrianglePairs.begin(),
        mActiveVertexTrianglePairs.end(),
        Index(0),
        fKeyFromVertexTriangle);
    pbat::common::PrefixSumFromSortedKeys(
        mActiveVertexTrianglePairs.begin(),
        mActiveVertexTrianglePairs.end(),
        mVertexTriangleCountingSortRange.data() + 1,
        mVertexTriangleCountingSortRange.data() + nVerts + 1,
        fKeyFromVertexTriangle);
    auto const fKeyFromVertexEdge = [](VertexEdgeContact const& ve) {
        return ve.v;
    };
    pbat::common::CountingSort(
        mVertexTriangleCountingSortRange.data(),
        mVertexTriangleCountingSortRange.data() + nVerts,
        mActiveEdgeEdgePairs.begin(),
        mActiveEdgeEdgePairs.end(),
        Index(0),
        fKeyFromVertexEdge);
    pbat::common::PrefixSumFromSortedKeys(
        mActiveEdgeEdgePairs.begin(),
        mActiveEdgeEdgePairs.end(),
        mVertexEdgeCountingSortRange.data() + 1,
        mVertexEdgeCountingSortRange.data() + nVerts + 1,
        fKeyFromVertexEdge);
}

} // namespace pbat::sim::contact
