#include "MultibodyTriangleMeshMixedCcdDcd.h"

namespace pbat::sim::contact {

void MultibodyTriangleMeshMixedCcdDcd::UpdateMeshBvhs()
{
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        mVertexBvhs[o].Update(VB);
        mEdgeBvhs[o].Update(EB);
        mTriangleBvhs[o].Update(FB);
    }
}

void MultibodyTriangleMeshMixedCcdDcd::RecomputeBodyBvh()
{
    mBodyBvh.Construct(mBodyAabbs);
    mBodyBvh.Update(mBodyAabbs);
}

} // namespace pbat::sim::contact
