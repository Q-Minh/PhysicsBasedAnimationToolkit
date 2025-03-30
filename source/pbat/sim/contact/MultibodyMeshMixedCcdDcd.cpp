#include "MultibodyMeshMixedCcdDcd.h"

namespace pbat::sim::contact {

void MultibodyMeshMixedCcdDcd::ComputeBodyAabbs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeBodyAabbs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const& bvh = mVertexBvhs[o];
        auto LO         = mBodyAabbs.col(o).head<kDims>();
        auto UO         = mBodyAabbs.col(o).tail<kDims>();
        auto const& IBO = bvh.InternalNodeBoundingBoxes();
        Index root      = bvh.Tree().Root();
        LO              = IBO.col(root).head<kDims>();
        UO              = IBO.col(root).tail<kDims>();
    }
#include "pbat/warning/Pop.h"
}

void MultibodyMeshMixedCcdDcd::UpdateMeshVertexBvhs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.UpdateMeshVertexBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        mVertexBvhs[o].Update(VB.topRows<kDims>(), VB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MultibodyMeshMixedCcdDcd::UpdateMeshEdgeBvhs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.UpdateMeshEdgeBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        mEdgeBvhs[o].Update(EB.topRows<kDims>(), EB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MultibodyMeshMixedCcdDcd::UpdateMeshTriangleBvhs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.UpdateMeshTriangleBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        mTriangleBvhs[o].Update(FB.topRows<kDims>(), FB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MultibodyMeshMixedCcdDcd::UpdateMeshTetrahedronBvhs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.UpdateMeshTetrahedronBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto TB = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(mTP(o), mTP(o + 1) - 1));
        mTetrahedronBvhs[o].Update(TB.topRows<kDims>(), TB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MultibodyMeshMixedCcdDcd::RecomputeBodyBvh()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.RecomputeBodyBvh");
    auto L = mBodyAabbs.topRows<kDims>();
    auto U = mBodyAabbs.bottomRows<kDims>();
    mBodyBvh.Construct(L, U);
    mBodyBvh.Update(L, U);
}

} // namespace pbat::sim::contact
