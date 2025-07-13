#include "MeshVertexTetrahedronDcd.h"

namespace pbat::sim::contact {

void MeshVertexTetrahedronDcd::ComputeBodyAabbs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.ComputeBodyAabbs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const& bvh = mTriangleMeshBvhs[o];
        auto LO         = mBodyAabbs.col(o).head<kDims>();
        auto UO         = mBodyAabbs.col(o).tail<kDims>();
        Index root      = bvh.Tree().Root();
        LO              = bvh.Lower(root);
        UO              = bvh.Upper(root);
    }
#include "pbat/warning/Pop.h"
}

void MeshVertexTetrahedronDcd::UpdateMeshTriangleBvhs()
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.UpdateMeshTriangleBvhs");
    auto const nObjects = mBodyAabbs.cols();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        mTriangleMeshBvhs[o].Update(FB.topRows<kDims>(), FB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MeshVertexTetrahedronDcd::UpdateMeshTetrahedronHashGrid(IndexType bodyIndex)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshVertexTetrahedronDcd.UpdateMeshTetrahedronHashGrid");
    auto TB = mTetrahedronAabbs(
        Eigen::placeholders::all,
        Eigen::seq(mTP(bodyIndex), mTP(bodyIndex + 1) - 1));
    auto bodyIndexStl = static_cast<std::size_t>(bodyIndex);
    mTetrahedronMeshHashGrids[bodyIndexStl].Construct(TB.topRows<kDims>(), TB.bottomRows<kDims>());
}

void MeshVertexTetrahedronDcd::RecomputeBodiesBvh()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.RecomputeBodiesBvh");
    auto L = mBodyAabbs.topRows<kDims>();
    auto U = mBodyAabbs.bottomRows<kDims>();
    mBodiesBvh.Construct(L, U);
    mBodiesBvh.Update(L, U);
}

} // namespace pbat::sim::contact

#include <doctest/doctest.h>

#include "pbat/geometry/model/Cube.h"
#include "pbat/geometry/MeshBoundary.h"

TEST_CASE("[sim][contact] MeshVertexTetrahedronDcd") 
{
    using namespace pbat::geometry::model;
    auto const [Vcube, Tcube] = Cube(EMesh::Tetrahedral, 1);

    pbat::geometry::SimplexMeshBoundary(Tcube, Vcube.cols());
}
