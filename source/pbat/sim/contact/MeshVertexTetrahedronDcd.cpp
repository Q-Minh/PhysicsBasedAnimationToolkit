#include "MeshVertexTetrahedronDcd.h"

namespace pbat::sim::contact {

MeshVertexTetrahedronDcd::MeshVertexTetrahedronDcd(
    Eigen::Ref<Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic>> X,
    Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T)
    : mMultibodySystem(X, T),
      mTriangleAabbs(2 * kDims, mMultibodySystem.NumContactTriangles()),
      mTetrahedronAabbs(2 * kDims, mMultibodySystem.NumTetrahedra()),
      mBodyAabbs(2 * kDims, mMultibodySystem.NumBodies()),
      mTriangleMeshBvhs(static_cast<std::size_t>(mMultibodySystem.NumBodies())),
      mTetrahedronMeshHashGrids(static_cast<std::size_t>(mMultibodySystem.NumBodies())),
      mBodiesBvh(),
      mIsTetrahedronMeshHashGridDirty(mMultibodySystem.NumBodies()),
      mBodyVertexToOtherBodiesValence(static_cast<std::size_t>(mMultibodySystem.NumBodies())),
      mVFC(kMaxVertexTriangleContacts, mMultibodySystem.NumContactVertices())
{
    // Compute initial AABBs
    ComputeTriangleAabbs(X.derived());
    // Construct spatial acceleration structures
    auto const nObjects = mMultibodySystem.NumBodies();
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const oStl           = static_cast<std::size_t>(o);
        auto const [fbegin, fend] = mMultibodySystem.ContactTrianglesRangeFor(o);
        auto const [tbegin, tend] = mMultibodySystem.TetrahedraRangeFor(o);
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(fbegin, fend - 1));
        // Construct object o's mesh BVH tree topology
        mTriangleMeshBvhs[oStl].Construct(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        auto const nTets = tend - tbegin;
        mTetrahedronMeshHashGrids[oStl].Configure(nTets);
    }
    ComputeBodyAabbs();
    mBodiesBvh.Construct(mBodyAabbs.topRows<kDims>(), mBodyAabbs.bottomRows<kDims>());
    mIsTetrahedronMeshHashGridDirty.setConstant(true);
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const oStl = static_cast<std::size_t>(o);
        std::unordered_map<IndexType, IndexType>& bodyVertexToOtherBodiesValenceMap =
            mBodyVertexToOtherBodiesValence[oStl];
        auto const [vbegin, vend] = mMultibodySystem.ContactVerticesRangeFor(o);
        bodyVertexToOtherBodiesValenceMap.reserve(static_cast<std::size_t>(vend - vbegin));
    }
    mVFC.setConstant(-1);
}

void MeshVertexTetrahedronDcd::UpdateActiveSet(
    Eigen::Ref<Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.UpdateActiveSet");
    mVFC.setConstant(-1);
    ComputeTriangleAabbs(X);
    UpdateMeshTriangleBvhs();
    ComputeBodyAabbs();
    RecomputeBodiesBvh();
    ForEachBodyPair([&](IndexType oi, IndexType oj) {
        if (mIsTetrahedronMeshHashGridDirty(oi))
        {
            ComputeTetrahedronMeshAabbs(oi, X, T);
            UpdateMeshTetrahedronHashGrid(oi);
            mIsTetrahedronMeshHashGridDirty(oi) = false;
        }
        if (mIsTetrahedronMeshHashGridDirty(oj))
        {
            ComputeTetrahedronMeshAabbs(oj, X, T);
            UpdateMeshTetrahedronHashGrid(oj);
            mIsTetrahedronMeshHashGridDirty(oj) = false;
        }
        MarkPenetratingVertices(oi, oj, X, T);
        MarkPenetratingVertices(oj, oi, X, T);
        FindNearestTrianglesToPenetratingVertices(oi, oj, X);
        FindNearestTrianglesToPenetratingVertices(oj, oi, X);
    });
    mIsTetrahedronMeshHashGridDirty.setConstant(true);
}

MultibodyTetrahedralMeshSystem<MeshVertexTetrahedronDcd::IndexType> const&
MeshVertexTetrahedronDcd::MultibodySystem() const
{
    return mMultibodySystem;
}

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
        auto const [fbegin, fend] = mMultibodySystem.ContactTrianglesRangeFor(o);
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(fbegin, fend - 1));
        mTriangleMeshBvhs[o].Update(FB.topRows<kDims>(), FB.bottomRows<kDims>());
    }
#include "pbat/warning/Pop.h"
}

void MeshVertexTetrahedronDcd::UpdateMeshTetrahedronHashGrid(IndexType bodyIndex)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshVertexTetrahedronDcd.UpdateMeshTetrahedronHashGrid");
    auto const [tbegin, tend] = mMultibodySystem.TetrahedraRangeFor(bodyIndex);
    auto TB           = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tbegin, tend - 1));
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

#include "pbat/geometry/MeshBoundary.h"
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MeshVertexTetrahedronDcd")
{
    using namespace pbat::geometry::model;
    [[maybe_unused]] auto const [Vcube, Tcube] = Cube(EMesh::Tetrahedral, 1);
}
