#include "Hierarchy.h"

#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

Hierarchy::Hierarchy(
    Data root,
    std::vector<VolumeMesh> const& cages,
    Eigen::Ref<IndexVectorX const> const& cycle,
    Eigen::Ref<IndexVectorX const> const& smoothingSchedule,
    Eigen::Ref<IndexVectorX const> const& transitionSchedule,
    std::vector<CageQuadratureParameters> const& cageQuadParams)
    : mRoot(std::move(root)),
      mLevels(),
      mCycle(cycle),
      mSmoothingSchedule(smoothingSchedule),
      mTransitionSchedule(transitionSchedule),
      mTransitions()
{
    auto const nCoarseLevels = cages.size();
    mLevels.reserve(nCoarseLevels);
    for (auto l = 0ULL; l < cages.size(); ++l)
    {
        mLevels.push_back(
            Level(cages[l])
                .WithCageQuadrature(
                    mRoot,
                    cageQuadParams.empty() ? CageQuadratureParameters{} : cageQuadParams[l])
                .WithDirichletQuadrature(mRoot)
                .WithMomentumEnergy(mRoot)
                .WithElasticEnergy(mRoot)
                .WithDirichletEnergy(mRoot));
    }
    if (mCycle.size() == 0)
    {
        Index const nLevels = static_cast<Index>(mLevels.size());
        mCycle.resize(nLevels + 2);
        mCycle(0) = Index(-1);
        for (Index l = nLevels - 1, k = 1; l >= 0; --l, ++k)
            mCycle(k) = l;
        mCycle(nLevels + 1) = Index(-1);
    }
    if (mSmoothingSchedule.size() == 0)
    {
        mSmoothingSchedule.setConstant(mCycle.size(), Index(10));
    }
    if (mTransitionSchedule.size() == 0)
    {
        mTransitionSchedule.setConstant(mCycle.size() - 1, Index(10));
    }

    mTransitions.reserve(mTransitionSchedule.size() * 3ULL);
    Index lCurrent = mCycle(0);
    for (Index lNext : mCycle.tail(mCycle.size() - 1))
    {
        IndexVector<2> const transition = {lCurrent, lNext};
        bool const bIsNewTransition     = mTransitions.find(transition) == mTransitions.end();
        if (bIsNewTransition)
        {
            bool const bIsProlongation = transition(0) > transition(1);
            auto li                    = static_cast<std::size_t>(transition(0));
            auto lj                    = static_cast<std::size_t>(transition(1));
            if (bIsProlongation)
            {
                VolumeMesh const& CM = mLevels[li].mesh;
                VolumeMesh const& FM = transition(1) > -1 ? mLevels[lj].mesh : mRoot.mesh;
                mTransitions.insert({transition, Prolongation(FM, CM)});
            }
            else
            {
                Level const& lc          = mLevels[lj];
                CageQuadrature const& CQ = lc.Qcage;
                mTransitions.insert({transition, Restriction(CQ)});
            }
        }
        lCurrent = lNext;
    }
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][lod] Hierarchy")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::lod::Hierarchy;
    using sim::vbd::lod::VolumeMesh;

    auto [XR, ER] = geometry::model::Cube();
    XR.colwise() -= XR.rowwise().mean();
    std::vector<VolumeMesh> cages{
        {VolumeMesh(Scalar(1.1) * XR, ER), VolumeMesh(Scalar(1.2) * XR, ER)}};
    Data root = Data().WithVolumeMesh(XR, ER).Construct();

    SUBCASE("Default constructor parameters")
    {
        Hierarchy H(root, cages);
        CHECK_GT(H.mCycle.size(), 0ULL);
        CHECK_EQ(H.mCycle.size(), H.mSmoothingSchedule.size());
        CHECK_EQ(H.mCycle.size(), H.mTransitionSchedule.size() + 1ULL);
        CHECK_EQ(H.mLevels.size(), cages.size());
        CHECK_GT(H.mTransitions.size(), 0ULL);
        CHECK_EQ(H.mCycle(0), Index(-1));
        CHECK_EQ(H.mCycle(H.mCycle.size() - 1), Index(-1));
    }
    SUBCASE("User-specified constructor parameters")
    {
        IndexVectorX cycle(7);
        cycle << -1, 0, 1, 0, 1, 0, -1;
        IndexVectorX riters(6);
        riters << 10, 10, 10, 10, 10, 10;
        IndexVectorX siters(7);
        siters << 10, 10, 10, 10, 10, 10, 10;
        Hierarchy H(root, cages, cycle, siters, riters);
        CHECK_EQ(H.mCycle.size(), cycle.size());
        CHECK_EQ(H.mSmoothingSchedule.size(), siters.size());
        CHECK_EQ(H.mTransitionSchedule.size(), riters.size());
        CHECK_EQ(H.mLevels.size(), cages.size());
        CHECK_EQ(H.mTransitions.size(), 4ULL);
    }
}