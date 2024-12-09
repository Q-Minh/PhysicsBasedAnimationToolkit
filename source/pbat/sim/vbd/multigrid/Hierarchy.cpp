#include "Hierarchy.h"

#include "Mesh.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Hierarchy::Hierarchy(
    Data root,
    std::vector<Eigen::Ref<MatrixX const>> const& X,
    std::vector<Eigen::Ref<IndexMatrixX const>> const& E,
    Eigen::Ref<IndexMatrixX const> const& cycle,
    Eigen::Ref<IndexVectorX const> const& transitionSchedule,
    Eigen::Ref<IndexVectorX const> const& smoothingSchedule)
    : mRoot(std::move(root)),
      mLevels(),
      mCycle(cycle),
      mTransitionSchedule(transitionSchedule),
      mSmoothingSchedule(smoothingSchedule),
      mTransitions()
{
    mLevels.reserve(X.size());
    for (auto l = 0ULL; l < X.size(); ++l)
    {
        mLevels.push_back(
            Level(VolumeMesh(X[l], E[l]))
                .WithCageQuadrature(mRoot, ECageQuadratureStrategy::PolynomialSubCellIntegration)
                .WithDirichletQuadrature(mRoot)
                .WithMomentumEnergy(mRoot)
                .WithElasticEnergy(mRoot));
    }
    if (mCycle.size() == 0)
    {
        Index const nLevels = static_cast<Index>(mLevels.size());
        mCycle.resize(2, nLevels + 1);
        mCycle.col(0) = IndexVector<2>{Index(-1), nLevels - 1};
        for (Index l = nLevels - 1, k = 1; l >= 0; --l, ++k)
            mCycle.col(k) << l, l - 1;
    }
    if (mTransitionSchedule.size() == 0)
    {
        mTransitionSchedule.setConstant(mCycle.size(), Index(10));
    }
    if (mSmoothingSchedule.size() == 0)
    {
        mSmoothingSchedule.setConstant(mCycle.size() + 1, Index(10));
    }

    mTransitions.reserve(mTransitionSchedule.size() * 3ULL);
    for (Index t = 0; t < mTransitionSchedule.cols(); ++t)
    {
        IndexVector<2> const transition = mTransitionSchedule.col(t);
        bool const bIsNewTransition     = mTransitions.find(transition) == mTransitions.end();
        if (bIsNewTransition)
        {
            bool const bIsProlongation = transition(0) > transition(1);
            if (bIsProlongation)
            {
                Level const& lc      = mLevels[static_cast<std::size_t>(transition(0))];
                Level const& lf      = mLevels[static_cast<std::size_t>(transition(1))];
                VolumeMesh const& CM = lc.mesh;
                VolumeMesh const& FM = transition(1) > -1 ? lf.mesh : mRoot.mesh;
                mTransitions.insert({transition, Prolongation(FM, CM)});
            }
            else
            {
                Level const& lf          = mLevels[static_cast<std::size_t>(transition(0))];
                Level const& lc          = mLevels[static_cast<std::size_t>(transition(1))];
                VolumeMesh const& FM     = transition(0) > -1 ? lf.mesh : mRoot.mesh;
                VolumeMesh const& CM     = lc.mesh;
                CageQuadrature const& CQ = lc.Qcage;
                mTransitions.insert({transition, Restriction(mRoot, FM, CM, CQ)});
            }
        }
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat