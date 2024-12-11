#include "Integrator.h"

#include "Hierarchy.h"
#include "Smoother.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

void Integrator::Step(Scalar dt, Index substeps, Hierarchy& H)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Integrator.Step");
    Scalar sdt           = dt / static_cast<Scalar>(substeps);
    Scalar sdt2          = sdt * sdt;
    auto const nVertices = H.mRoot.x.cols();
    for (auto s = 0; s < substeps; ++s)
    {
        using namespace math::linalg;
        using mini::FromEigen;
        using mini::ToEigen;
        // Store previous positions
        H.mRoot.xt = H.mRoot.x;
        // Compute inertial target positions
        H.mRoot.xtilde = H.mRoot.xt + sdt * H.mRoot.v + sdt2 * H.mRoot.aext;
        // Initialize block coordinate descent's, i.e. BCD's, solution
        tbb::parallel_for(Index(0), nVertices, [&](Index i) {
            using pbat::sim::vbd::kernels::InitialPositionsForSolve;
            auto x = InitialPositionsForSolve(
                FromEigen(H.mRoot.xt.col(i).head<3>()),
                FromEigen(H.mRoot.vt.col(i).head<3>()),
                FromEigen(H.mRoot.v.col(i).head<3>()),
                FromEigen(H.mRoot.aext.col(i).head<3>()),
                sdt,
                sdt2,
                H.mRoot.strategy);
            H.mRoot.x.col(i) = ToEigen(x);
        });
        // Minimize Backward Euler, i.e. BDF1, objective, using hierarchy
        // 1. Propagate xtilde
        for (auto& l : H.mLevels)
            l.Ekinetic.UpdateInertialTargetPositions(H.mRoot);
        // 2. Cycle
        IndexVectorX const& cycle = H.mCycle;
        Smoother S{};
        Index const nLevelVisits = cycle.size();
        for (Index c = 0; c < nLevelVisits; ++c)
        {
            // Smooth current level
            Index lCurrent                 = cycle(c);
            auto lCurrentStl               = static_cast<std::size_t>(lCurrent);
            Index siters                   = H.mSmoothingSchedule(c);
            bool const bCurrentLevelIsRoot = lCurrent < 0;
            bCurrentLevelIsRoot ? S.Apply(siters, sdt, H.mRoot) :
                                  S.Apply(siters, sdt, H.mLevels[lCurrentStl]);
            // Transition to next level
            bool const bShouldTransition = (c + 1) < nLevelVisits;
            if (bShouldTransition)
            {
                Index lNext                 = cycle(c + 1);
                bool const bNextLevelIsRoot = lNext < 0;
                auto& T                     = H.mTransitions.at({lCurrent, lNext});
                Index riters                = H.mTransitionSchedule(c);
                auto lNextStl               = static_cast<std::size_t>(lNext);
                if (Prolongation* P = std::get_if<Prolongation>(&T))
                {
                    bNextLevelIsRoot ? P->Apply(H.mLevels[lCurrentStl], H.mRoot) :
                                       P->Apply(H.mLevels[lCurrentStl], H.mLevels[lNextStl]);
                }
                if (Restriction* R = std::get_if<Restriction>(&T))
                {
                    bCurrentLevelIsRoot ?
                        R->Apply(riters, H.mRoot, H.mLevels[lNextStl]) :
                        R->Apply(riters, H.mLevels[lCurrentStl], H.mLevels[lNextStl]);
                }
            }
        }
        // Update velocity
        H.mRoot.vt = H.mRoot.v;
        H.mRoot.v  = (H.mRoot.x - H.mRoot.xt) / sdt;
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Integrator")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::multigrid::Hierarchy;
    using sim::vbd::multigrid::Integrator;
    using sim::vbd::multigrid::VolumeMesh;

    // Arrange
    auto [XR, ER] = geometry::model::Cube();
    XR.colwise() -= XR.rowwise().mean();
    std::vector<VolumeMesh> cages{
        {VolumeMesh(Scalar(1.1) * XR, ER), VolumeMesh(Scalar(1.2) * XR, ER)}};
    Data root = Data().WithVolumeMesh(XR, ER).Construct();
    Hierarchy H(root, cages);
    Scalar dt{1e-2};
    Index substeps{1};

    // Act
    Integrator mvbd{};
    mvbd.Step(dt, substeps, H);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = H.mRoot.x - XR;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}