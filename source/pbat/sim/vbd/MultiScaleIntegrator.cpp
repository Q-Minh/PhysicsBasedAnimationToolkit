#include "MultiScaleIntegrator.h"

#include "Hierarchy.h"
#include "Kernels.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

void MultiScaleIntegrator::Step(Scalar dt, Index substeps, Hierarchy& H)
{
    Scalar const sdt  = dt / static_cast<Scalar>(substeps);
    Scalar const sdt2 = sdt * sdt;
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        H.root.xt = H.root.x;
        // Compute inertial target positions
        H.root.xtilde = H.root.xt + sdt * H.root.vt + sdt2 * H.root.aext;
        // Propagate inertial target positions to all levels
        for (auto l = 0ULL; l < H.levels.size(); ++l)
        {
            Level& L                = H.levels[l];
            IndexVectorX const& erg = L.RPB.erg;
            MatrixX const& Nrg      = L.RPB.Nrg;
            auto nQuadPts           = L.E.wg.size();
            tbb::parallel_for(Index(0), nQuadPts, [&](Index g) {
                auto e       = erg(g);
                auto inds    = H.root.T.col(e).head<4>();
                auto N       = Nrg.col(g).head<4>();
                auto xtildee = H.root.xtilde(Eigen::placeholders::all, inds).block<3, 4>(0, 0);
                L.E.xtildeg.col(g) = xtildee * N;
            });
        }
        // Initialize block coordinate descent's, i.e. BCD's, solution
        auto nVertices = H.root.x.cols();
        tbb::parallel_for(Index(0), nVertices, [&](Index i) {
            using math::linalg::mini::FromEigen;
            using math::linalg::mini::ToEigen;
            auto x = kernels::InitialPositionsForSolve(
                FromEigen(H.root.xt.col(i).head<3>()),
                FromEigen(H.root.vt.col(i).head<3>()),
                FromEigen(H.root.v.col(i).head<3>()),
                FromEigen(H.root.aext.col(i).head<3>()),
                sdt,
                sdt2,
                H.root.strategy);
            H.root.x.col(i) = ToEigen(x);
        });
        // Minimize time integration energy using multiscale approach
        H.smoothers.front().Apply(sdt, H.root);
        for (auto t = 0ULL; t < H.transitions.size(); ++t)
        {
            std::visit([&](auto&& transition) { transition.Apply(H); }, H.transitions[t]);
            Index l = std::visit(
                [&](auto&& transition) { return transition.EndLevel(); },
                H.transitions[t]);
            bool const bNextLevelIsRoot = l < 0;
            if (bNextLevelIsRoot)
            {
                H.smoothers[t + 1].Apply(sdt, H.root);
            }
            else
            {
                auto lStl = static_cast<std::size_t>(l);
                H.smoothers[t + 1].Apply(sdt, H.levels[lStl]);
            }
        }
        // Update velocity
        H.root.vt = H.root.v;
        H.root.v  = (H.root.x - H.root.xt) / sdt;
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat