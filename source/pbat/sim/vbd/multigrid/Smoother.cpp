#include "Smoother.h"

#include "Kernels.h"
#include "Level.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Kernels.h"
#include "pbat/profiling/Profiling.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

void Smoother::Apply(Index iters, Scalar dt, Data& data) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Smoother.Apply");
    Scalar const dt2 = dt * dt;
    // Minimize Backward Euler, i.e. BDF1, objective
    for (auto k = 0; k < iters; ++k)
    {
        auto const nPartitions = data.Pptr.size() - 1;
        for (Index p = 0; p < nPartitions; ++p)
        {
            auto const pBegin = data.Pptr(p);
            auto const pEnd   = data.Pptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index k) {
                using namespace math::linalg;
                using mini::FromEigen;
                using mini::ToEigen;

                auto i     = data.Padj(k);
                auto begin = data.GVGp(i);
                auto end   = data.GVGp(i + 1);
                // Elastic energy
                mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                for (auto n = begin; n < end; ++n)
                {
                    auto ilocal                     = data.GVGilocal(n);
                    auto e                          = data.GVGe(n);
                    auto lamee                      = data.lame.col(e);
                    auto wg                         = data.wg(e);
                    auto Te                         = data.mesh.E.col(e);
                    mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
                    mini::SMatrix<Scalar, 3, 4> xe =
                        FromEigen(data.x(Eigen::placeholders::all, Te).block<3, 4>(0, 0));
                    mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
                    physics::StableNeoHookeanEnergy<3> Psi{};
                    mini::SVector<Scalar, 9> gF;
                    mini::SMatrix<Scalar, 9, 9> HF;
                    Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                    using namespace pbat::sim::vbd::kernels;
                    AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                    AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
                }
                // Update vertex position
                Scalar m                         = data.m(i);
                mini::SVector<Scalar, 3> xti     = FromEigen(data.xt.col(i).head<3>());
                mini::SVector<Scalar, 3> xtildei = FromEigen(data.xtilde.col(i).head<3>());
                mini::SVector<Scalar, 3> xi      = FromEigen(data.x.col(i).head<3>());
                using namespace pbat::sim::vbd::kernels;
                AddDamping(dt, xti, xi, data.kD, gi, Hi);
                AddInertiaDerivatives(dt2, m, xtildei, xi, gi, Hi);
                IntegratePositions(gi, Hi, xi, data.detHZero);
                data.x.col(i) = ToEigen(xi);
            });
        }
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat