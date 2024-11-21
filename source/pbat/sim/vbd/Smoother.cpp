#include "Smoother.h"

#include "Kernels.h"
#include "Level.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

void Smoother::Apply(Level& L)
{
    auto nPartitions = L.P.ptr.size() - 1;
    for (auto iter = 0; iter < iterations; ++iter)
    {
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto pBegin = L.P.ptr[p];
            auto pEnd   = L.P.ptr[p + 1];
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::SMatrix;
                using mini::SVector;
                using mini::FromEigen;
                using mini::ToEigen;

                Index i                  = L.P.adj[kp];
                SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                auto gBegin              = L.VG.GVGp[i];
                auto gEnd                = L.VG.GVGp[i + 1];
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g              = L.VG.GVGg(kg);
                    Index e              = L.VG.GVGe(kg);
                    Index ilocal         = L.VG.GVGilocal(kg);
                    Scalar wg            = L.E.wg(g);
                    Scalar mug           = L.E.mug(g);
                    Scalar lambdag       = L.E.lambdag(g);
                    bool const bSingular = L.E.sg(g);
                    if (bSingular)
                    {
                        auto inds = L.C.E.col(e);
                        SMatrix<Scalar, 3, 4> xeg =
                            FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                        SMatrix<Scalar, 4, 3> GNeg = FromEigen(L.E.GNeg.block<4, 3>(0, g * 3));
                        SMatrix<Scalar, 3, 3> Feg  = xeg * GNeg;
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        SVector<Scalar, 9> gF;
                        SMatrix<Scalar, 9, 9> HF;
                        Psi.gradAndHessian(Feg, mug, lambdag, gF, HF);
                        Scalar dt2 = L.E.dt * L.E.dt;
                        kernels::AccumulateElasticGradient(ilocal, dt2 * wg, GNeg, gF, gi);
                        kernels::AccumulateElasticHessian(ilocal, dt2 * wg, GNeg, HF, Hi);
                    }
                    else
                    {
                        Scalar rhog = L.E.rhog(g);
                        // Energy is 1/2 w_g rho_g || Nc*xc - Nf*xf ||_2^2
                        auto inds = L.C.E.col(e);
                        SMatrix<Scalar, 3, 4> xeg =
                            FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                        SVector<Scalar, 4> Nc      = FromEigen(L.E.Ng.col(g).head<4>());
                        SVector<Scalar, 3> xtildeg = FromEigen(L.E.xtildeg.col(g).head<3>());
                        auto xcg                   = xeg * Nc;
                        gi += wg * rhog * (xcg - xtildeg) * Nc(ilocal);
                        mini::Diag(Hi) += wg * rhog * Nc(ilocal) * Nc(ilocal);
                    }
                }
                SVector<Scalar, 3> dx = mini::Inverse(Hi) * gi;
                L.C.x.col(i) -= ToEigen(dx);
            });
        }
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat