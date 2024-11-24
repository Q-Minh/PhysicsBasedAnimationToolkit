#include "Smoother.h"

#include "Kernels.h"
#include "Level.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Tetrahedron.h"
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
                    Index g        = L.VG.GVGg(kg);
                    Index e        = L.VG.GVGe(kg);
                    Index ilocal   = L.VG.GVGilocal(kg);
                    Scalar wg      = L.E.wg(g);
                    Scalar mug     = L.E.mug(g);
                    Scalar lambdag = L.E.lambdag(g);
                    Scalar dt2     = L.E.dt * L.E.dt;
                    physics::StableNeoHookeanEnergy<3> Psi{};
                    bool const bSingular = L.E.sg(g);
                    if (bSingular)
                    {
                        // NOTE:
                        // For singular quadrature point, we can't use the root level's energy,
                        // so just use the coarse cage's elastic energy.
                        auto inds = L.C.E.col(e);
                        SMatrix<Scalar, 3, 4> xcg =
                            FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                        SMatrix<Scalar, 4, 3> GNcg = FromEigen(L.E.GNcg.block<4, 3>(0, g * 3));
                        kernels::smoothing::AccumulateSingularEnergy(
                            dt2,
                            ilocal,
                            wg,
                            Psi,
                            mug,
                            lambdag,
                            xcg,
                            GNcg,
                            gi,
                            Hi);
                    }
                    else
                    {
                        Scalar rhog = L.E.rhog(g);
                        // Kinetic energy is 1/2 w_g rho_g || Nc*xc - Nf*xf ||_2^2 + w_g
                        // \Psi(Ncf*xc)
                        auto inds = L.C.E.col(e);
                        SMatrix<Scalar, 3, 4> xcg =
                            FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                        SVector<Scalar, 4> Nc      = FromEigen(L.E.Ncg.col(g).head<4>());
                        SVector<Scalar, 3> xtildeg = FromEigen(L.E.xtildeg.col(g).head<3>());
                        kernels::smoothing::AccumulateKineticEnergy(
                            ilocal,
                            wg,
                            rhog,
                            xcg,
                            Nc,
                            xtildeg,
                            gi,
                            Hi);
                        // Potential energy is w_g \Psi(Neg * xc)
                        IndexVector<4> erg = L.E.erg.col(g);
                        auto rinds         = L.C.E(Eigen::placeholders::all, erg).reshaped();
                        SMatrix<Scalar, 3, 16> xcr =
                            FromEigen(L.C.x(Eigen::placeholders::all, rinds).block<3, 16>(0, 0));
                        SMatrix<Scalar, 4, 4> Nce   = FromEigen(L.E.Nrg.block<4, 4>(0, 4 * g));
                        SMatrix<Scalar, 4, 3> GNer  = FromEigen(L.E.GNfg.block<4, 3>(0, 3 * g));
                        SVector<Index, 4> indicator = FromEigen(erg) == e;
                        kernels::smoothing::AccumulatePotentialEnergy(
                            dt2,
                            ilocal,
                            wg,
                            Psi,
                            mug,
                            lambdag,
                            xcr,
                            Nce,
                            GNer,
                            indicator,
                            gi,
                            Hi);
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