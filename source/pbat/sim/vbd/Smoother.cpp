#include "Smoother.h"

#include "Data.h"
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

void Smoother::Apply(Scalar dt, Level& L)
{
    Scalar const dt2 = dt * dt;
    auto nPartitions = L.C.ptr.size() - 1;
    for (auto iter = 0; iter < iterations; ++iter)
    {
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto pBegin = L.C.ptr[p];
            auto pEnd   = L.C.ptr[p + 1];
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::SMatrix;
                using mini::SVector;
                using mini::FromEigen;
                using mini::ToEigen;

                Index i                  = L.C.adj[kp];
                SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                // Kinetic + Potential energy
                auto gBegin = L.E.GVGp[i];
                auto gEnd   = L.E.GVGp[i + 1];
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g        = L.E.GVGg(kg);
                    Index e        = L.E.GVGe(kg);
                    Index ilocal   = L.E.GVGilocal(kg);
                    Scalar wg      = L.E.wg(g);
                    Scalar mug     = L.E.mug(g);
                    Scalar lambdag = L.E.lambdag(g);
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
                        auto inds   = L.C.E.col(e);
                        SMatrix<Scalar, 3, 4> xcg =
                            FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                        SVector<Scalar, 4> Ncg     = FromEigen(L.E.Ncg.col(g).head<4>());
                        SVector<Scalar, 3> xtildeg = FromEigen(L.E.xtildeg.col(g).head<3>());
                        kernels::smoothing::AccumulateKineticEnergy(
                            ilocal,
                            wg,
                            rhog,
                            xcg,
                            Ncg,
                            xtildeg,
                            gi,
                            Hi);
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
                // Dirichlet energy
                gBegin = L.E.GVDGp[i];
                gEnd   = L.E.GVDGp[i + 1];
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g      = L.E.GVDGg(kg);
                    Index e      = L.E.GVDGe(kg);
                    Index ilocal = L.E.GVDGilocal(kg);
                    Scalar wg    = L.E.dwg(g);
                    auto inds    = L.C.E.col(e);
                    SMatrix<Scalar, 3, 4> xcg =
                        FromEigen(L.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                    SVector<Scalar, 4> Ncg = FromEigen(L.E.dNcg.col(g).head<4>());
                    SVector<Scalar, 3> dxg = FromEigen(L.E.dxg.col(g).head<3>());
                    // Dirichlet energy is 1/2 w_g || N*x - dx ||_2^2
                    auto xg = xcg * Ncg;
                    gi += wg * Ncg(ilocal) * (xg - dxg);
                    mini::Diag(Hi) += wg * Ncg(ilocal) * Ncg(ilocal);
                }
                // Commit descent
                SVector<Scalar, 3> dx = mini::Inverse(Hi) * gi;
                L.C.x.col(i) -= ToEigen(dx);
            });
        }
    }
}

void Smoother::Apply(Scalar dt, Scalar rho, Data& data)
{
    auto const nVertices                 = data.x.cols();
    Scalar const dt2                     = dt * dt;
    bool const bUseChebyshevAcceleration = rho > Scalar(0) and rho < Scalar(1);
    Scalar omega{};
    Scalar rho2 = rho * rho;
    // Minimize Backward Euler, i.e. BDF1, objective
    for (auto k = 0; k < iterations; ++k)
    {
        if (bUseChebyshevAcceleration)
            omega = kernels::ChebyshevOmega(k, rho2, omega);

        auto const nPartitions = data.Pptr.size() - 1;
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto const pBegin = data.Pptr(p);
            auto const pEnd   = data.Pptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index k) {
                using namespace math::linalg;
                using mini::FromEigen;
                using mini::ToEigen;

                auto i     = data.Padj[k];
                auto begin = data.GVGp(i);
                auto end   = data.GVGp(i + 1);
                // Elastic energy
                mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                for (auto n = begin; n < end; ++n)
                {
                    auto ilocal                     = data.GVGilocal[n];
                    auto e                          = data.GVGe[n];
                    auto lamee                      = data.lame.col(e);
                    auto wg                         = data.wg[e];
                    auto Te                         = data.T.col(e);
                    mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
                    mini::SMatrix<Scalar, 3, 4> xe =
                        FromEigen(data.x(Eigen::placeholders::all, Te).block<3, 4>(0, 0));
                    mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
                    physics::StableNeoHookeanEnergy<3> Psi{};
                    mini::SVector<Scalar, 9> gF;
                    mini::SMatrix<Scalar, 9, 9> HF;
                    Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                    kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                    kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
                }
                // Update vertex position
                Scalar m                         = data.m[i];
                mini::SVector<Scalar, 3> xti     = FromEigen(data.xt.col(i).head<3>());
                mini::SVector<Scalar, 3> xtildei = FromEigen(data.xtilde.col(i).head<3>());
                mini::SVector<Scalar, 3> xi      = FromEigen(data.x.col(i).head<3>());
                kernels::AddDamping(dt, xti, xi, data.kD, gi, Hi);
                kernels::AddInertiaDerivatives(dt2, m, xtildei, xi, gi, Hi);
                kernels::IntegratePositions(gi, Hi, xi, data.detHZero);
                data.x.col(i) = ToEigen(xi);
            });
        }

        if (bUseChebyshevAcceleration)
        {
            tbb::parallel_for(Index(0), nVertices, [&](Index i) {
                using namespace math::linalg;
                using mini::FromEigen;
                auto xkm2eig = data.xchebm2.col(i).head<3>();
                auto xkm1eig = data.xchebm1.col(i).head<3>();
                auto xkeig   = data.x.col(i).head<3>();
                auto xkm2    = FromEigen(xkm2eig);
                auto xkm1    = FromEigen(xkm1eig);
                auto xk      = FromEigen(xkeig);
                kernels::ChebyshevUpdate(k, omega, xkm2, xkm1, xk);
            });
        }
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat