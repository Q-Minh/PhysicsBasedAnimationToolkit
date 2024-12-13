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
namespace lod {

void Smoother::Apply(Index iters, Scalar dt, Level& l) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.lod.Smoother.Apply");
    Scalar const dt2              = dt * dt;
    auto const [ptr, adj]         = std::tie(l.ptr, l.adj);
    CageQuadrature const& CQ      = l.Qcage;
    DirichletQuadrature const& DQ = l.Qdirichlet;
    MatrixX& x                    = l.x;
    for (auto iter = 0; iter < iters; ++iter)
    {
        auto nPartitions = ptr.size() - 1;
        for (Index p = 0; p < nPartitions; ++p)
        {
            auto pBegin = ptr(p);
            auto pEnd   = ptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::SMatrix;
                using mini::SVector;
                using mini::FromEigen;
                using mini::ToEigen;

                Index i                  = adj(kp);
                SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                // Kinetic + Potential energy
                MomentumEnergy const& Ekin = l.Ekinetic;
                ElasticEnergy const& Epot  = l.Epotential;
                auto gBegin                = CQ.GVGp(i);
                auto gEnd                  = CQ.GVGp(i + 1);
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g      = CQ.GVGg(kg);
                    Index ilocal = CQ.GVGilocal(kg);
                    Scalar wg    = CQ.wg(g);
                    Index e      = CQ.eg(g);
                    auto inds    = l.mesh.E.col(e);
                    SMatrix<Scalar, 3, 4> xcg =
                        FromEigen(x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                    SVector<Scalar, 4> Ncg = FromEigen(CQ.Ncg.col(g).head<4>());
                    bool const bSingular   = CQ.sg(g);
                    if (not bSingular)
                    {
                        using kernels::AccumulateShapeMatchingEnergy;
                        Scalar rhog                = Ekin.rhog(g);
                        SVector<Scalar, 3> xtildeg = FromEigen(Ekin.xtildeg.col(g).head<3>());
                        AccumulateShapeMatchingEnergy(ilocal, wg, rhog, xcg, Ncg, xtildeg, gi, Hi);
                    }
                    using kernels::AccumulateElasticEnergy;
                    Scalar mug     = Epot.mug(g);
                    Scalar lambdag = Epot.lambdag(g);
                    physics::StableNeoHookeanEnergy<3> Psi{};
                    SMatrix<Scalar, 4, 3> GNcg = FromEigen(CQ.GNcg.block<4, 3>(0, 3 * g));
                    AccumulateElasticEnergy(ilocal, dt2 * wg, Psi, mug, lambdag, xcg, GNcg, gi, Hi);
                }
                // Dirichlet energy
                gBegin                      = DQ.GVGp(i);
                gEnd                        = DQ.GVGp(i + 1);
                DirichletEnergy const& Edir = l.Edirichlet;
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g      = DQ.GVGg(kg);
                    Index ilocal = DQ.GVGilocal(kg);
                    Index e      = DQ.eg(g);
                    Scalar wg    = DQ.wg(g);
                    auto inds    = l.mesh.E.col(e);
                    SMatrix<Scalar, 3, 4> xcg =
                        FromEigen(x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                    Scalar muD             = Edir.muD;
                    SVector<Scalar, 4> Ncg = FromEigen(DQ.Ncg.col(g).head<4>());
                    SVector<Scalar, 3> dxg = FromEigen(Edir.dg.col(g).head<3>());
                    // Dirichlet energy is 1/2 w_g mu_D || N*x - dx ||_2^2
                    using kernels::AccumulateShapeMatchingEnergy;
                    AccumulateShapeMatchingEnergy(ilocal, wg, muD, xcg, Ncg, dxg, gi, Hi);
                }
                // Commit descent
                if (std::abs(Determinant(Hi)) < Scalar(1e-8))
                    return;
                SVector<Scalar, 3> dx = -(mini::Inverse(Hi) * gi);
                x.col(i) += ToEigen(dx);
            });
        }
    }
}

void Smoother::Apply(Index iters, Scalar dt, Data& data) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.lod.Smoother.Apply");
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

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat