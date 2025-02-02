// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/xpbd/Kernels.h"

#include <thrust/async/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace impl {
namespace xpbd {

Integrator::Integrator(Data const& data)
    : x(data.x.cols()),
      T(data.T.cols()),
      cd(data.BV.cast<GpuIndex>(), data.V.cast<GpuIndex>(), data.F.cast<GpuIndex>()),
      xt(data.x.cols()),
      xb(data.x.cols()),
      v(data.v.cols()),
      aext(data.aext.cols()),
      minv(data.minv.size()),
      lame(data.lame.size()),
      DmInv(data.DmInv.size()),
      gamma(data.gammaSNH.size()),
      lagrange(),
      alpha(),
      beta(),
      Pptr(data.Pptr),
      Padj(data.Padj.size()),
      SGptr(data.SGptr),
      SGadj(data.SGadj.size()),
      Cptr(data.Cptr.size()),
      Cadj(data.Cadj.size()),
      muC(data.muV.size()),
      muS{static_cast<GpuScalar>(data.muS)},
      muK{static_cast<GpuScalar>(data.muD)}
{
    // Initialize particle data
    common::ToBuffer(data.x, x);
    xt = x;
    common::ToBuffer(data.v, v);
    common::ToBuffer(data.T, T);
    common::ToBuffer(data.aext, aext);
    common::ToBuffer(data.minv, minv);
    common::ToBuffer(data.lame, lame);
    common::ToBuffer(data.DmInv, DmInv);
    common::ToBuffer(data.gammaSNH, gamma);
    // Setup constraints
    int const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
    lagrange[snhConstraintId].Resize(data.lambda[snhConstraintId].size());
    alpha[snhConstraintId].Resize(data.alpha[snhConstraintId].size());
    beta[snhConstraintId].Resize(data.beta[snhConstraintId].size());
    common::ToBuffer(data.lambda[snhConstraintId], lagrange[snhConstraintId]);
    common::ToBuffer(data.alpha[snhConstraintId], alpha[snhConstraintId]);
    common::ToBuffer(data.beta[snhConstraintId], beta[snhConstraintId]);

    int const collisionConstraintId = static_cast<int>(EConstraint::Collision);
    lagrange[collisionConstraintId].Resize(data.lambda[collisionConstraintId].size());
    alpha[collisionConstraintId].Resize(data.alpha[collisionConstraintId].size());
    beta[collisionConstraintId].Resize(data.beta[collisionConstraintId].size());
    common::ToBuffer(data.lambda[collisionConstraintId], lagrange[collisionConstraintId]);
    common::ToBuffer(data.alpha[collisionConstraintId], alpha[collisionConstraintId]);
    common::ToBuffer(data.beta[collisionConstraintId], beta[collisionConstraintId]);
    // Setup partitions
    common::ToBuffer(pbat::common::ToEigen(data.Padj).cast<GpuIndex>().eval(), Padj);
    bool const bHasClusteredPartitions = not SGptr.empty();
    if (bHasClusteredPartitions)
    {
        common::ToBuffer(pbat::common::ToEigen(data.SGadj).cast<GpuIndex>().eval(), SGadj);
        common::ToBuffer(pbat::common::ToEigen(data.Cptr).cast<GpuIndex>().eval(), Cptr);
        common::ToBuffer(pbat::common::ToEigen(data.Cadj).cast<GpuIndex>().eval(), Cadj);
    }
    // Copy collision data
    common::ToBuffer(data.muV, muC);
}

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.xpbd.Integrator.Step");

    GpuScalar const sdt       = dt / static_cast<GpuScalar>(substeps);
    GpuScalar const sdt2      = sdt * sdt;
    GpuIndex const nParticles = static_cast<GpuIndex>(x.Size());

    // Determine active contact vertices for the whole step using predicted positions
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nParticles),
        [dt,
         dt2  = dt * dt,
         xt   = xt.Raw(),
         x    = x.Raw(),
         vt   = v.Raw(),
         aext = aext.Raw()] PBAT_DEVICE(GpuIndex i) {
            using pbat::sim::xpbd::kernels::InitialPosition;
            using namespace pbat::math::linalg::mini;
            auto xi = InitialPosition(
                FromBuffers<3, 1>(xt, i),
                FromBuffers<3, 1>(vt, i),
                FromBuffers<3, 1>(aext, i),
                dt,
                dt2);
            ToBuffers(xi, x, i);
        });
    using pbat::math::linalg::mini::FromEigen;
    cd.InitializeActiveSet(xt, x, FromEigen(Smin), FromEigen(Smax));

    // Use substepping for accelerated convergence
    for (auto s = 0; s < substeps; ++s)
    {
        PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
            subStepCtx,
            "pbat.gpu.impl.xpbd.Integrator.Step.SubStep");

        // Reset "Lagrange" multipliers
        for (auto d = 0; d < kConstraintTypes; ++d)
        {
            lagrange[d].SetConstant(GpuScalar(0));
        }
        // Initialize constraint solve
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            [xt   = xt.Raw(),
             x    = x.Raw(),
             vt   = v.Raw(),
             aext = aext.Raw(),
             dt   = sdt,
             dt2  = sdt2] PBAT_DEVICE(GpuIndex i) {
                using pbat::sim::xpbd::kernels::InitialPosition;
                using namespace pbat::math::linalg::mini;
                auto xi = InitialPosition(
                    FromBuffers<3, 1>(xt, i),
                    FromBuffers<3, 1>(vt, i),
                    FromBuffers<3, 1>(aext, i),
                    dt,
                    dt2);
                ToBuffers(xi, x, i);
            });
        // Update active set
        cd.UpdateActiveSet(x);
        // Solve constraints
        PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
            constraintSolveCtx,
            "pbat.gpu.impl.xpbd.Integrator.Step.ConstraintSolve");
        for (auto k = 0; k < iterations; ++k)
        {
            // Elastic constraints
            bool const bHasClusterPartitions = not SGptr.empty();
            if (bHasClusterPartitions)
            {
                ProjectClusteredBlockNeoHookeanConstraints(sdt, sdt2);
            }
            else
            {
                ProjectBlockNeoHookeanConstraints(sdt, sdt2);
            }
            ProjectCollisionConstraints(sdt, sdt2);
        }
        PBAT_PROFILE_CUDA_HOST_SCOPE_END(constraintSolveCtx);
        // Update simulation state
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            [xt = xt.Raw(), x = x.Raw(), v = v.Raw(), dt = sdt] PBAT_DEVICE(GpuIndex i) {
                using pbat::sim::xpbd::kernels::IntegrateVelocity;
                using namespace pbat::math::linalg::mini;
                auto vi = IntegrateVelocity(FromBuffers<3, 1>(xt, i), FromBuffers<3, 1>(x, i), dt);
                ToBuffers(vi, v, i);
                ToBuffers(FromBuffers<3, 1>(x, i), xt, i);
            });

        PBAT_PROFILE_CUDA_HOST_SCOPE_END(subStepCtx);
    }
    cd.FinalizeActiveSet(x);

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::SetCompliance(Eigen::Ref<GpuMatrixX const> const& alphaIn, EConstraint eConstraint)
{
    common::ToBuffer(alphaIn, alpha[static_cast<int>(eConstraint)]);
}

void Integrator::SetFrictionCoefficients(GpuScalar muSin, GpuScalar muKin)
{
    this->muS = muSin;
    this->muK = muKin;
}

void Integrator::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    Smin = min;
    Smax = max;
}

common::Buffer<GpuScalar> const& Integrator::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    return lagrange[static_cast<int>(eConstraint)];
}

common::Buffer<GpuScalar> const& Integrator::GetCompliance(EConstraint eConstraint) const
{
    return alpha[static_cast<int>(eConstraint)];
}

PBAT_DEVICE static void ProjectBlockNeoHookeanConstraint(
    std::array<GpuScalar*, 3> x,
    std::array<GpuScalar*, 3> xt,
    GpuScalar* lambda,
    std::array<GpuIndex*, 4> T,
    GpuScalar* minv,
    GpuScalar* alpha,
    GpuScalar* beta,
    GpuScalar* DmInv,
    GpuScalar* gammaSNH,
    GpuScalar dt,
    GpuScalar dt2,
    GpuIndex c)
{
    using pbat::sim::xpbd::kernels::ProjectBlockNeoHookean;
    using namespace pbat::math::linalg::mini;
    SVector<GpuIndex, 4> Tc       = FromBuffers<4, 1>(T, c);
    SVector<GpuScalar, 4> minvc   = FromFlatBuffer(minv, Tc);
    SVector<GpuScalar, 2> atildec = FromFlatBuffer<2, 1>(alpha, c) / dt2;
    SVector<GpuScalar, 2> betac   = FromFlatBuffer<2, 1>(beta, c);
    SVector<GpuScalar, 2> gammac{atildec(0) * betac(0) * dt, atildec(1) * betac(1) * dt};
    SMatrix<GpuScalar, 3, 3> DmInvc = FromFlatBuffer<3, 3>(DmInv, c);
    SVector<GpuScalar, 2> lambdac   = FromFlatBuffer<2, 1>(lambda, c);
    SMatrix<GpuScalar, 3, 4> xtc    = FromBuffers(xt, Tc.Transpose());
    SMatrix<GpuScalar, 3, 4> xc     = FromBuffers(x, Tc.Transpose());
    ProjectBlockNeoHookean(minvc, DmInvc, gammaSNH[c], atildec, gammac, xtc, lambdac, xc);
    ToFlatBuffer(lambdac, lambda, c);
    ToBuffers(xc, Tc.Transpose(), x);
}

void Integrator::ProjectBlockNeoHookeanConstraints(GpuScalar dt, GpuScalar dt2)
{
    auto const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
    auto const nPartitions     = static_cast<Index>(Pptr.size()) - 1;
    for (auto p = 0; p < nPartitions; ++p)
    {
        auto pbegin = Pptr[p];
        auto pend   = Pptr[p + 1];
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(pbegin),
            thrust::make_counting_iterator(pend),
            [partition = Padj.Raw(),
             x         = x.Raw(),
             xt        = xt.Raw(),
             lambda    = lagrange[snhConstraintId].Raw(),
             T         = T.Raw(),
             minv      = minv.Raw(),
             alpha     = alpha[snhConstraintId].Raw(),
             beta      = beta[snhConstraintId].Raw(),
             DmInv     = DmInv.Raw(),
             gammaSNH  = gamma.Raw(),
             dt,
             dt2] PBAT_DEVICE(Index k) {
                GpuIndex c = partition[k];
                ProjectBlockNeoHookeanConstraint(
                    x,
                    xt,
                    lambda,
                    T,
                    minv,
                    alpha,
                    beta,
                    DmInv,
                    gammaSNH,
                    dt,
                    dt2,
                    c);
            });
    }
}

void Integrator::ProjectClusteredBlockNeoHookeanConstraints(GpuScalar dt, GpuScalar dt2)
{
    auto const snhConstraintId    = static_cast<int>(EConstraint::StableNeoHookean);
    auto const nClusterPartitions = static_cast<Index>(SGptr.size()) - 1;
    for (Index cp = 0; cp < nClusterPartitions; ++cp)
    {
        auto cpbegin = SGptr[cp];
        auto cpend   = SGptr[cp + 1];
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(cpbegin),
            thrust::make_counting_iterator(cpend),
            [SGadj    = SGadj.Raw(),
             Cptr     = Cptr.Raw(),
             Cadj     = Cadj.Raw(),
             x        = x.Raw(),
             xt       = xt.Raw(),
             lambda   = lagrange[snhConstraintId].Raw(),
             T        = T.Raw(),
             minv     = minv.Raw(),
             alpha    = alpha[snhConstraintId].Raw(),
             beta     = beta[snhConstraintId].Raw(),
             DmInv    = DmInv.Raw(),
             gammaSNH = gamma.Raw(),
             dt,
             dt2] PBAT_DEVICE(Index ks) {
                auto kc     = SGadj[ks];
                auto cbegin = Cptr[kc];
                auto cend   = Cptr[kc + 1];
                for (auto k = cbegin; k < cend; ++k)
                {
                    GpuIndex c = Cadj[k];
                    ProjectBlockNeoHookeanConstraint(
                        x,
                        xt,
                        lambda,
                        T,
                        minv,
                        alpha,
                        beta,
                        DmInv,
                        gammaSNH,
                        dt,
                        dt2,
                        c);
                }
            });
    }
}

void Integrator::ProjectCollisionConstraints(GpuScalar dt, GpuScalar dt2)
{
    auto const collisionConstraintId = static_cast<int>(EConstraint::Collision);
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(cd.nActive),
        [dt,
         dt2,
         x      = x.Raw(),
         xb     = xb.Raw(),
         xt     = xt.Raw(),
         lambda = lagrange[collisionConstraintId].Raw(),
         alpha  = alpha[collisionConstraintId].Raw(),
         beta   = beta[collisionConstraintId].Raw(),
         V      = cd.V.Raw(),
         F      = cd.F.Raw(),
         av     = cd.av.Raw(),
         nn     = cd.nn.Raw(),
         minv   = minv.Raw(),
         muC    = muC.Raw(),
         muS    = muS,
         muD    = muK] PBAT_DEVICE(GpuIndex c) {
            using pbat::sim::xpbd::kernels::ProjectVertexTriangle;
            using namespace pbat::math::linalg::mini;
            GpuIndex const v              = av[c];
            GpuIndex const i              = V[v];
            auto constexpr kMaxNeighbours = decltype(cd)::kMaxNeighbours;
            auto kb                       = v * kMaxNeighbours;
            for (auto k = 0; k < kMaxNeighbours; ++k)
            {
                auto f = nn[kb + k];
                if (f < 0)
                    break;
                auto fv                      = FromBuffers<3, 1>(F, f);
                SMatrix<GpuScalar, 3, 3> xft = FromBuffers(xt, fv.Transpose());
                SMatrix<GpuScalar, 3, 3> xf  = FromBuffers(x, fv.Transpose());
                SVector<GpuScalar, 3> xvt    = FromBuffers<3, 1>(xt, i);
                SVector<GpuScalar, 3> xv     = FromBuffers<3, 1>(x, i);
                GpuScalar minvv              = minv[i];
                GpuScalar atildec            = alpha[c] / dt2;
                GpuScalar gammac             = atildec * beta[c] * dt;
                GpuScalar lambdac            = lambda[c];
                GpuScalar muc                = muC[v];
                bool const bProject          = ProjectVertexTriangle(
                    minvv,
                    xvt,
                    xft,
                    xf,
                    muc,
                    muS,
                    muD,
                    atildec,
                    gammac,
                    lambdac,
                    xv);
                if (bProject)
                {
                    lambda[c] = lambdac;
                }
                ToBuffers(xv, xb, c);
            }
        });
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(cd.nActive),
        [x = x.Raw(), xb = xb.Raw(), V = cd.V.Raw(), av = cd.av.Raw()] PBAT_DEVICE(GpuIndex c) {
            using namespace pbat::math::linalg::mini;
            GpuIndex const i         = V[av[c]];
            SVector<GpuScalar, 3> xv = FromBuffers<3, 1>(xb, c);
            ToBuffers(xv, x, i);
        });
}

} // namespace xpbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>

#pragma nv_diag_suppress 177

TEST_CASE("[gpu][impl][xpbd] Integrator") {}