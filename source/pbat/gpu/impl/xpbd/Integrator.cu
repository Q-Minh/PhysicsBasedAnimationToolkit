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
      V(data.V.size()),
      F(data.F.cols()),
      T(data.T.cols()),
      B(data.BV.size()),
      Tbvh(static_cast<GpuIndex>(data.V.cols() + data.T.cols())),
      Fbvh(static_cast<GpuIndex>(data.F.cols())),
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
    common::ToBuffer(data.v, v);
    common::ToBuffer(data.V, V);
    common::ToBuffer(data.F, F);
    common::ToBuffer(data.T, T);
    common::ToBuffer(data.BV, B);
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
    // Detect collision candidates and setup collision constraint solve
    // TODO:
    // 1. Compute AABBs of line segments x_i^t -> \Tilde{x_i}
    // 2. Sort line segment AABBs via Morton encoding (thrust::sort_by_key on "morton codes" and
    // active mask buffer)
    // 3. Compute AABBs of linearly swept triangles x_{jkl} -> \Tilde{x_{jkl}}
    // 4. Build triangle BVH over swept triangle volumes
    // 5. Detect overlaps between line segments and swept triangles
    // 6. For each overlapping pair (i, jkl), mark i as active
    // 7. Compact (sorted) active vertices into an active vertex list (thrust::copy_if from "active
    // mask buffer" to "active vertices buffer" should be sufficient)
    // 8. Compute AABBs of (non-swept) triangles
    // 9. Build triangle BVH over triangles
    // 10. Find nearest triangles f to active vertices i to form contact pairs (i, f) and compute
    // the signed distances sd(i,f)

    // Determine active vertices/particles
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        xt = x;
        // Reset "Lagrange" multipliers
        for (auto d = 0; d < kConstraintTypes; ++d)
        {
            lagrange[d].SetConstant(GpuScalar(0));
        }
        // Initialize constraint solve
        thrust::device_event e = thrust::async::for_each(
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
        // Solve constraints
        for (auto k = 0; k < iterations; ++k)
        {
            // Elastic constraints
            bool const bHasClusterPartitions = not SGptr.empty();
            if (bHasClusterPartitions)
            {
                ProjectClusteredBlockNeoHookeanConstraints(e, sdt, sdt2);
            }
            else
            {
                ProjectBlockNeoHookeanConstraints(e, sdt, sdt2);
            }
            // Collision constraints
            /*auto const collisionConstraintId = static_cast<int>(EConstraint::Collision);
            e                                = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                [x      = x.Raw(),
                 xb     = xb.Raw(),
                 xt     = xt.Raw(),
                 lambda = lagrange[collisionConstraintId].Raw(),
                 alpha  = alpha[collisionConstraintId].Raw(),
                 beta   = beta[collisionConstraintId].Raw(),
                 V      = V.Raw(),
                 F      = F.Raw(),
                 minv   = minv.Raw(),
                 dt     = sdt,
                 dt2    = sdt2,
                 muC    = muC.Raw(),
                 muS    = muS,
                 muD    = muK] PBAT_DEVICE(GpuIndex c) {
                    using pbat::sim::xpbd::kernels::ProjectVertexTriangle;
                    using namespace pbat::math::linalg::mini;
                    auto sv = pairs[c].first;
                    auto v  = V[0][sv];
                    SVector<GpuIndex, 3> f{
                        F[0][pairs[c].second],
                        F[1][pairs[c].second],
                        F[2][pairs[c].second]};
                    GpuScalar minvv              = minv[v];
                    SVector<GpuScalar, 3> xvt    = FromBuffers<3, 1>(xt, v);
                    SVector<GpuScalar, 3> xv     = FromBuffers<3, 1>(x, v);
                    SMatrix<GpuScalar, 3, 3> xft = FromBuffers(xt, f.Transpose());
                    SMatrix<GpuScalar, 3, 3> xf  = FromBuffers(x, f.Transpose());
                    GpuScalar atildec            = alpha[c] / dt2;
                    GpuScalar gammac             = atildec * beta[c] * dt;
                    GpuScalar lambdac            = lambda[c];
                    GpuScalar muc                = muC[sv];
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
                });
            e = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nContacts),
                [x = x.Raw(), xb = xb.Raw(), V = V.Raw()] PBAT_DEVICE(GpuIndex c) {
                    using namespace pbat::math::linalg::mini;
                    auto v                   = CV[0][pairs[c].first];
                    SVector<GpuScalar, 3> xv = FromBuffers<3, 1>(xb, c);
                    ToBuffers(xv, x, v);
                });*/
        }
        // Update simulation state
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            [xt = xt.Raw(), x = x.Raw(), v = v.Raw(), dt = sdt] PBAT_DEVICE(GpuIndex i) {
                using pbat::sim::xpbd::kernels::IntegrateVelocity;
                using namespace pbat::math::linalg::mini;
                auto vi = IntegrateVelocity(FromBuffers<3, 1>(xt, i), FromBuffers<3, 1>(x, i), dt);
                ToBuffers(vi, v, i);
            });

        // TODO:
        // 1. Update nearest neighbours f of i with a warm-started (using sd(i,f) as a query upper
        // bound) nearest neighbour search over the triangle BVH.

        e.wait();
    }

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

void Integrator::ProjectBlockNeoHookeanConstraints(thrust::device_event& e, Scalar dt, Scalar dt2)
{
    auto const snhConstraintId = static_cast<int>(EConstraint::StableNeoHookean);
    auto const nPartitions     = static_cast<Index>(Pptr.size()) - 1;
    for (auto p = 0; p < nPartitions; ++p)
    {
        auto pbegin = Pptr[p];
        auto pend   = Pptr[p + 1];
        e           = thrust::async::for_each(
            thrust::device.after(e),
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

void Integrator::ProjectClusteredBlockNeoHookeanConstraints(
    thrust::device_event& e,
    Scalar dt,
    Scalar dt2)
{
    auto const snhConstraintId    = static_cast<int>(EConstraint::StableNeoHookean);
    auto const nClusterPartitions = static_cast<Index>(SGptr.size()) - 1;
    for (Index cp = 0; cp < nClusterPartitions; ++cp)
    {
        auto cpbegin = SGptr[cp];
        auto cpend   = SGptr[cp + 1];
        e            = thrust::async::for_each(
            thrust::device.after(e),
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

} // namespace xpbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>

#pragma nv_diag_suppress 177

TEST_CASE("[gpu][impl][xpbd] Integrator") {}