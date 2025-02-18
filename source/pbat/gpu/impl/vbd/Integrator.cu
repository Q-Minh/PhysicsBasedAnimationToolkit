// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/impl/common/Cuda.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"

#include <cuda/api.hpp>
#include <cuda/functional>
// #include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace impl {
namespace vbd {

Integrator::Integrator(Data const& data)
    : x(data.x.cols()),
      T(data.E.cols()),
      mWorldMin(),
      mWorldMax(),
      cd(data.B.cast<GpuIndex>(), data.V.cast<GpuIndex>(), data.F.cast<GpuIndex>()),
      fc(data.x.cols() * kernels::BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex),
      XVA(data.XVA.size()),
      FA(data.FA.size()),
      mActiveSetUpdateFrequency(static_cast<GpuIndex>(data.mActiveSetUpdateFrequency)),
      mPositionsAtT(data.xt.cols()),
      mInertialTargetPositions(data.xtilde.cols()),
      xkm2(data.xchebm2.cols()),
      xkm1(data.xchebm1.cols()),
      Uetr(data.E.cols()),
      ftr(data.X.cols()),
      xb(data.x.cols()),
      mVelocitiesAtT(data.vt.cols()),
      mVelocities(data.v.cols()),
      mExternalAcceleration(data.aext.cols()),
      mMass(data.x.cols()),
      mQuadratureWeights(data.wg.size()),
      mShapeFunctionGradients(data.GP.size()),
      mLameCoefficients(data.lame.size()),
      mDetHZero(static_cast<GpuScalar>(data.detHZero)),
      mVertexTetrahedronPrefix(data.GVGp.size()),
      mVertexTetrahedronNeighbours(data.GVGe.size()),
      mVertexTetrahedronLocalVertexIndices(data.GVGilocal.size()),
      mRayleighDamping(static_cast<GpuScalar>(data.kD)),
      mCollisionPenalty(static_cast<GpuScalar>(data.muC)),
      mFrictionCoefficient(static_cast<GpuScalar>(data.muF)),
      mSmoothFrictionRelativeVelocityThreshold(static_cast<GpuScalar>(data.epsv)),
      mPptr(data.Pptr.cast<GpuIndex>()),
      mPadj(data.Padj.size()),
      mInitializationStrategy(data.strategy),
      mGpuThreadBlockSize(64),
      mStream(common::Device(common::EDeviceSelectionPreference::HighestComputeCapability)
                  .create_stream(/*synchronize_with_default_stream=*/false))
{
    common::ToBuffer(data.x, x);
    mPositionsAtT = x;
    common::ToBuffer(data.E, T);

    fc.SetConstant(GpuIndex(-1));
    common::ToBuffer(data.XVA, XVA);
    common::ToBuffer(data.FA, FA);

    common::ToBuffer(data.v, mVelocities);
    common::ToBuffer(data.aext, mExternalAcceleration);
    common::ToBuffer(data.m, mMass);

    common::ToBuffer(data.wg, mQuadratureWeights);
    common::ToBuffer(data.GP, mShapeFunctionGradients);
    common::ToBuffer(data.lame, mLameCoefficients);

    common::ToBuffer(data.GVGp, mVertexTetrahedronPrefix);
    mVertexTetrahedronNeighbours.Resize(data.GVGe.size());
    mVertexTetrahedronLocalVertexIndices.Resize(data.GVGilocal.size());
    common::ToBuffer(data.GVGe, mVertexTetrahedronNeighbours);
    common::ToBuffer(data.GVGilocal, mVertexTetrahedronLocalVertexIndices);

    common::ToBuffer(data.Padj.cast<GpuIndex>().eval(), mPadj);
}

void Integrator::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.vbd.Integrator.Step");

    GpuScalar sdt  = dt / static_cast<GpuScalar>(substeps);
    GpuScalar sdt2 = sdt * sdt;

    InitializeActiveSet(dt);
    auto bdf = BdfDeviceParameters(sdt, sdt2);
    for (auto s = 0; s < substeps; ++s)
    {
        PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(subCtx, "pbat.gpu.impl.vbd.Integrator.SubStep");
        ComputeInertialTargets(sdt, sdt2);
        InitializeBcdSolution(sdt, sdt2);
        if (s % mActiveSetUpdateFrequency == 0)
            UpdateActiveSet();
        Solve(bdf, iterations, rho);
        UpdateBdfState(sdt);
        PBAT_PROFILE_CUDA_HOST_SCOPE_END(subCtx);
    }
    cd.FinalizeActiveSet(x);

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::SetSceneBoundingBox(
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    mWorldMin = min;
    mWorldMax = max;
}

void Integrator::SetBlockSize(GpuIndex blockSize)
{
    mGpuThreadBlockSize = std::clamp(blockSize, GpuIndex{32}, GpuIndex{256});
}

void Integrator::InitializeActiveSet(GpuScalar dt)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.InitializeActiveSet");

    GpuIndex const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(nVertices),
        [dt,
         dt2      = dt * dt,
         xt       = mPositionsAtT.Raw(),
         vt       = mVelocities.Raw(),
         aext     = mExternalAcceleration.Raw(),
         x        = x.Raw(),
         strategy = mInitializationStrategy] PBAT_DEVICE(auto i) {
            using namespace pbat::math::linalg::mini;
            auto xti   = FromBuffers<3, 1>(xt, i);
            auto vti   = FromBuffers<3, 1>(vt, i);
            auto aexti = FromBuffers<3, 1>(aext, i);
            auto xi    = xti + dt * vti + dt2 * aexti;
            ToBuffers(xi, x, i);
        });
    using pbat::math::linalg::mini::FromEigen;
    cd.InitializeActiveSet(mPositionsAtT, x, FromEigen(mWorldMin), FromEigen(mWorldMax));

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::ComputeInertialTargets(GpuScalar sdt, GpuScalar sdt2)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.ComputeInertialTargets");

    GpuIndex const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(nVertices),
        [xt     = mPositionsAtT.Raw(),
         vt     = mVelocities.Raw(),
         aext   = mExternalAcceleration.Raw(),
         xtilde = mInertialTargetPositions.Raw(),
         dt     = sdt,
         dt2    = sdt2] PBAT_DEVICE(auto i) {
            using pbat::sim::vbd::kernels::InertialTarget;
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto y = InertialTarget(
                FromBuffers<3, 1>(xt, i),
                FromBuffers<3, 1>(vt, i),
                FromBuffers<3, 1>(aext, i),
                dt,
                dt2);
            ToBuffers(y, xtilde, i);
        });

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::InitializeBcdSolution(GpuScalar sdt, GpuScalar sdt2)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.InitializeBcdSolution");

    GpuIndex const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(nVertices),
        [xt       = mPositionsAtT.Raw(),
         vtm1     = mVelocitiesAtT.Raw(),
         vt       = mVelocities.Raw(),
         aext     = mExternalAcceleration.Raw(),
         x        = x.Raw(),
         dt       = sdt,
         dt2      = sdt2,
         strategy = mInitializationStrategy] PBAT_DEVICE(auto i) {
            using pbat::sim::vbd::kernels::InitialPositionsForSolve;
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto x0 = InitialPositionsForSolve(
                FromBuffers<3, 1>(xt, i),
                FromBuffers<3, 1>(vtm1, i),
                FromBuffers<3, 1>(vt, i),
                FromBuffers<3, 1>(aext, i),
                dt,
                dt2,
                strategy);
            ToBuffers(x0, x, i);
        });

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::UpdateActiveSet()
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.vbd.Integrator.UpdateActiveSet");

    cd.UpdateActiveSet(x);
    static auto constexpr kMaxContacts =
        kernels::BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
    static auto constexpr kMaxNeighbours = contact::VertexTriangleMixedCcdDcd::kMaxNeighbours;
    thrust::for_each(
        thrust::device,
        cd.av.Data(),
        cd.av.Data() + cd.nActive,
        [V = cd.V.Raw(), nn = cd.nn.Raw(), fc = fc.Raw()] PBAT_DEVICE(GpuIndex v) {
            using namespace pbat::math::linalg::mini;
            GpuIndex i                            = V[v];
            SVector<GpuIndex, kMaxNeighbours> nnv = FromFlatBuffer<kMaxNeighbours, 1>(nn, v);
            SVector<GpuIndex, kMaxContacts> f     = -Ones<GpuIndex, kMaxContacts, 1>();
            auto const top                        = min(kMaxContacts, kMaxNeighbours);
            for (auto c = 0; c < top; ++c)
                if (nnv(c) >= 0)
                    f(c) = nnv(c);
            ToFlatBuffer(f, fc, i);
        });

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations, GpuScalar rho)
{
    bool bUseChebyshevAcceleration = rho > GpuScalar{0} and rho < GpuScalar{1};
    bool bUseTrustRegion           = false;
    bool bRunVanillaVbd            = not bUseChebyshevAcceleration and not bUseTrustRegion;
    if (bRunVanillaVbd)
        SolveWithVanillaVbd(bdf, iterations);
    if (bUseChebyshevAcceleration)
        SolveWithChebyshevVbd(bdf, iterations, rho);
}

void Integrator::SolveWithVanillaVbd(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.SolveWithVanillaVbd");
    for (auto k = 0; k < iterations; ++k)
    {
        RunVbdIteration(bdf);
    }
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::SolveWithChebyshevVbd(
    kernels::BackwardEulerMinimization& bdf,
    GpuIndex iterations,
    GpuScalar rho)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.SolveWithChebyshevVbd");
    GpuScalar rho2 = rho * rho;
    GpuScalar omega{};
    for (auto k = 0; k < iterations; ++k)
    {
        using pbat::sim::vbd::kernels::ChebyshevOmega;
        omega = ChebyshevOmega(k, rho2, omega);
        RunVbdIteration(bdf);
        UpdateChebyshevIterates(k, omega);
    }
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::SolveWithLinearTrustRegionVbd(
    kernels::BackwardEulerMinimization& bdf,
    GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.SolveWithLinearTrustRegionVbd");

    static auto constexpr kNumPastIterates = decltype(ftr)::kDims;
    auto const fObjective                  = [this, &bdf]() {
        ComputeVertexEnergies(bdf);
        return thrust::reduce(
            thrust::device,
            ftr[kNumPastIterates - 1].begin(),
            ftr[kNumPastIterates - 1].end(),
            GpuScalar{0});
    };
    Eigen::Vector<GpuScalar, kNumPastIterates> fk{};
    auto const fUpdateTrustRegionIterates = [this, &fk]() {
        UpdateTrustRegionIterates();
        pbat::common::ForRange<0, kNumPastIterates - 1>([&]<auto k>() { fk[k] = fk[k + 1]; });
    };

    // Quadratic energy proxy
    Eigen::Matrix<GpuScalar, 3, 3> Qinv{};
    // Q can be derived by simply considering the quadratic function
    // q(t)=at^2 + bt + c, where q(-1)=f^{k-2}, q(0)=f^{k-1}, q(1)=f^k
    // Then, Qinv=Q^{-1}
    // clang-format off
    Qinv <<  GpuScalar(0.5), GpuScalar(-1), GpuScalar(0.5),
            GpuScalar(-0.5),  GpuScalar(0), GpuScalar(0.5),
              GpuScalar(0.),  GpuScalar(1), GpuScalar(0);
    // clang-format on
    Eigen::Vector<GpuScalar, 3> abc{};
    auto const fProxy = [&abc](GpuScalar t) {
        return abc(0) * t * t + abc(1) * t + abc(2);
    };

    // Compute step size
    auto const fSquaredStepSize = [this]() {
        auto const nVertices = static_cast<GpuIndex>(x.Size());
        return thrust::transform_reduce(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nVertices),
            cuda::proclaim_return_type<GpuScalar>(
                [x = x.Raw(), xkm1 = xkm1.Raw()] PBAT_DEVICE(GpuIndex i) {
                    using namespace pbat::math::linalg::mini;
                    auto xkp1 = FromBuffers<3, 1>(x, i);
                    auto xk   = FromBuffers<3, 1>(xkm1, i);
                    return SquaredNorm(xkp1 - xk);
                }),
            GpuScalar{0},
            thrust::plus<GpuScalar>());
    };

    // Compute TR step
    auto const fTrStep = [this](GpuScalar t) {
        auto const nVertices = static_cast<GpuIndex>(x.Size());
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nVertices),
            [x = x.Raw(), xkm1 = xkm1.Raw(), t] PBAT_DEVICE(GpuIndex i) {
                using namespace pbat::math::linalg::mini;
                auto xkp1 = FromBuffers<3, 1>(x, i);
                auto xk   = FromBuffers<3, 1>(xkm1, i);
                auto xtr  = xk + t * (xkp1 - xk);
                ToBuffers(xtr, x, i);
            });
    };

    // Rollback TR step
    auto const fRollbackTrStep = [this](GpuScalar t) {
        auto const nVertices = static_cast<GpuIndex>(x.Size());
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nVertices),
            [x = x.Raw(), xkm1 = xkm1.Raw(), t] PBAT_DEVICE(GpuIndex i) {
                using namespace pbat::math::linalg::mini;
                // x^{k+1}-x^k = t \Delta x
                auto xkp1 = FromBuffers<3, 1>(x, i);
                auto xk   = FromBuffers<3, 1>(xkm1, i);
                auto dx   = (xkp1 - xk) / t;
                ToBuffers(xk + dx, x, i);
            });
    };

    // TR parameters
    GpuScalar constexpr eta{0.2f};
    GpuScalar constexpr tau{2};
    GpuScalar R2{0};
    GpuScalar constexpr zero{1e-6f};

    // TR VBD
    fk(kNumPastIterates - 1) = fObjective();
    for (auto k = 0; k < iterations; ++k)
    {
        if (k >= kNumPastIterates - 1)
        {
            // 1. Compute af,bf,cf s.t. fproxy(t) = af*t^2 + bf*t + cf
            // and fproxy(-1)=f^{k-2}, fproxy(0)=f^{k-1}, fproxy(1)=f^k
            abc = Qinv * fk;
            // 2. Throw away x^{k-2},
            // Now fk[kNumPastIterates-3] = f^{k-1}, fk[kNumPastIterates-2] = f^k
            // xkm2 = x^{k-1}, xkm1 = x^k
            fUpdateTrustRegionIterates();
            // 3. Compute VBD step, after which x = x^k + \Delta x
            RunVbdIteration(bdf);
            GpuScalar dx2 = fSquaredStepSize(); // |\Delta x|_2^2
            if (R2 < dx2 + zero)
            {
                // Initialize radius to tau times the VBD step size
                // R=tau*|\Delta x|_2
                // => R^2=tau^2*|\Delta x|_2^2
                R2 = tau * tau * dx2;
            }
            // 4. Compute TR accelerated step
            GpuScalar constexpr lower{1};
            GpuScalar const upper = std::sqrt(R2 / dx2);
            GpuScalar t           = -abc(1) / (GpuScalar{2} * abc(0)); // Minimum of fproxy
            t                     = std::clamp(t, lower, upper);       // Enforce TR constraint
            fTrStep(t); // Now x = x^{k-1} + t * \Delta x
            // 5. Compute energy at TR step
            fk(kNumPastIterates - 1) =
                fObjective(); // Now fk[kNumPastIterates-1] = f(x^{k-1} + t * \Delta x)
            GpuScalar fNextActual = fk(kNumPastIterates - 1);
            GpuScalar fCurrentActual =
                fk(kNumPastIterates - 2); // Recall f[kNumPastIterates-2] = f^k
            GpuScalar fNextProxy = fProxy(t);
            GpuScalar fCurrentProxy =
                fk(kNumPastIterates -
                   3); // Recall fproxy(0) = f^{k-1}, and fk[kNumPastIterates-3] = f^{k-1}
            GpuScalar const rho = (fCurrentActual - fNextActual) / (fCurrentProxy - fNextProxy);
            // 6. Accept or reject TR step, simultaneously updating TR radius
            bool const bStepAccepted = rho > eta;
            if (bStepAccepted)
            {
                bool const bIsStepAtBound = std::abs(dx2 - R2) < zero;
                if (bIsStepAtBound)
                {
                    // R' = tau*R
                    // -> R'^2 = tau^2*R^2
                    R2 *= tau * tau;
                }
            }
            else
            {
                // R' = R/tau
                // -> R'^2 = R^2/tau^2
                R2 /= tau * tau;
                fRollbackTrStep(t);
                fk(kNumPastIterates - 1) = fObjective();
            }
        }
        else
        {
            fUpdateTrustRegionIterates();
            RunVbdIteration(bdf);
            fk(kNumPastIterates - 1) = fObjective();
        }
    }
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::RunVbdIteration(kernels::BackwardEulerMinimization& bdf)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.vbd.Integrator.RunVbdIteration");
    auto const nPartitions = mPptr.size() - 1;
    for (auto p = 0; p < nPartitions; ++p)
    {
        auto pBegin                     = mPptr[p];
        auto pEnd                       = mPptr[p + 1];
        bdf.partition                   = mPadj.Raw() + pBegin;
        auto const nVerticesInPartition = static_cast<cuda::grid::dimension_t>(pEnd - pBegin);
        kernels::Invoke<kernels::VbdIterationTraits>(
            nVerticesInPartition,
            mGpuThreadBlockSize,
            bdf);
        // Copy xb back to x
        thrust::for_each(
            thrust::device,
            bdf.partition,
            bdf.partition + nVerticesInPartition,
            [xb = xb.Raw(), x = x.Raw()] PBAT_DEVICE(GpuIndex i) {
                using namespace pbat::math::linalg::mini;
                ToBuffers(FromBuffers<3, 1>(xb, i), x, i);
            });
    }
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::UpdateBdfState(GpuScalar sdt)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(ctx, "pbat.gpu.impl.vbd.Integrator.UpdateBdfState");
    GpuIndex const nVertices = static_cast<GpuIndex>(x.Size());
    mVelocitiesAtT           = mVelocities;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(nVertices),
        [xt = mPositionsAtT.Raw(), x = x.Raw(), v = mVelocities.Raw(), dt = sdt] PBAT_DEVICE(
            auto i) {
            using pbat::sim::vbd::kernels::IntegrateVelocity;
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto vtp1 = IntegrateVelocity(FromBuffers<3, 1>(xt, i), FromBuffers<3, 1>(x, i), dt);
            ToBuffers(vtp1, v, i);
        });
    mPositionsAtT = x;
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::ComputeElementElasticEnergies()
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.ComputeElementElasticEnergies");
    auto const nElements = static_cast<GpuIndex>(T.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nElements),
        [x    = x.Raw(),
         T    = T.Raw(),
         GP   = mShapeFunctionGradients.Raw(),
         lame = mLameCoefficients.Raw(),
         w    = mQuadratureWeights.Raw(),
         Ue   = Uetr.Raw()] PBAT_DEVICE(GpuIndex e) {
            using namespace pbat::math::linalg::mini;
            auto tinds                   = FromBuffers<4, 1>(T, e);
            auto xe                      = FromBuffers(x, tinds.Transpose());
            SMatrix<GpuScalar, 4, 3> GPe = FromFlatBuffer<4, 3>(GP, e);
            SVector<GpuScalar, 2> lamee  = FromFlatBuffer<2, 1>(lame, e);
            GpuScalar wg                 = w[e];
            SMatrix<GpuScalar, 3, 3> Fe  = xe * GPe;
            pbat::physics::StableNeoHookeanEnergy<3> Psi{};
            Ue[e] = wg * Psi.eval(Fe, lamee(0), lamee(1));
        });
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::ComputeVertexEnergies(kernels::BackwardEulerMinimization& bdf)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.ComputeVertexEnergies");
    // Reduce into element energies into vertices
    ComputeElementElasticEnergies();
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    kernels::Invoke<kernels::AccumulateVertexEnergiesTraits>(nVertices, mGpuThreadBlockSize, bdf);
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::UpdateChebyshevIterates(GpuIndex k, GpuScalar omega)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.UpdateChebyshevIterates");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [k = k, omega = omega, xkm2 = xkm2.Raw(), xkm1 = xkm1.Raw(), xk = x.Raw()] PBAT_DEVICE(
            auto i) {
            using pbat::sim::vbd::kernels::ChebyshevUpdate;
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto xkm2i = FromBuffers<3, 1>(xkm2, i);
            auto xkm1i = FromBuffers<3, 1>(xkm1, i);
            auto xki   = FromBuffers<3, 1>(xk, i);
            ChebyshevUpdate(k, omega, xkm2i, xkm1i, xki);
            ToBuffers(xkm2i, xkm2, i);
            ToBuffers(xkm1i, xkm1, i);
            ToBuffers(xki, xk, i);
        });
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::UpdateTrustRegionIterates()
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.Integrator.UpdateTrustRegionIterates");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [xkm2 = xkm2.Raw(), xkm1 = xkm1.Raw(), xk = x.Raw(), ftr = ftr.Raw()] PBAT_DEVICE(
            GpuIndex i) {
            using pbat::math::linalg::mini::FromBuffers;
            using pbat::math::linalg::mini::ToBuffers;
            auto xkm2i = FromBuffers<3, 1>(xkm2, i);
            auto xkm1i = FromBuffers<3, 1>(xkm1, i);
            auto xki   = FromBuffers<3, 1>(xk, i);
            xkm2i      = xkm1i;
            xkm1i      = xki;
            ToBuffers(xkm2i, xkm2, i);
            ToBuffers(xkm1i, xkm1, i);
            ToBuffers(xki, xk, i);
            static auto constexpr kMaxPastIterates = std::tuple_size_v<decltype(ftr)> - 1;
            pbat::common::ForRange<0, kMaxPastIterates>([&]<auto k> { ftr[k][i] = ftr[k + 1][i]; });
        });
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

kernels::BackwardEulerMinimization Integrator::BdfDeviceParameters(GpuScalar dt, GpuScalar dt2)
{
    kernels::BackwardEulerMinimization bdf{};
    bdf.dt        = dt;
    bdf.dt2       = dt2;
    bdf.m         = mMass.Raw();
    bdf.xtilde    = mInertialTargetPositions.Raw();
    bdf.xt        = mPositionsAtT.Raw();
    bdf.x         = x.Raw();
    bdf.xb        = xb.Raw();
    bdf.T         = T.Raw();
    bdf.wg        = mQuadratureWeights.Raw();
    bdf.GP        = mShapeFunctionGradients.Raw();
    bdf.lame      = mLameCoefficients.Raw();
    bdf.detHZero  = mDetHZero;
    bdf.GVTp      = mVertexTetrahedronPrefix.Raw();
    bdf.GVTn      = mVertexTetrahedronNeighbours.Raw();
    bdf.GVTilocal = mVertexTetrahedronLocalVertexIndices.Raw();
    bdf.kD        = mRayleighDamping;
    bdf.muC       = mCollisionPenalty;
    bdf.muF       = mFrictionCoefficient;
    bdf.epsv      = mSmoothFrictionRelativeVelocityThreshold;
    bdf.fc        = fc.Raw();
    bdf.F         = cd.F.Raw();
    bdf.XVA       = XVA.Raw();
    bdf.FA        = FA.Raw();
    bdf.nVertices = static_cast<GpuIndex>(x.Size());
    bdf.Uetr      = Uetr.Raw();
    bdf.ftr       = ftr.Raw().back();
    return bdf;
}

} // namespace vbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"

#include <Eigen/SparseCore>
#include <doctest/doctest.h>
#include <span>
#include <vector>

TEST_CASE("[gpu][impl][vbd] Integrator")
{
    using namespace pbat;
    using pbat::common::ToEigen;
    // Arrange
    // Cube mesh
    MatrixX P(3, 8);
    IndexMatrixX V(1, 8);
    IndexMatrixX T(4, 5);
    IndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.reshaped().setLinSpaced(0, static_cast<Index>(P.cols() - 1));
    // Problem parameters
    auto constexpr dt         = GpuScalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 10;
    auto const worldMin       = P.rowwise().minCoeff().cast<GpuScalar>().eval();
    auto const worldMax       = P.rowwise().maxCoeff().cast<GpuScalar>().eval();

    // Act
    using pbat::gpu::impl::vbd::Integrator;
    Integrator vbd{sim::vbd::Data()
                       .WithVolumeMesh(P, T)
                       .WithSurfaceMesh(V, F)
                       .WithBodies(IndexVectorX::Ones(P.cols()))
                       .Construct()};
    vbd.SetSceneBoundingBox(worldMin, worldMax);
    vbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero = GpuScalar{1e-4};
    GpuMatrixX dx =
        ToEigen(vbd.x.Get()).reshaped(P.cols(), P.rows()).transpose() - P.cast<GpuScalar>();
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < GpuScalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}
