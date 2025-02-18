// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "TrustRegionIntegrator.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

namespace pbat::gpu::impl::vbd {

TrustRegionIntegrator::TrustRegionIntegrator(Data const& data)
    : Integrator(data),
      eta(static_cast<GpuScalar>(data.eta)),
      tau(static_cast<GpuScalar>(data.tau)),
      xkm1(data.X.cols()),
      xkm2(data.X.cols()),
      Qinv(),
      aQ(),
      bUseCurvedPath(data.bCurved)
{
    // Q can be derived by simply considering the quadratic function
    // q(t)=at^2 + bt + c, where q(-1)=f^{k-2}, q(0)=f^{k-1}, q(1)=f^k
    // Then, Qinv=Q^{-1}
    // clang-format off
    // Qinv <<  GpuScalar(0.5), GpuScalar(-1), GpuScalar(0.5),
    //         GpuScalar(-0.5),  GpuScalar(0), GpuScalar(0.5),
    //            GpuScalar(0),  GpuScalar(1), GpuScalar(0);
    // clang-format on
    Qinv(0, 0) = GpuScalar(0.5);
    Qinv(0, 1) = GpuScalar(-1);
    Qinv(0, 2) = GpuScalar(0.5);
    Qinv(1, 0) = GpuScalar(-0.5);
    Qinv(1, 1) = GpuScalar(0);
    Qinv(1, 2) = GpuScalar(0.5);
    Qinv(2, 0) = GpuScalar(0);
    Qinv(2, 1) = GpuScalar(1);
    Qinv(2, 2) = GpuScalar(0);
}

void TrustRegionIntegrator::Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations)
{
    if (bUseCurvedPath)
    {
        SolveWithCurvedAccelerationPath(bdf, iterations);
    }
    else
    {
        SolveWithLinearAcceleratedPath(bdf, iterations);
    }
}

void TrustRegionIntegrator::SolveWithLinearAcceleratedPath(
    kernels::BackwardEulerMinimization& bdf,
    GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.SolveWithLinearAcceleratedPath");

    static auto constexpr kNumPastIterates = 3; ///< fk, fkm1, fkm2

    // Objective function
    auto const fObjective = [this, &bdf]() {
        return ObjectiveFunction(bdf.dt, bdf.dt2);
    };

    // TR parameters
    GpuScalar R2{0};
    GpuScalar constexpr zero{1e-6f};

    // TR VBD
    fk = fObjective();
    for (auto k = 0; k < iterations; ++k)
    {
        if (k >= kNumPastIterates - 1)
        {
            ConstructProxy();
            // 2. Throw away x^{k-2},
            // Now fk <- f^{k-1}, f^{k-1} <- f^k
            // x^{k-2} <- x^{k-1}, x^{k-1} <- x^k
            UpdateIterates();
            // 3. Compute VBD step, after which x = x^k + \Delta x
            RunVbdIteration(bdf);
            GpuScalar dx2 = SquaredStepSize(); // |\Delta x|_2^2
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
            GpuScalar t           = -aQ(1) / (GpuScalar{2} * aQ(0)); // Minimum of fproxy
            t                     = std::clamp(t, lower, upper);     // Enforce TR constraint
            TakeLinearStep(t); // Now x = x^{k-1} + t * \Delta x
            // 5. Compute energy at TR step
            fk = fObjective(); // Now fk[kNumPastIterates-1] = f(x^{k-1} + t * \Delta x)
            GpuScalar fNextActual    = fk;
            GpuScalar fCurrentActual = fkm1; // Recall f^{k-1} = f^k
            GpuScalar fNextProxy     = ProxyObjectiveFunction(t);
            GpuScalar fCurrentProxy  = fkm2; // Recall fproxy(0) = f^{k-1}, and f^{k-2} = f^{k-1}
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
                RollbackLinearStep(t);
                fk = fObjective();
            }
        }
        else
        {
            // Un-accelerated VBD iterations, with additional bookkeeping of past iterates and
            // objective function values
            UpdateIterates();
            RunVbdIteration(bdf);
            fk = fObjective();
        }
    }
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void TrustRegionIntegrator::SolveWithCurvedAccelerationPath(
    [[maybe_unused]] kernels::BackwardEulerMinimization& bdf,
    [[maybe_unused]] GpuIndex iterations)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.SolveWithCurvedAccelerationPath");

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

GpuScalar TrustRegionIntegrator::ObjectiveFunction(GpuScalar dt, GpuScalar dt2)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.ObjectiveFunction");

    auto const nElements = static_cast<GpuIndex>(T.Size());
    auto const nVertices = static_cast<GpuIndex>(x.Size());

    // 1. Total elastic potential energy
    GpuScalar U = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nElements),
        cuda::proclaim_return_type<GpuScalar>(
            [x    = x.Raw(),
             T    = T.Raw(),
             GP   = mShapeFunctionGradients.Raw(),
             lame = mLameCoefficients.Raw(),
             w    = mQuadratureWeights.Raw()] PBAT_DEVICE(GpuIndex e) {
                using namespace pbat::math::linalg::mini;
                auto tinds                   = FromBuffers<4, 1>(T, e);
                auto xe                      = FromBuffers(x, tinds.Transpose());
                SMatrix<GpuScalar, 4, 3> GPe = FromFlatBuffer<4, 3>(GP, e);
                SVector<GpuScalar, 2> lamee  = FromFlatBuffer<2, 1>(lame, e);
                GpuScalar wg                 = w[e];
                SMatrix<GpuScalar, 3, 3> Fe  = xe * GPe;
                pbat::physics::StableNeoHookeanEnergy<3> Psi{};
                return wg * Psi.eval(Fe, lamee(0), lamee(1));
            }),
        GpuScalar{0},
        thrust::plus<GpuScalar>());

    // 2. Vertex inertial energy
    GpuScalar K = thrust::transform_reduce(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        cuda::proclaim_return_type<GpuScalar>([x      = x.Raw(),
                                               xtilde = mInertialTargetPositions.Raw(),
                                               m      = mMass.Raw()] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto xi      = FromBuffers<3, 1>(x, i);
            auto xitilde = FromBuffers<3, 1>(xtilde, i);
            return GpuScalar(0.5) * m[i] * SquaredNorm(xi - xitilde);
        }),
        GpuScalar{0},
        thrust::plus<GpuScalar>());

    // 3. Contact energy
    GpuScalar C = thrust::transform_reduce(
        thrust::device,
        cd.av.Data(),
        cd.av.Data() + cd.nActive,
        cuda::proclaim_return_type<GpuScalar>(
            [dt,
             x    = x.Raw(),
             xt   = mPositionsAtT.Raw(),
             V    = cd.V.Raw(),
             F    = cd.F.Raw(),
             fc   = fc.Raw(),
             FA   = FA.Raw(),
             XVA  = XVA.Raw(),
             muC  = mCollisionPenalty,
             muF  = mFrictionCoefficient,
             epsv = mSmoothFrictionRelativeVelocityThreshold] PBAT_DEVICE(GpuIndex v) {
                using namespace pbat::math::linalg::mini;
                static auto constexpr kMaxContacts =
                    kernels::BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
                GpuIndex i = V[v];
                // Load vertex data and vertex-triangle contact data
                auto xi  = FromBuffers<3, 1>(x, i);
                auto xti = FromBuffers<3, 1>(xt, i);
                // Compute contact energy
                kernels::ContactPenalty<kMaxContacts> cp{i, fc, XVA, FA, muC};
                GpuScalar fi = GpuScalar{0};
                for (auto c = 0; c < cp.nContacts; ++c)
                {
                    using pbat::sim::vbd::kernels::AccumulateVertexTriangleContact;
                    auto finds = FromBuffers<3, 1>(F, cp.Triangle(c));
                    auto xtf   = FromBuffers(xt, finds.Transpose());
                    auto xf    = FromBuffers(x, finds.Transpose());
                    fi += AccumulateVertexTriangleContact(
                        xti,
                        xi,
                        xtf,
                        xf,
                        dt,
                        cp.Penalty(c),
                        muF,
                        epsv,
                        static_cast<SVector<GpuScalar, 3>*>(nullptr),
                        static_cast<SMatrix<GpuScalar, 3, 3>*>(nullptr));
                }
                return fi;
            }),
        GpuScalar{0},
        thrust::plus<GpuScalar>());

    GpuScalar f = K + dt2 * U + C;
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
    return f;
}

void TrustRegionIntegrator::UpdateIterates()
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.UpdateIterates");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [xkm2 = xkm2.Raw(), xkm1 = xkm1.Raw(), xk = x.Raw()] PBAT_DEVICE(GpuIndex i) {
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
        });
    fkm2 = fkm1;
    fkm1 = fk;
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

GpuScalar TrustRegionIntegrator::ProxyObjectiveFunction(GpuScalar t) const
{
    return aQ(0) * t * t + aQ(1) * t + aQ(2);
}

void TrustRegionIntegrator::ConstructProxy()
{
    // 1. Compute af,bf,cf s.t. fproxy(t) = af*t^2 + bf*t + cf
    // and fproxy(-1)=f^{k-2}, fproxy(0)=f^{k-1}, fproxy(1)=f^k
    using Vector3 = pbat::math::linalg::mini::SVector<GpuScalar, 3>;
    aQ            = Qinv * Vector3{fkm2, fkm1, fk};
}

GpuScalar TrustRegionIntegrator::SquaredStepSize() const
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.SquaredStepSize");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    auto dx2             = thrust::transform_reduce(
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
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
    return dx2;
}

void TrustRegionIntegrator::TakeLinearStep(GpuScalar t)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.TakeLinearStep");
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
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void TrustRegionIntegrator::RollbackLinearStep(GpuScalar t)
{
    PBAT_PROFILE_NAMED_CUDA_HOST_SCOPE_START(
        ctx,
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.RollbackLinearStep");
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [x = x.Raw(), xkm1 = xkm1.Raw(), t] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto xtr = FromBuffers<3, 1>(x, i);
            auto xk  = FromBuffers<3, 1>(xkm1, i);
            ToBuffers(xk + (xtr - xk) / t, x, i);
        });
    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

} // namespace pbat::gpu::impl::vbd
