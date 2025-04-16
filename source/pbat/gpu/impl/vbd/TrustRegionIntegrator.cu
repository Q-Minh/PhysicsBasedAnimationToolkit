// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "TrustRegionIntegrator.cuh"
#include "pbat/gpu/profiling/Profiling.h"
#include "pbat/math/linalg/mini/Mini.h"
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
      Q(),
      aQ(),
      am(),
      bm(),
      sigmax(),
      ax(data.X.cols()),
      bx(data.X.cols()),
      cx(data.X.cols()),
      bUseCurvedPath(data.bCurved)
{
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
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
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
    PBAT_PROFILE_CUDA_CLOG("----------");
    fk = fObjective();
    tk = GpuScalar{-1};
    PBAT_PROFILE_CUDA_PLOT("vbd.TRI -- f(x)", fk);
    for (auto k = 0; k < iterations; ++k)
    {
        PBAT_PROFILE_CUDA_FLOG("iter={} -----", k);
        if (k >= kNumPastIterates - 1)
        {
            ConstructModel();
            PBAT_PROFILE_CUDA_FLOG(
                "Model function -> aQ={},{},{}, tkm2={},tkm1={},tk={}",
                aQ(0),
                aQ(1),
                aQ(2),
                tkm2,
                tkm1,
                tk);
            // After update
            // fk <- f^{k-1}, f^{k-1} <- f^k
            // x^{k-2} <- x^{k-1}, x^{k-1} <- x^k
            // tk <- 0, t^{k-1} <- t^{k-1} - t^k, t^{k-2} <- t^{k-2} - t^k
            UpdateIterates();
            // Compute VBD step, after which x^k = x^{k-1} + \Delta x
            RunVbdIteration(bdf);
            GpuScalar dx2 = SquaredStepSize(); // |\Delta x|_2^2
            PBAT_PROFILE_CUDA_FLOG("dx2={}", dx2);
            // Ensure TR radius is not smaller than the VBD step,
            // so that we can potentially improve it.
            if (R2 < dx2 + zero)
            {
                // Heuristically initialize radius to tau times the VBD step size.
                // R=tau*|\Delta x|_2
                // => R^2=tau^2*|\Delta x|_2^2
                R2 = tau * tau * dx2;
            }
            PBAT_PROFILE_CUDA_FLOG("TR radius={}", std::sqrt(R2));
            // Compute accelerated step by minimizing model function subject to TR constraint
            // TODO: Determine lower and upper t bounds
            GpuScalar constexpr lower{1};
            GpuScalar const upper = std::sqrt(R2 / dx2); // \f$ t_{\text{upper}} |\Delta x| = R \f$
            PBAT_PROFILE_CUDA_FLOG("TR step upper={}", upper);
            GpuScalar const tstar = ModelOptimalStep();
            GpuScalar const t     = std::clamp(tstar, lower, upper);
            PBAT_PROFILE_CUDA_FLOG("t={},t*={}", t, tstar);
            bool const bIsStepAtLowerBound       = std::abs(t - lower) < zero;
            bool const bShouldTryAcceleratedStep = not bIsStepAtLowerBound;
            bool bStepAccepted{false};
            if (bShouldTryAcceleratedStep)
            {
                TakeLinearStep(t);
                // Compute actual vs expected energy reduction ratio
                fk                 = fObjective();
                GpuScalar fNext    = fk;
                GpuScalar fCurrent = fkm1;
                GpuScalar mNext    = ModelFunction(t);
                GpuScalar mCurrent =
                    fkm1; // t > 1 when bShouldTryAcceleratedStep, so fproxy(0) > fproxy(t)
                GpuScalar const rho = (fCurrent - fNext) / (mCurrent - mNext);
                // Accept step if model function is accurate enough (i.e. the expected energy
                // reduction "matches" the actual energy reduction)
                bStepAccepted = rho > eta;
                PBAT_PROFILE_CUDA_FLOG("rho={},eta={}", rho, eta);
            }
            if (bStepAccepted)
            {
                PBAT_PROFILE_CUDA_CLOG("Accepted TR step");
                bool const bIsStepAtUpperBound = (upper - t) < zero; // upper >= t after std::clamp
                // Increase TR radius if our model function is accurate
                // and the optimal step was outside the current trust region
                if (bIsStepAtUpperBound)
                {
                    R2 *= tau * tau; // R' = tau*R -> R'^2 = tau^2*R^2
                }
                tk = t;
            }
            else
            {
                // Decrease TR radius if our model function is inaccurate
                PBAT_PROFILE_CUDA_CLOG("Rejected TR step");
                R2 /= tau * tau; // R' = R/tau -> R'^2 = R^2/tau^2
                if (bShouldTryAcceleratedStep)
                    RollbackLinearStep(t); // fall-back to initial VBD step
                fk = fObjective();
                tk = tkm1 + 1;
            }
        }
        else
        {
            // Un-accelerated VBD iterations, with additional bookkeeping of past iterates and
            // objective function values
            UpdateIterates();
            RunVbdIteration(bdf);
            fk = fObjective();
            tk = static_cast<GpuScalar>(k);
        }
        PBAT_PROFILE_CUDA_PLOT("vbd.TRI -- f(x)", fk);
    }
}

void TrustRegionIntegrator::SolveWithCurvedAccelerationPath(
    [[maybe_unused]] kernels::BackwardEulerMinimization& bdf,
    [[maybe_unused]] GpuIndex iterations)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.SolveWithCurvedAccelerationPath");

    static auto constexpr kNumPastIterates = 3; ///< fk, fkm1, fkm2

    // Objective function
    auto const fObjective = [this, &bdf]() {
        return ObjectiveFunction(bdf.dt, bdf.dt2);
    };

    // TR parameters
    Scalar R2min             = std::numeric_limits<Scalar>::epsilon();
    Scalar R2                = R2min;
    GpuScalar constexpr zero = std::numeric_limits<GpuScalar>::min();

    // TR VBD
    PBAT_PROFILE_CUDA_CLOG("----------");
    fk = fObjective();
    tk = GpuScalar{-1};
    PBAT_PROFILE_CUDA_PLOT("vbd.TRI -- f(x)", fk);
    for (auto k = 0; k < iterations; ++k)
    {
        PBAT_PROFILE_CUDA_FLOG("iter={} -----", k);
        if (k >= kNumPastIterates - 1)
        {
            PBAT_PROFILE_CUDA_FLOG("TR radius={}", std::sqrt(R2));
            ConstructModel();
            PBAT_PROFILE_CUDA_FLOG(
                "Model function -> am={},bm={}, tkm2={},tkm1={},tk={}",
                am,
                bm,
                tkm2,
                tkm1,
                tk);
            ComputeCurvedPath();
            ComputePolynomialConstraintCoefficients();
            UpdateIterates();
            Scalar const upper    = SolveCurvedTrustRegionConstraint(R2);
            GpuScalar const tstar = ModelOptimalStep();
            GpuScalar const t     = std::clamp(tstar, zero, static_cast<GpuScalar>(upper));
            PBAT_PROFILE_CUDA_FLOG("t={},t*={},tup={}", t, tstar, upper);
            bool const bIsStepAtLowerBound       = (t - zero) < zero;
            bool const bShouldTryAcceleratedStep = not bIsStepAtLowerBound;
            bool bStepAccepted{false};
            if (bShouldTryAcceleratedStep)
            {
                TakeCurvedStep(t);
                // Compute actual vs expected energy reduction ratio
                fk                  = fObjective();
                GpuScalar fnext     = fk;
                GpuScalar fprev     = fkm1; // fkm1 <- fk after UpdateIterates()
                GpuScalar mnext     = ModelFunction(t);
                GpuScalar mprev     = ModelFunction(tk);
                GpuScalar const rho = (fnext - fprev) / (mnext - mprev);
                // Accept step if model function is accurate enough (i.e. the expected energy
                // reduction "matches" the actual energy reduction)
                bStepAccepted = rho > eta;
                PBAT_PROFILE_CUDA_FLOG("rho={},eta={}", rho, eta);
            }
            if (bStepAccepted)
            {
                PBAT_PROFILE_CUDA_CLOG("Accepted TR step");
                bool const bIsStepAtUpperBound = (upper - t) < zero; // upper >= t after std::clamp
                // Increase TR radius if our model function is accurate
                // and the optimal step was outside the current trust region
                if (bIsStepAtUpperBound)
                    R2 *= tau * tau; // R' = tau*R -> R'^2 = tau^2*R^2
                tk = t;
            }
            else
            {
                // Decrease TR radius if our model function is inaccurate
                PBAT_PROFILE_CUDA_CLOG("Rejected TR step");
                R2 = std::max(R2 / (tau * tau), R2min); // R' = R/tau -> R'^2 = R^2/tau^2
                if (bShouldTryAcceleratedStep)
                    RollbackCurvedStep();
                RunVbdIteration(bdf); // fall-back to initial VBD step
                fk = fObjective();
                tk = tkm1 + 1;
            }
        }
        else
        {
            // Un-accelerated VBD iterations, with additional bookkeeping of past iterates and
            // objective function values
            UpdateIterates();
            RunVbdIteration(bdf);
            fk = fObjective();
            tk = static_cast<GpuScalar>(k);
        }
        PBAT_PROFILE_CUDA_PLOT("vbd.TRI -- f(x)", fk);
    }
}

GpuScalar TrustRegionIntegrator::ObjectiveFunction(GpuScalar dt, GpuScalar dt2)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.TrustRegionIntegrator.ObjectiveFunction");

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
            return m[i] * SquaredNorm(xi - xitilde);
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

    GpuScalar f = K / 2 + dt2 * U + C;
    return f;
}

void TrustRegionIntegrator::UpdateIterates()
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.TrustRegionIntegrator.UpdateIterates");
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
    tkm2 = tkm1;
    tkm1 = tk;
}

GpuScalar TrustRegionIntegrator::ModelFunction(GpuScalar t) const
{
    if (bUseCurvedPath)
        return am * t + bm;
    else
        return aQ(0) * t * t + aQ(1) * t + aQ(2);
}

GpuScalar TrustRegionIntegrator::ModelOptimalStep() const
{
    static GpuScalar constexpr zero = std::numeric_limits<GpuScalar>::min();
    static GpuScalar constexpr inf  = std::numeric_limits<GpuScalar>::max();
    if (bUseCurvedPath)
    {
        return (am > zero) ? -inf : inf;
    }
    else
    {
        bool const bIsQuadratic = std::abs(aQ(0)) > zero;
        bool const bIsLinear    = not bIsQuadratic and std::abs(aQ(1)) > zero;
        if (bIsQuadratic)
        {
            bool const bIsPositiveDefinite = aQ(0) > zero;
            // If quadratic function is positive definite, minimum is -b/2a, else +-inf, so we take
            // the positive optimum, so that we always step forward.
            return bIsPositiveDefinite ? -aQ(1) / (GpuScalar{2} * aQ(0)) : inf;
        }
        else if (bIsLinear)
        {
            // If slope of linear function is positive, minimum is -inf, else +inf
            return aQ(1) > 0 ? -inf : inf;
        }
        else
        {
            return GpuScalar{
                0}; // Constant function is minimized everywhere, return trivial solution
        }
    }
}

void TrustRegionIntegrator::ConstructModel()
{
    if (bUseCurvedPath)
    {
        using Matrix32 = pbat::math::linalg::mini::SMatrix<GpuScalar, 3, 2>;
        using Matrix22 = pbat::math::linalg::mini::SMatrix<GpuScalar, 2, 2>;
        using Vector2  = pbat::math::linalg::mini::SVector<GpuScalar, 2>;
        Matrix32 J;
        J(0, 0)      = tkm2;
        J(1, 0)      = tkm1;
        J(2, 0)      = tk;
        J(0, 1)      = GpuScalar(1);
        J(1, 1)      = GpuScalar(1);
        J(2, 1)      = GpuScalar(1);
        Matrix22 JTJ = J.Transpose() * J;
        Vector2 JTf  = J.Transpose() * Vector3{fkm2, fkm1, fk};
        auto abf     = Inverse(JTJ) * JTf;
        am           = abf(0);
        bm           = abf(1);
    }
    else
    {
        // Translate time so that tk = 0
        tkm2 -= tk;
        tkm1 -= tk;
        tk      = GpuScalar{0};
        Q(0, 0) = tkm2 * tkm2;
        Q(1, 0) = tkm1 * tkm1;
        Q(2, 0) = tk * tk;
        Q(0, 1) = tkm2;
        Q(1, 1) = tkm1;
        Q(2, 1) = tk;
        Q(0, 2) = GpuScalar{1};
        Q(1, 2) = GpuScalar{1};
        Q(2, 2) = GpuScalar{1};
        aQ      = Inverse(Q) * Vector3{fkm2, fkm1, fk};
    }
}

GpuScalar TrustRegionIntegrator::SquaredStepSize() const
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.TrustRegionIntegrator.SquaredStepSize");
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
    return dx2;
}

void TrustRegionIntegrator::TakeLinearStep(GpuScalar t)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.TrustRegionIntegrator.TakeLinearStep");
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
}

void TrustRegionIntegrator::RollbackLinearStep(GpuScalar t)
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE("pbat.gpu.impl.vbd.TrustRegionIntegrator.RollbackLinearStep");
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
}

void TrustRegionIntegrator::ComputePolynomialConstraintCoefficients()
{
    PBAT_PROFILE_CUDA_NAMED_SCOPE(
        "pbat.gpu.impl.vbd.TrustRegionIntegrator.ComputePolynomialConstraintCoefficients");

    auto const nVertices = static_cast<GpuIndex>(x.Size());
    auto begin           = thrust::make_counting_iterator(0);
    auto end             = thrust::make_counting_iterator(nVertices);
    sigmax(0)            = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        cuda::proclaim_return_type<Scalar>([ax = ax.Raw(), tk = tk, tkm2 = tkm2] PBAT_DEVICE(
                                               GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto axi = FromBuffers<3, 1>(ax, i);
            Scalar const z0 =
                1.0 / (((tk) * (tk) * (tk) * (tk)) - 4 * ((tk) * (tk) * (tk)) * tkm2 +
                       6 * ((tk) * (tk)) * ((tkm2) * (tkm2)) - 4 * tk * ((tkm2) * (tkm2) * (tkm2)) +
                       ((tkm2) * (tkm2) * (tkm2) * (tkm2)));
            return z0 * ((axi[0]) * (axi[0])) + z0 * ((axi[1]) * (axi[1])) +
                   z0 * ((axi[2]) * (axi[2]));
        }),
        Scalar{0},
        thrust::plus<Scalar>());
    sigmax(1) = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        cuda::proclaim_return_type<Scalar>(
            [ax = ax.Raw(), bx = bx.Raw(), tk = tk, tkm2 = tkm2] PBAT_DEVICE(GpuIndex i) {
                using namespace pbat::math::linalg::mini;
                auto axi        = FromBuffers<3, 1>(ax, i);
                auto bxi        = FromBuffers<3, 1>(bx, i);
                Scalar const z0 = ((tk) * (tk) * (tk));
                Scalar const z1 = ((tkm2) * (tkm2) * (tkm2));
                Scalar const z2 = ((tkm2) * (tkm2));
                Scalar const z3 = ((tk) * (tk));
                Scalar const z4 = 1.0 / (3 * tk * z2 - 3 * tkm2 * z3 + z0 - z1);
                Scalar const z5 = 4 * tk;
                Scalar const z6 =
                    z5 / (((tk) * (tk) * (tk) * (tk)) + ((tkm2) * (tkm2) * (tkm2) * (tkm2)) -
                          4 * tkm2 * z0 - z1 * z5 + 6 * z2 * z3);
                return 2 * z4 * axi[0] * bxi[0] + 2 * z4 * axi[1] * bxi[1] +
                       2 * z4 * axi[2] * bxi[2] - z6 * ((axi[0]) * (axi[0])) -
                       z6 * ((axi[1]) * (axi[1])) - z6 * ((axi[2]) * (axi[2]));
            }),
        Scalar{0},
        thrust::plus<Scalar>());
    sigmax(2) = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        cuda::proclaim_return_type<Scalar>([xk   = x.Raw(),
                                            ax   = ax.Raw(),
                                            bx   = bx.Raw(),
                                            cx   = cx.Raw(),
                                            tk   = tk,
                                            tkm2 = tkm2] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto xki         = FromBuffers<3, 1>(xk, i);
            auto axi         = FromBuffers<3, 1>(ax, i);
            auto bxi         = FromBuffers<3, 1>(bx, i);
            auto cxi         = FromBuffers<3, 1>(cx, i);
            Scalar const z0  = ((tk) * (tk));
            Scalar const z1  = ((tkm2) * (tkm2));
            Scalar const z2  = 1.0 / (-2 * tk * tkm2 + z0 + z1);
            Scalar const z3  = 2 * z2;
            Scalar const z4  = z3 * axi[0];
            Scalar const z5  = z3 * axi[1];
            Scalar const z6  = z3 * axi[2];
            Scalar const z7  = ((tk) * (tk) * (tk));
            Scalar const z8  = ((tkm2) * (tkm2) * (tkm2));
            Scalar const z9  = 6 * tk / (3 * tk * z1 - 3 * tkm2 * z0 + z7 - z8);
            Scalar const z10 = 6 * z0;
            Scalar const z11 =
                z10 / (((tk) * (tk) * (tk) * (tk)) - 4 * tk * z8 +
                       ((tkm2) * (tkm2) * (tkm2) * (tkm2)) - 4 * tkm2 * z7 + z1 * z10);
            return z11 * ((axi[0]) * (axi[0])) + z11 * ((axi[1]) * (axi[1])) +
                   z11 * ((axi[2]) * (axi[2])) + z2 * ((bxi[0]) * (bxi[0])) +
                   z2 * ((bxi[1]) * (bxi[1])) + z2 * ((bxi[2]) * (bxi[2])) + z4 * cxi[0] -
                   z4 * xki[0] + z5 * cxi[1] - z5 * xki[1] + z6 * cxi[2] - z6 * xki[2] -
                   z9 * axi[0] * bxi[0] - z9 * axi[1] * bxi[1] - z9 * axi[2] * bxi[2];
        }),
        Scalar{0},
        thrust::plus<Scalar>());
    sigmax(3) = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        cuda::proclaim_return_type<Scalar>([xk   = x.Raw(),
                                            ax   = ax.Raw(),
                                            bx   = bx.Raw(),
                                            cx   = cx.Raw(),
                                            tk   = tk,
                                            tkm2 = tkm2] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto axi         = FromBuffers<3, 1>(ax, i);
            auto bxi         = FromBuffers<3, 1>(bx, i);
            auto cxi         = FromBuffers<3, 1>(cx, i);
            auto xki         = FromBuffers<3, 1>(xk, i);
            Scalar const z0  = 1.0 / (tk - tkm2);
            Scalar const z1  = 2 * z0;
            Scalar const z2  = ((tk) * (tk));
            Scalar const z3  = ((tkm2) * (tkm2));
            Scalar const z4  = 2 * tk;
            Scalar const z5  = 1.0 / (-tkm2 * z4 + z2 + z3);
            Scalar const z6  = z4 * z5;
            Scalar const z7  = 4 * tk;
            Scalar const z8  = z5 * z7;
            Scalar const z9  = ((tk) * (tk) * (tk));
            Scalar const z10 = ((tkm2) * (tkm2) * (tkm2));
            Scalar const z11 = 1.0 / (3 * tk * z3 - 3 * tkm2 * z2 - z10 + z9);
            Scalar const z12 = 4 * z9;
            Scalar const z13 =
                z12 / (((tk) * (tk) * (tk) * (tk)) + ((tkm2) * (tkm2) * (tkm2) * (tkm2)) -
                       tkm2 * z12 - z10 * z7 + 6 * z2 * z3);
            return 4 * tk * z5 * axi[0] * xki[0] + 4 * tk * z5 * axi[1] * xki[1] +
                   4 * tk * z5 * axi[2] * xki[2] + 2 * z0 * bxi[0] * cxi[0] +
                   2 * z0 * bxi[1] * cxi[1] + 2 * z0 * bxi[2] * cxi[2] - z1 * bxi[0] * xki[0] -
                   z1 * bxi[1] * xki[1] - z1 * bxi[2] * xki[2] + 6 * z11 * z2 * axi[0] * bxi[0] +
                   6 * z11 * z2 * axi[1] * bxi[1] + 6 * z11 * z2 * axi[2] * bxi[2] -
                   z13 * ((axi[0]) * (axi[0])) - z13 * ((axi[1]) * (axi[1])) -
                   z13 * ((axi[2]) * (axi[2])) - z6 * ((bxi[0]) * (bxi[0])) -
                   z6 * ((bxi[1]) * (bxi[1])) - z6 * ((bxi[2]) * (bxi[2])) - z8 * axi[0] * cxi[0] -
                   z8 * axi[1] * cxi[1] - z8 * axi[2] * cxi[2];
        }),
        Scalar{0},
        thrust::plus<Scalar>());
    sigmax(4) = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        cuda::proclaim_return_type<Scalar>([xk   = x.Raw(),
                                            ax   = ax.Raw(),
                                            bx   = bx.Raw(),
                                            cx   = cx.Raw(),
                                            tk   = tk,
                                            tkm2 = tkm2] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto xki         = FromBuffers<3, 1>(xk, i);
            auto axi         = FromBuffers<3, 1>(ax, i);
            auto bxi         = FromBuffers<3, 1>(bx, i);
            auto cxi         = FromBuffers<3, 1>(cx, i);
            Scalar const z0  = 2 * cxi[0];
            Scalar const z1  = 2 * cxi[1];
            Scalar const z2  = 2 * cxi[2];
            Scalar const z3  = 1.0 / (tk - tkm2);
            Scalar const z4  = tk * z3;
            Scalar const z5  = 2 * tk;
            Scalar const z6  = z3 * z5;
            Scalar const z7  = ((tk) * (tk));
            Scalar const z8  = ((tkm2) * (tkm2));
            Scalar const z9  = z7 / (-tkm2 * z5 + z7 + z8);
            Scalar const z10 = 2 * z9;
            Scalar const z11 = ((tk) * (tk) * (tk) * (tk));
            Scalar const z12 = ((tkm2) * (tkm2) * (tkm2));
            Scalar const z13 = ((tk) * (tk) * (tk));
            Scalar const z14 = z11 / (-4 * tk * z12 + ((tkm2) * (tkm2) * (tkm2) * (tkm2)) -
                                      4 * tkm2 * z13 + z11 + 6 * z7 * z8);
            Scalar const z15 = 2 * z13 / (3 * tk * z8 - 3 * tkm2 * z7 - z12 + z13);
            return -z0 * z4 * bxi[0] + z0 * z9 * axi[0] - z0 * xki[0] - z1 * z4 * bxi[1] +
                   z1 * z9 * axi[1] - z1 * xki[1] - z10 * axi[0] * xki[0] - z10 * axi[1] * xki[1] -
                   z10 * axi[2] * xki[2] + z14 * ((axi[0]) * (axi[0])) +
                   z14 * ((axi[1]) * (axi[1])) + z14 * ((axi[2]) * (axi[2])) -
                   z15 * axi[0] * bxi[0] - z15 * axi[1] * bxi[1] - z15 * axi[2] * bxi[2] -
                   z2 * z4 * bxi[2] + z2 * z9 * axi[2] - z2 * xki[2] + z6 * bxi[0] * xki[0] +
                   z6 * bxi[1] * xki[1] + z6 * bxi[2] * xki[2] + z9 * ((bxi[0]) * (bxi[0])) +
                   z9 * ((bxi[1]) * (bxi[1])) + z9 * ((bxi[2]) * (bxi[2])) + ((cxi[0]) * (cxi[0])) +
                   ((cxi[1]) * (cxi[1])) + ((cxi[2]) * (cxi[2])) + ((xki[0]) * (xki[0])) +
                   ((xki[1]) * (xki[1])) + ((xki[2]) * (xki[2]));
        }),
        Scalar{0},
        thrust::plus<Scalar>());
}

Scalar TrustRegionIntegrator::SolveCurvedTrustRegionConstraint([[maybe_unused]] Scalar R2) const
{
    return Scalar();
}

void TrustRegionIntegrator::ComputeCurvedPath()
{
    cx                   = x; // c_{i,d} = x_k^{i,d}
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [xk   = x.Raw(),
         xkm1 = xkm1.Raw(),
         xkm2 = xkm2.Raw(),
         ax   = ax.Raw(),
         bx   = bx.Raw(),
         tkm2 = tkm2,
         tkm1 = tkm1,
         tk   = tk] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            using Matrix2 = pbat::math::linalg::mini::SMatrix<GpuScalar, 2, 2>;
            // NOTE: tbar_{k-2} = \frac{t_{k-2} - t_k}{t_k - t_{k-2}} = -1
            GpuScalar tbarkm1 = (tkm1 - tk) / (tk - tkm2);
            Matrix2 V;
            V(0, 0)        = GpuScalar(1);
            V(1, 0)        = tbarkm1 * tbarkm1;
            V(0, 1)        = GpuScalar(-1);
            V(1, 1)        = tbarkm1;
            auto Vinv      = Inverse(V);
            using Matrix32 = pbat::math::linalg::mini::SMatrix<GpuScalar, 3, 2>;
            Matrix32 AB;
            pbat::common::ForRange<0, 3>([&]<auto d>() {
                using Vector2 = pbat::math::linalg::mini::SVector<GpuScalar, 2>;
                Vector2 b{xkm2[d][i] - xk[d][i], xkm1[d][i] - xk[d][i]};
                auto ab  = (Vinv * b);
                AB(d, 0) = ab(0);
                AB(d, 1) = ab(1);
            });
            ToBuffers(AB.Col(0), ax, i);
            ToBuffers(AB.Col(1), bx, i);
        });
}

void TrustRegionIntegrator::TakeCurvedStep(GpuScalar t)
{
    auto tbar            = (t - tk) / (tk - tkm2);
    auto const nVertices = static_cast<GpuIndex>(x.Size());
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nVertices),
        [x    = x.Raw(),
         xkm1 = xkm1.Raw(),
         xkm2 = xkm2.Raw(),
         ax   = ax.Raw(),
         bx   = bx.Raw(),
         cx   = cx.Raw(),
         tbar] PBAT_DEVICE(GpuIndex i) {
            using namespace pbat::math::linalg::mini;
            auto axi = FromBuffers<3, 1>(ax, i);
            auto bxi = FromBuffers<3, 1>(bx, i);
            auto cxi = FromBuffers<3, 1>(cx, i);
            auto xtr = axi * tbar * tbar + bxi * tbar + cxi;
            ToBuffers(xtr, x, i);
        });
}

void TrustRegionIntegrator::RollbackCurvedStep()
{
    // Recall that x(t) = ax (t-t_k)^2 + bx (t-t_k) + cx .
    // We know that x(t_k) = x_k, thus
    // x(t_k) = cx = x_k
    x = cx;
}

} // namespace pbat::gpu::impl::vbd

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <Eigen/SparseCore>
#include <doctest/doctest.h>
#include <span>
#include <vector>

TEST_CASE("[gpu][impl][vbd] TrustRegionIntegrator")
{
    // using namespace pbat;
    // using pbat::common::ToEigen;
    // // Arrange
    // // Cube mesh
    // MatrixX P(3, 8);
    // IndexMatrixX V(1, 8);
    // IndexMatrixX T(4, 5);
    // IndexMatrixX F(3, 12);
    // // clang-format off
    // P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
    //      0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
    //      0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    // T << 0, 3, 5, 6, 0,
    //      1, 2, 4, 7, 5,
    //      3, 0, 6, 5, 3,
    //      5, 6, 0, 3, 6;
    // F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
    //      1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
    //      4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // // clang-format on
    // V.reshaped().setLinSpaced(0, static_cast<Index>(P.cols() - 1));
    // // Problem parameters
    // auto constexpr dt         = GpuScalar{1e-2};
    // auto constexpr substeps   = 1;
    // auto constexpr iterations = 10;
    // auto const worldMin       = P.rowwise().minCoeff().cast<GpuScalar>().eval();
    // auto const worldMax       = P.rowwise().maxCoeff().cast<GpuScalar>().eval();
    // Scalar constexpr eta{0.1};
    // Scalar constexpr tau{2.};
    // Scalar constexpr Y{1e7};
    // Scalar constexpr nu{0.45};
    // Scalar constexpr rho{1e3};
    // auto const [mu, lambda] = physics::LameCoefficients(Y, nu);

    // SUBCASE("Free fall")
    // {
    //     auto data = sim::vbd::Data().WithVolumeMesh(P, T).WithSurfaceMesh(V, F).WithMaterial(
    //         VectorX::Constant(T.cols(), rho),
    //         VectorX::Constant(T.cols(), mu),
    //         VectorX::Constant(T.cols(), lambda));
    //     SUBCASE("Linear path")
    //     {
    //         data.WithTrustRegionAcceleration(eta, tau, false);
    //     }
    //     SUBCASE("Curved path")
    //     {
    //         data.WithTrustRegionAcceleration(eta, tau, true);
    //     }
    //     // Act
    //     using pbat::gpu::impl::vbd::TrustRegionIntegrator;
    //     TrustRegionIntegrator vbd{data.Construct()};
    //     vbd.SetSceneBoundingBox(worldMin, worldMax);
    //     vbd.Step(dt, iterations, substeps);

    //     // Assert
    //     auto constexpr zero = GpuScalar{1e-4};
    //     GpuMatrixX dx =
    //         ToEigen(vbd.x.Get()).reshaped(P.cols(), P.rows()).transpose() - P.cast<GpuScalar>();
    //     bool const bVerticesFallUnderGravity = (dx.row(2).array() < GpuScalar{0}).all();
    //     CHECK(bVerticesFallUnderGravity);
    //     bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    //     CHECK(bVerticesOnlyFall);
    // }
}