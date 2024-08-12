// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "XpbdImpl.cuh"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>
#include <cuda/std/cmath>
#include <exception>
#include <sstream>
#include <string>
#include <thrust/async/for_each.h>
#include <thrust/async/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

namespace pbat {
namespace gpu {
namespace xpbd {
namespace kernels {

struct FInitializeNeoHookeanConstraint
{
    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;
        // Load vertex positions of element c
        GpuIndex const v[4] = {T[0][c], T[1][c], T[2][c], T[3][c]};
        Matrix<GpuScalar, 3, 4> xc{};
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                xc(d, j) = x[d][v[j]];
        // Compute shape matrix and its inverse
        Matrix<GpuScalar, 3, 3> Ds = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0)));
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        DmInvC = Inverse(Ds);
        // Compute constraint compliance
        GpuScalar const tetVolume = Determinant(Ds) / GpuScalar{6.};
        MatrixView<GpuScalar, 2, 1> alphac{alpha + 2 * c};
        MatrixView<GpuScalar, 2, 1> lamec{lame + 2 * c};
        alphac(0) = GpuScalar{1.} / (lamec(0) * tetVolume);
        alphac(1) = GpuScalar{1.} / (lamec(1) * tetVolume);
        // Compute rest stability
        gamma[c] = GpuScalar{1.} + lamec(0) / lamec(1);
    }
    std::array<GpuScalar*, 3> x;
    std::array<GpuIndex*, 4> T;
    GpuScalar* lame;
    GpuScalar* DmInv;
    GpuScalar* alpha;
    GpuScalar* gamma;
};

struct FInitializeSolution
{
    __device__ void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            x[d][i] = xt[d][i] + dt * v[d][i] + dt2 * minv[i] * f[d][i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    std::array<GpuScalar*, 3> f;
    GpuScalar* minv;
    GpuScalar dt;
    GpuScalar dt2;
};

struct FStableNeoHookeanConstraint
{
    __device__ void Project(
        GpuScalar C,
        math::linalg::Matrix<GpuScalar, 3, 4> const& gradC,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        GpuScalar dlambda =
            -(C + atilde * lambdac) /
            (minvc(0) * SquaredNorm(gradC.Col(0)) + minvc(1) * SquaredNorm(gradC.Col(1)) +
             minvc(2) * SquaredNorm(gradC.Col(2)) + minvc(3) * SquaredNorm(gradC.Col(3)) + atilde);
        lambdac += dlambda;
        xc.Col(0) += (minvc(0) * dlambda) * gradC.Col(0);
        xc.Col(1) += (minvc(1) * dlambda) * gradC.Col(1);
        xc.Col(2) += (minvc(2) * dlambda) * gradC.Col(2);
        xc.Col(3) += (minvc(3) * dlambda) * gradC.Col(3);
    }

    __device__ void ProjectHydrostatic(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar gammac,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C               = Determinant(F) - gammac;
        Matrix<GpuScalar, 3, 3> P{};
        P.Col(0) = Cross(F.Col(1), F.Col(2));
        P.Col(1) = Cross(F.Col(2), F.Col(0));
        P.Col(2) = Cross(F.Col(0), F.Col(1));
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = P * DmInvC.Transpose();
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    __device__ void ProjectDeviatoric(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C               = Norm(F);
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = (F * DmInvC.Transpose()) / (C /*+ 1e-8*/);
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;

        // 1. Load constraint data in local memory
        GpuIndex const v[4] = {T[0][c], T[1][c], T[2][c], T[3][c]};
        Matrix<GpuScalar, 3, 4> xc{};
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                xc(d, j) = x[d][v[j]];
        Matrix<GpuScalar, 4, 1> minvc{};
        for (auto j = 0; j < 4; ++j)
            minvc(j) = minv[v[j]];
        Matrix<GpuScalar, 2, 1> lambdac{};
        lambdac(0) = lambda[2 * c];
        lambdac(1) = lambda[2 * c + 1];
        Matrix<GpuScalar, 2, 1> atilde{};
        atilde(0) = alpha[2 * c] / dt2;
        atilde(1) = alpha[2 * c + 1] / dt2;

        // 2. Project elastic constraints
        ProjectDeviatoric(c, minvc, atilde(0), lambdac(0), xc);
        ProjectHydrostatic(c, minvc, atilde(1), gamma[c], lambdac(1), xc);

        // 3. Update global "Lagrange" multipliers and positions
        lambda[2 * c]     = lambdac(0);
        lambda[2 * c + 1] = lambdac(1);
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                x[d][v[j]] = xc(d, j);
    }

    std::array<GpuScalar*, 3> x;
    GpuScalar* lambda;

    std::array<GpuIndex*, 4> T;
    GpuScalar* minv;
    GpuScalar* alpha;
    GpuScalar* DmInv;
    GpuScalar* gamma;
    GpuScalar dt2;
};

__device__ math::linalg::Matrix<GpuScalar, 3, 1>
PointOnPlane(auto const& X, auto const& P, auto const& n)
{
    using namespace pbat::gpu::math::linalg;
    GpuScalar const t          = (n.Transpose() * (X - P))(0, 0);
    Matrix<GpuScalar, 3, 1> Xp = X - t * n;
    return Xp;
}

__device__ math::linalg::Matrix<GpuScalar, 3, 1>
TriangleBarycentricCoordinates(auto const& AP, auto const& AB, auto const& AC)
{
    using namespace pbat::gpu::math::linalg;
    GpuScalar const d00   = (AB.Transpose() * AB)(0, 0);
    GpuScalar const d01   = (AB.Transpose() * AC)(0, 0);
    GpuScalar const d11   = (AC.Transpose() * AC)(0, 0);
    GpuScalar const d20   = (AP.Transpose() * AB)(0, 0);
    GpuScalar const d21   = (AP.Transpose() * AC)(0, 0);
    GpuScalar const denom = d00 * d11 - d01 * d01;
    GpuScalar const v     = (d11 * d20 - d01 * d21) / denom;
    GpuScalar const w     = (d00 * d21 - d01 * d20) / denom;
    GpuScalar const u     = GpuScalar{1.} - v - w;
    Matrix<GpuScalar, 3, 1> uvw{};
    uvw(0, 0) = u;
    uvw(1, 0) = v;
    uvw(2, 0) = w;
    return uvw;
};

struct FVertexTriangleContactConstraint
{
    using OverlapType = typename geometry::SweepAndPruneImpl::OverlapType;

    __device__ bool ProjectVertexTriangle(
        GpuScalar minvv,
        math::linalg::Matrix<GpuScalar, 3, 1> const& minvf,
        math::linalg::Matrix<GpuScalar, 3, 1> const& xvt,
        math::linalg::Matrix<GpuScalar, 3, 3> const& xft,
        GpuScalar atildec,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 1>& xv,
        math::linalg::Matrix<GpuScalar, 3, 3>& xf)
    {
        using namespace pbat::gpu::math::linalg;
        // Numerically zero inverse mass makes the Schur complement ill-conditioned/singular
        if (minvv < GpuScalar{1e-10})
            return false;
        // Compute triangle normal
        Matrix<GpuScalar, 3, 1> T1       = xf.Col(1) - xf.Col(0);
        Matrix<GpuScalar, 3, 1> T2       = xf.Col(2) - xf.Col(0);
        Matrix<GpuScalar, 3, 1> n        = Cross(T1, T2);
        GpuScalar const doublearea       = Norm(n);
        bool const bIsTriangleDegenerate = doublearea <= GpuScalar{1e-8f};
        if (bIsTriangleDegenerate)
            return false;

        n /= doublearea;
        Matrix<GpuScalar, 3, 1> xc = PointOnPlane(xv, xf.Col(0), n);
        // Check if xv projects to the triangle's interior by checking its barycentric coordinates
        Matrix<GpuScalar, 3, 1> b = TriangleBarycentricCoordinates(xc - xf.Col(0), T1, T2);
        // If xv doesn't project inside triangle, then we don't generate a contact response
        // clang-format off
        bool const bIsVertexInsideTriangle = 
            (b(0) >= GpuScalar{0.f}) and (b(0) <= GpuScalar{1.f}) and
            (b(1) >= GpuScalar{0.f}) and (b(1) <= GpuScalar{1.f}) and
            (b(2) >= GpuScalar{0.f}) and (b(2) <= GpuScalar{1.f});
        // clang-format on
        if (not bIsVertexInsideTriangle)
            return false;

        // Project xv onto triangle's plane into xc
        GpuScalar const C = (n.Transpose() * (xv - xf.Col(0)))(0, 0);
        // If xv is positively oriented w.r.t. triangles xf, there is no penetration
        if (C > GpuScalar{0.})
            return false;
        // Prevent super violent projection for stability, i.e. if the collision constraint
        // violation is already too large, we give up.
        if (C < -kMaxPenetration)
            return false;

        // Project constraints:
        // We assume that the triangle is static (although it is not), so that the gradient is n for
        // the vertex.

        // Collision constraint
        GpuScalar dlambda          = -(C + atildec * lambdac) / (minvv + atildec);
        Matrix<GpuScalar, 3, 1> dx = dlambda * minvv * n;
        xv += dx;
        lambdac += dlambda;

        // Friction constraint
        GpuScalar const d = Norm(dx);
        dx                = (xv - xvt) - (xf * b - xft * b);
        dx                = dx - n * n.Transpose() * dx;
        GpuScalar const dxd = Norm(dx);
        if (dxd > muS * d)
        {
            if constexpr (std::is_same_v<GpuScalar, float>)
                dx *= cuda::std::fminf(muK * d / dxd, 1.f);
            if constexpr (std::is_same_v<GpuScalar, double>)
                dx *= cuda::std::fminl(muK * d / dxd, 1.);
        }
        xv += dx;
        return true;
    }

    __device__ void SetParticlePosition(
        GpuIndex v,
        math::linalg::Matrix<GpuScalar, 3, 1> const& xv,
        std::array<GpuScalar*, 3> const& xx)
    {
        for (auto d = 0; d < 3; ++d)
            xx[d][v] = xv(d, 0);
    }

    __device__ math::linalg::Matrix<GpuScalar, 3, 1>
    GetParticlePosition(GpuIndex v, std::array<GpuScalar*, 3> const& xx)
    {
        using namespace pbat::gpu::math::linalg;
        Matrix<GpuScalar, 3, 1> xv{};
        for (auto d = 0; d < 3; ++d)
            xv(d, 0) = xx[d][v];
        return xv;
    }

    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;
        GpuIndex vv    = V[0][overlaps[c].first]; ///< Colliding vertex
        GpuIndex vf[3] = {
            F[0][overlaps[c].second],
            F[1][overlaps[c].second],
            F[2][overlaps[c].second]}; ///< Colliding triangle

        Matrix<GpuScalar, 3, 1> xv = GetParticlePosition(vv, x);
        Matrix<GpuScalar, 3, 3> xf{};
        for (auto j = 0; j < 3; ++j)
            xf.Col(j) = GetParticlePosition(vf[j], x);

        Matrix<GpuScalar, 3, 1> xvt = GetParticlePosition(vv, xt);
        Matrix<GpuScalar, 3, 3> xft{};
        for (auto j = 0; j < 3; ++j)
            xft.Col(j) = GetParticlePosition(vf[j], xt);

        GpuScalar const minvv = minv[vv];
        Matrix<GpuScalar, 3, 1> minvf{};
        for (auto j = 0; j < 3; ++j)
            minvf(j, 0) = minv[vf[j]];
        GpuScalar atildec = alpha[c] / dt2;
        GpuScalar lambdac = lambda[c];

        // 2. Project collision constraint
        if (not ProjectVertexTriangle(minvv, minvf, xvt, xft, atildec, lambdac, xv, xf))
            return;

        // 3. Update global positions
        // lambda[c] = lambdac;
        SetParticlePosition(vv, xv, x);
    }

    std::array<GpuScalar*, 3> x;
    GpuScalar* lambda;

    OverlapType const* overlaps;
    std::array<GpuIndex*, 4> V;
    std::array<GpuIndex*, 4> F;
    GpuScalar* minv;
    GpuScalar* alpha;
    std::array<GpuScalar*, 3> xt;
    GpuScalar dt2;
    GpuScalar const kMaxPenetration;
    GpuScalar const muS; ///< Static Coulomb friction coefficient
    GpuScalar const muK; ///< Dynamic Coulomb friction coefficient
};

struct FUpdateSolution
{
    __device__ void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            v[d][i]  = (x[d][i] - xt[d][i]) / dt;
            xt[d][i] = x[d][i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    GpuScalar dt;
};

} // namespace kernels

XpbdImpl::XpbdImpl(
    Eigen::Ref<GpuMatrixX const> const& Xin,
    Eigen::Ref<GpuIndexMatrixX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin,
    Eigen::Ref<GpuIndexMatrixX const> const& Tin,
    std::size_t nMaxVertexTriangleOverlaps,
    GpuScalar kMaxCollisionPenetration)
    : X(Xin),
      V(Vin),
      F(Fin),
      T(Tin),
      SAP(Xin.cols() + Fin.cols(), nMaxVertexTriangleOverlaps),
      mPositions(Xin.cols()),
      mVelocities(Xin.cols()),
      mExternalForces(Xin.cols()),
      mMassInverses(Xin.cols()),
      mLame(2 * Tin.cols()),
      mShapeMatrixInverses(9 * Tin.cols()),
      mRestStableGamma(Tin.cols()),
      mLagrangeMultipliers(),
      mCompliance(),
      mPartitions(),
      mStaticFrictionCoefficient{0.5},
      mDynamicFrictionCoefficient{0.3},
      mAverageEdgeLength{},
      mMaxCollisionPenetration{kMaxCollisionPenetration}
{
    mLagrangeMultipliers[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mLagrangeMultipliers[Collision].Resize(X.NumberOfPoints());
    mCompliance[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mCompliance[Collision].Resize(V.NumberOfSimplices());
    mAverageEdgeLength =
        (GpuScalar{1.} / GpuScalar{3.}) *
        ((Xin(Eigen::all, Fin.row(0)) - Xin(Eigen::all, Fin.row(1))).colwise().norm().mean() +
         (Xin(Eigen::all, Fin.row(1)) - Xin(Eigen::all, Fin.row(2))).colwise().norm().mean() +
         (Xin(Eigen::all, Fin.row(2)) - Xin(Eigen::all, Fin.row(0))).colwise().norm().mean());
    // Initialize particle data
    for (auto d = 0; d < X.Dimensions(); ++d)
    {
        thrust::copy(X.x[d].begin(), X.x[d].end(), mPositions[d].begin());
        thrust::fill(mVelocities[d].begin(), mVelocities[d].end(), GpuScalar{0.});
        thrust::fill(mExternalForces[d].begin(), mExternalForces[d].end(), GpuScalar{0.});
        thrust::fill(
            mMassInverses.Data(),
            mMassInverses.Data() + mMassInverses.Size(),
            GpuScalar{1.});
    }
}

void XpbdImpl::PrepareConstraints()
{
    thrust::fill(
        thrust::device,
        mCompliance[Collision].Data(),
        mCompliance[Collision].Data() + mCompliance[Collision].Size(),
        GpuScalar{0.});
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<GpuIndex>(0),
        thrust::make_counting_iterator<GpuIndex>(T.NumberOfSimplices()),
        kernels::FInitializeNeoHookeanConstraint{
            X.x.Raw(),
            T.inds.Raw(),
            mLame.Raw(),
            mShapeMatrixInverses.Raw(),
            mCompliance[StableNeoHookean].Raw(),
            mRestStableGamma.Raw()});
}

void XpbdImpl::Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps)
{
    GpuScalar const sdt       = dt / static_cast<GpuScalar>(substeps);
    GpuScalar const sdt2      = sdt * sdt;
    GpuIndex const nParticles = static_cast<GpuIndex>(NumberOfParticles());
    // Detect collision candidates and setup collision constraint solve
    using OverlapType = typename geometry::SweepAndPruneImpl::OverlapType;
    SAP.SortAndSweep(X, V, F, GpuScalar{1e-3});
    GpuIndex const nVertexTriangleOverlaps      = SAP.no.Get();
    common::Buffer<OverlapType> const& overlaps = SAP.o;

    auto& nextPositions = X.x;
    for (auto s = 0; s < substeps; ++s)
    {
        // Reset "Lagrange" multipliers
        for (auto d = 0; d < kConstraintTypes; ++d)
        {
            thrust::fill(
                thrust::device,
                mLagrangeMultipliers[d].Data(),
                mLagrangeMultipliers[d].Data() + mLagrangeMultipliers[d].Size(),
                GpuScalar{0.});
        }
        // Initialize constraint solve
        thrust::device_event e = thrust::async::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            kernels::FInitializeSolution{
                mPositions.Raw(),
                nextPositions.Raw(),
                mVelocities.Raw(),
                mExternalForces.Raw(),
                mMassInverses.Raw(),
                sdt,
                sdt2});
        // Solve constraints
        for (auto k = 0; k < iterations; ++k)
        {
            // Elastic constraints
            for (common::Buffer<GpuIndex> const& partition : mPartitions)
            {
                e = thrust::async::for_each(
                    thrust::device.after(e),
                    partition.Data(),
                    partition.Data() + partition.Size(),
                    kernels::FStableNeoHookeanConstraint{
                        nextPositions.Raw(),
                        mLagrangeMultipliers[StableNeoHookean].Raw(),
                        T.inds.Raw(),
                        mMassInverses.Raw(),
                        mCompliance[StableNeoHookean].Raw(),
                        mShapeMatrixInverses.Raw(),
                        mRestStableGamma.Raw(),
                        sdt2});
            }
            // Collision constraints
            e = thrust::async::for_each(
                thrust::device.after(e),
                thrust::make_counting_iterator<GpuIndex>(0),
                thrust::make_counting_iterator<GpuIndex>(nVertexTriangleOverlaps),
                kernels::FVertexTriangleContactConstraint{
                    nextPositions.Raw(),
                    mLagrangeMultipliers[Collision].Raw(),
                    overlaps.Raw(),
                    V.inds.Raw(),
                    F.inds.Raw(),
                    mMassInverses.Raw(),
                    mCompliance[Collision].Raw(),
                    mPositions.Raw(),
                    sdt2,
                    mMaxCollisionPenetration * mAverageEdgeLength,
                    mStaticFrictionCoefficient,
                    mDynamicFrictionCoefficient});
        }
        // Update simulation state
        e = thrust::async::for_each(
            thrust::device.after(e),
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nParticles),
            kernels::FUpdateSolution{
                mPositions.Raw(),
                nextPositions.Raw(),
                mVelocities.Raw(),
                sdt});
        e.wait();
    }
}

std::size_t XpbdImpl::NumberOfParticles() const
{
    return X.x.Size();
}

std::size_t XpbdImpl::NumberOfConstraints() const
{
    return T.inds.Size();
}

void XpbdImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin)
{
    auto const nParticles = static_cast<GpuIndex>(X.x.Size());
    if (Xin.rows() != 3 and Xin.cols() != nParticles)
    {
        std::ostringstream ss{};
        ss << "Expected positions of dimensions " << X.x.Dimensions() << "x" << X.x.Size()
           << ", but got " << Xin.rows() << "x" << Xin.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mVelocities.Dimensions(); ++d)
    {
        thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), X.x[d].begin());
        thrust::copy(Xin.row(d).begin(), Xin.row(d).end(), mPositions[d].begin());
    }
}

void XpbdImpl::SetVelocities(Eigen::Ref<GpuMatrixX const> const& vIn)
{
    auto const nParticles = static_cast<GpuIndex>(mVelocities.Size());
    if (vIn.rows() != 3 and vIn.cols() != nParticles)
    {
        std::ostringstream ss{};
        ss << "Expected velocities of dimensions " << mVelocities.Dimensions() << "x"
           << mVelocities.Size() << ", but got " << vIn.rows() << "x" << vIn.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mVelocities.Dimensions(); ++d)
        thrust::copy(vIn.row(d).begin(), vIn.row(d).end(), mVelocities[d].begin());
}

void XpbdImpl::SetExternalForces(Eigen::Ref<GpuMatrixX const> const& fIn)
{
    auto const nParticles = static_cast<GpuIndex>(mExternalForces.Size());
    if (fIn.rows() != 3 and fIn.cols() != nParticles)
    {
        std::ostringstream ss{};
        ss << "Expected forces of dimensions " << mExternalForces.Dimensions() << "x"
           << mExternalForces.Size() << ", but got " << fIn.rows() << "x" << fIn.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mExternalForces.Dimensions(); ++d)
        thrust::copy(fIn.row(d).begin(), fIn.row(d).end(), mExternalForces[d].begin());
}

void XpbdImpl::SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv)
{
    auto const nParticles = static_cast<GpuIndex>(mMassInverses.Size());
    if (not(minv.rows() == 1 and minv.cols() == nParticles) and
        not(minv.rows() == nParticles and minv.cols() == 1))
    {
        std::ostringstream ss{};
        ss << "Expected mass inverses of dimensions " << mMassInverses.Dimensions() << "x"
           << mMassInverses.Size() << " or its transpose, but got " << minv.rows() << "x"
           << minv.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(minv.data(), minv.data() + minv.size(), mMassInverses.Data());
}

void XpbdImpl::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    auto const nTetrahedra = static_cast<GpuIndex>(T.inds.Size());
    if (l.rows() != 2 and l.cols() != nTetrahedra)
    {
        std::ostringstream ss{};
        ss << "Expected Lame coefficients of dimensions 2x" << T.inds.Size() << ", but got "
           << l.rows() << "x" << l.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(l.data(), l.data() + l.size(), mLame.Data());
}

void XpbdImpl::SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint)
{
    if (alpha.size() != mCompliance[eConstraint].Size())
    {
        std::ostringstream ss{};
        ss << "Expected compliance of dimensions " << mCompliance[eConstraint].Size()
           << ", but got " << alpha.size() << "\n";
        throw std::invalid_argument(ss.str());
    }
    thrust::copy(alpha.data(), alpha.data() + alpha.size(), mCompliance[eConstraint].Data());
}

void XpbdImpl::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p][0].resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
}

void XpbdImpl::SetMaxCollisionPenetration(GpuScalar kMaxCollisionPenetration)
{
    mMaxCollisionPenetration = kMaxCollisionPenetration;
}

void XpbdImpl::SetFrictionCoefficients(GpuScalar muS, GpuScalar muK)
{
    mStaticFrictionCoefficient  = muS;
    mDynamicFrictionCoefficient = muK;
}

common::Buffer<GpuScalar, 3> const& XpbdImpl::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& XpbdImpl::GetExternalForce() const
{
    return mExternalForces;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetMassInverse() const
{
    return mMassInverses;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetLameCoefficients() const
{
    return mLame;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetShapeMatrixInverse() const
{
    return mShapeMatrixInverses;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetRestStableGamma() const
{
    return mRestStableGamma;
}

common::Buffer<GpuScalar> const& XpbdImpl::GetLagrangeMultiplier(EConstraint eConstraint) const
{
    return mLagrangeMultipliers[eConstraint];
}

common::Buffer<GpuScalar> const& XpbdImpl::GetCompliance(EConstraint eConstraint) const
{
    return mCompliance[eConstraint];
}

std::vector<common::Buffer<GpuIndex>> const& XpbdImpl::GetPartitions() const
{
    return mPartitions;
}

thrust::host_vector<typename XpbdImpl::OverlapType>
XpbdImpl::GetVertexTriangleOverlapCandidates() const
{
    return SAP.Overlaps();
}

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>

TEST_CASE("[gpu][xpbd] Xpbd")
{
    using namespace pbat;
    // Arrange
    GpuMatrixX V(3, 4);
    GpuIndexMatrixX F(3, 4);
    GpuIndexMatrixX T(4, 1);
    GpuMatrixX lame(2, 1);
    auto constexpr Y        = 1e6;
    auto constexpr nu       = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    lame(0, 0)              = static_cast<GpuScalar>(mu);
    lame(1, 0)              = static_cast<GpuScalar>(lambda);
    // Unit tetrahedron
    // clang-format off
    V << 0.f, 1.f, 0.f, 0.f,
         0.f, 0.f, 1.f, 0.f,
         0.f, 0.f, 0.f, 1.f;
    F << 0, 1, 2, 0,
         1, 2, 0, 2,
         3, 3, 3, 1;
    T << 0, 
         1, 
         2, 
         3;
    // clang-format on
    GpuScalar constexpr tetVolumeExpected = GpuScalar{1.} / GpuScalar{6.};
    GpuMatrixX alphaExpected(2, 1);
    alphaExpected(0, 0)           = GpuScalar{1.} / (tetVolumeExpected * lame(0, 0));
    alphaExpected(1, 0)           = GpuScalar{1.} / (tetVolumeExpected * lame(1, 0));
    GpuScalar const gammaExpected = GpuScalar{1.} + lame(0, 0) / lame(1, 0);
    GpuScalar constexpr zero      = 1e-10f;

    // Act
    using pbat::gpu::xpbd::XpbdImpl;
    auto const nMaxOverlaps = static_cast<std::size_t>(10 * V.cols());
    XpbdImpl xpbd{
        V,
        GpuIndexMatrixX{GpuIndexVectorX::LinSpaced(V.cols(), 0, V.cols() - 1)}.transpose(),
        F,
        T,
        nMaxOverlaps};
    xpbd.SetLameCoefficients(lame);
    xpbd.PrepareConstraints();
    // Assert
    auto const& alphaGpu = xpbd.GetCompliance(XpbdImpl::EConstraint::StableNeoHookean);
    CHECK_EQ(alphaGpu.Size(), 2);
    GpuMatrixX alpha = common::ToEigen(alphaGpu.Get()).reshaped(2, alphaGpu.Size() / 2);
    CHECK(alpha.isApprox(alphaExpected, zero));
    auto const& DmInvGpu = xpbd.GetShapeMatrixInverse();
    CHECK_EQ(DmInvGpu.Size(), 9);
    GpuMatrixX DmInv = common::ToEigen(DmInvGpu.Get()).reshaped(3, DmInvGpu.Size() / 3);
    CHECK(DmInv.isApprox(GpuMatrixX::Identity(3, 3), zero));
    auto const& gammaGpu = xpbd.GetRestStableGamma();
    CHECK_EQ(gammaGpu.Size(), 1);
    GpuVectorX gamma = common::ToEigen(gammaGpu.Get());
    CHECK_LE(std::abs(gamma(0) - gammaExpected), zero);
}