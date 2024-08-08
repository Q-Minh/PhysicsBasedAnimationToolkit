// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "XpbdImpl.cuh"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>
#include <exception>
#include <sstream>
#include <string>
#include <thrust/async/for_each.h>
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

struct FProjectConstraint
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
    Eigen::Ref<GpuMatrixX const> const& Vin,
    Eigen::Ref<GpuIndexMatrixX const> const& Fin,
    Eigen::Ref<GpuIndexMatrixX const> const& Tin)
    : V(Vin),
      F(Fin),
      T(Tin),
      mPositions(Vin.cols()),
      mVelocities(Vin.cols()),
      mExternalForces(Vin.cols()),
      mMassInverses(Vin.cols()),
      mLame(2 * Tin.cols()),
      mShapeMatrixInverses(9 * Tin.cols()),
      mRestStableGamma(Tin.cols()),
      mLagrangeMultipliers(),
      mCompliance(),
      mPartitions(),
      muf{0.5}
{
    mLagrangeMultipliers[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mLagrangeMultipliers[Collision].Resize(V.NumberOfPoints());
    mCompliance[StableNeoHookean].Resize(2 * T.NumberOfSimplices());
    mCompliance[Collision].Resize(V.NumberOfPoints());
    // Initialize particle data
    for (auto d = 0; d < V.Dimensions(); ++d)
    {
        thrust::copy(V.x[d].begin(), V.x[d].end(), mPositions[d].begin());
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
            V.x.Raw(),
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
    // TODO: Detect collision candidates and setup collision constraint solve
    // ...

    auto& nextPositions = V.x;
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
                    kernels::FProjectConstraint{
                        nextPositions.Raw(),
                        thrust::raw_pointer_cast(mLagrangeMultipliers[StableNeoHookean].Data()),
                        T.inds.Raw(),
                        mMassInverses.Raw(),
                        thrust::raw_pointer_cast(mCompliance[StableNeoHookean].Data()),
                        mShapeMatrixInverses.Raw(),
                        mRestStableGamma.Raw(),
                        sdt2});
            }
            // TODO: Collision constraints
            // ...
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
    return V.x.Size();
}

std::size_t XpbdImpl::NumberOfConstraints() const
{
    return T.inds.Size();
}

void XpbdImpl::SetPositions(Eigen::Ref<GpuMatrixX const> const& X)
{
    auto const nParticles = static_cast<GpuIndex>(V.x.Size());
    if (X.rows() != 3 and X.cols() != nParticles)
    {
        std::ostringstream ss{};
        ss << "Expected positions of dimensions " << V.x.Dimensions() << "x" << V.x.Size()
           << ", but got " << X.rows() << "x" << X.cols() << "\n";
        throw std::invalid_argument(ss.str());
    }
    for (auto d = 0; d < mVelocities.Dimensions(); ++d)
        thrust::copy(X.row(d).begin(), X.row(d).end(), V.x[d].begin());
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

void XpbdImpl::SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions)
{
    mPartitions.resize(partitions.size());
    for (auto p = 0; p < partitions.size(); ++p)
    {
        mPartitions[p][0].resize(partitions[p].size());
        thrust::copy(partitions[p].begin(), partitions[p].end(), mPartitions[p].Data());
    }
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
    XpbdImpl xpbd{V, F, T};
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