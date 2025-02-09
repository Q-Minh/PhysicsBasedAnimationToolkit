// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Integrator.cuh"
#include "Kernels.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/gpu/impl/common/Cuda.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"

#include <cuda/api.hpp>
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
      mActiveSetUpdateFrequency(static_cast<GpuIndex>(data.mActiveSetUpdateFrequency)),
      mPositionsAtT(data.xt.cols()),
      mInertialTargetPositions(data.xtilde.cols()),
      mChebyshevPositionsM2(data.xchebm2.cols()),
      mChebyshevPositionsM1(data.xchebm1.cols()),
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

    GpuScalar sdt                        = dt / static_cast<GpuScalar>(substeps);
    GpuScalar sdt2                       = sdt * sdt;
    GpuIndex const nVertices             = static_cast<GpuIndex>(x.Size());
    bool const bUseChebyshevAcceleration = rho > GpuScalar{0} and rho < GpuScalar{1};

    // Initialize active set
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

    kernels::BackwardEulerMinimization bdf{};
    bdf.dt        = sdt;
    bdf.dt2       = sdt2;
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

    for (auto s = 0; s < substeps; ++s)
    {
        using pbat::math::linalg::mini::FromBuffers;
        using pbat::math::linalg::mini::FromFlatBuffer;
        using pbat::math::linalg::mini::ToBuffers;
        using pbat::math::linalg::mini::ToFlatBuffer;
        // Compute inertial target positions
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
                auto y = InertialTarget(
                    FromBuffers<3, 1>(xt, i),
                    FromBuffers<3, 1>(vt, i),
                    FromBuffers<3, 1>(aext, i),
                    dt,
                    dt2);
                ToBuffers(y, xtilde, i);
            });
        // Initialize block coordinate descent's, i.e. BCD's, solution
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
        // Update active set
        if (s % mActiveSetUpdateFrequency == 0)
        {
            cd.UpdateActiveSet(x);
            static auto constexpr kMaxContacts =
                kernels::BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
            static auto constexpr kMaxNeighbours =
                contact::VertexTriangleMixedCcdDcd::kMaxNeighbours;
            thrust::for_each(
                thrust::device,
                cd.av.Data(),
                cd.av.Data() + cd.nActive,
                [V = cd.V.Raw(), nn = cd.nn.Raw(), fc = fc.Raw()] PBAT_DEVICE(GpuIndex v) {
                    using namespace pbat::math::linalg::mini;
                    GpuIndex i = V[v];
                    SVector<GpuIndex, kMaxNeighbours> nnv =
                        FromFlatBuffer<kMaxNeighbours, 1>(nn, v);
                    SVector<GpuIndex, kMaxContacts> f = -Ones<GpuIndex, kMaxContacts, 1>();
                    auto const top                    = min(kMaxContacts, kMaxNeighbours);
                    for (auto c = 0; c < top; ++c)
                        if (nnv(c) >= 0)
                            f(c) = nnv(c);
                    ToFlatBuffer(f, fc, i);
                });
        }
        // Initialize Chebyshev semi-iterative method
        GpuScalar rho2 = rho * rho;
        GpuScalar omega{};
        // Minimize Backward Euler, i.e. BDF1, objective
        for (auto k = 0; k < iterations; ++k)
        {
            using pbat::sim::vbd::kernels::ChebyshevOmega;
            if (bUseChebyshevAcceleration)
                omega = ChebyshevOmega(k, rho2, omega);

            auto const nPartitions = mPptr.size() - 1;
            for (auto p = 0; p < nPartitions; ++p)
            {
                auto pBegin   = mPptr[p];
                auto pEnd     = mPptr[p + 1];
                bdf.partition = mPadj.Raw() + pBegin;
                auto const nVerticesInPartition =
                    static_cast<cuda::grid::dimension_t>(pEnd - pBegin);
                pbat::common::ForValues<32, 64, 128, 256>([&]<auto kBlockThreads>() {
                    if (mGpuThreadBlockSize > kBlockThreads / 2 and
                        mGpuThreadBlockSize <= kBlockThreads)
                    {
                        auto const kDynamicSharedMemoryCapacity =
                            static_cast<cuda::memory::shared::size_t>(
                                sizeof(typename kernels::BackwardEulerMinimization::BlockStorage<
                                       kBlockThreads>));
                        auto bcdLaunchConfiguration =
                            cuda::launch_config_builder()
                                .block_size(kBlockThreads)
                                .dynamic_shared_memory_size(kDynamicSharedMemoryCapacity)
                                .grid_size(nVerticesInPartition)
                                .build();
                        cuda::device::current::get().launch(
                            kernels::MinimizeBackwardEuler<kBlockThreads>,
                            bcdLaunchConfiguration,
                            bdf);
                    }
                });
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

            if (bUseChebyshevAcceleration)
            {
                thrust::for_each(
                    thrust::device,
                    thrust::make_counting_iterator<GpuIndex>(0),
                    thrust::make_counting_iterator<GpuIndex>(nVertices),
                    [k     = k,
                     omega = omega,
                     xkm2  = mChebyshevPositionsM2.Raw(),
                     xkm1  = mChebyshevPositionsM1.Raw(),
                     xk    = x.Raw()] PBAT_DEVICE(auto i) {
                        using pbat::sim::vbd::kernels::ChebyshevUpdate;
                        auto xkm2i = FromBuffers<3, 1>(xkm2, i);
                        auto xkm1i = FromBuffers<3, 1>(xkm1, i);
                        auto xki   = FromBuffers<3, 1>(xk, i);
                        ChebyshevUpdate(k, omega, xkm2i, xkm1i, xki);
                        ToBuffers(xkm2i, xkm2, i);
                        ToBuffers(xkm1i, xkm1, i);
                        ToBuffers(xki, xk, i);
                    });
            }
        }
        // Update velocities
        mVelocitiesAtT = mVelocities;
        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<GpuIndex>(0),
            thrust::make_counting_iterator<GpuIndex>(nVertices),
            [xt = mPositionsAtT.Raw(), x = x.Raw(), v = mVelocities.Raw(), dt = sdt] PBAT_DEVICE(
                auto i) {
                using pbat::sim::vbd::kernels::IntegrateVelocity;
                auto vtp1 =
                    IntegrateVelocity(FromBuffers<3, 1>(xt, i), FromBuffers<3, 1>(x, i), dt);
                ToBuffers(vtp1, v, i);
            });
        // Store past positions
        mPositionsAtT = x;
    }
    cd.FinalizeActiveSet(x);

    PBAT_PROFILE_CUDA_HOST_SCOPE_END(ctx);
}

void Integrator::SetPositions(Eigen::Ref<GpuMatrixX const> const& Xin)
{
    common::ToBuffer(Xin, x);
    mPositionsAtT = x;
}

void Integrator::SetVelocities(Eigen::Ref<GpuMatrixX const> const& v)
{
    common::ToBuffer(v, mVelocities);
    mVelocitiesAtT = mVelocities;
}

void Integrator::SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext)
{
    common::ToBuffer(aext, mExternalAcceleration);
}

void Integrator::SetMass(Eigen::Ref<GpuVectorX const> const& m)
{
    common::ToBuffer(m, mMass);
}

void Integrator::SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg)
{
    common::ToBuffer(wg, mQuadratureWeights);
}

void Integrator::SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP)
{
    common::ToBuffer(GP, mShapeFunctionGradients);
}

void Integrator::SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l)
{
    common::ToBuffer(l, mLameCoefficients);
}

void Integrator::SetNumericalZeroForHessianDeterminant(GpuScalar zero)
{
    mDetHZero = zero;
}

void Integrator::SetVertexTetrahedronAdjacencyList(
    Eigen::Ref<GpuIndexVectorX const> const& GVTp,
    Eigen::Ref<GpuIndexVectorX const> const& GVTn,
    Eigen::Ref<GpuIndexVectorX const> const& GVTilocal)
{
    if (GVTn.size() != GVTilocal.size())
    {
        std::ostringstream ss{};
        ss << "Expected vertex-tetrahedron adjacency graph's neighbour array and data (ilocal) "
              "array to have the same size, but got neighbours="
           << GVTn.size() << ", ilocal=" << GVTilocal.size() << " \n";
        throw std::invalid_argument(ss.str());
    }

    common::ToBuffer(GVTp, mVertexTetrahedronPrefix);
    mVertexTetrahedronNeighbours.Resize(GVTn.size());
    mVertexTetrahedronLocalVertexIndices.Resize(GVTilocal.size());
    common::ToBuffer(GVTn, mVertexTetrahedronNeighbours);
    common::ToBuffer(GVTilocal, mVertexTetrahedronLocalVertexIndices);
}

void Integrator::SetRayleighDampingCoefficient(GpuScalar kD)
{
    mRayleighDamping = kD;
}

void Integrator::SetVertexPartitions(
    Eigen::Ref<GpuIndexVectorX const> const& Pptr,
    Eigen::Ref<GpuIndexVectorX const> const& Padj)
{
    mPptr = Pptr;
    common::ToBuffer(Padj, mPadj);
}

void Integrator::SetInitializationStrategy(EInitializationStrategy strategy)
{
    mInitializationStrategy = strategy;
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

common::Buffer<GpuScalar, 3> const& Integrator::GetVelocity() const
{
    return mVelocities;
}

common::Buffer<GpuScalar, 3> const& Integrator::GetExternalAcceleration() const
{
    return mExternalAcceleration;
}

common::Buffer<GpuScalar> const& Integrator::GetMass() const
{
    return mMass;
}

common::Buffer<GpuScalar> const& Integrator::GetShapeFunctionGradients() const
{
    return mShapeFunctionGradients;
}

common::Buffer<GpuScalar> const& Integrator::GetLameCoefficients() const
{
    return mLameCoefficients;
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
