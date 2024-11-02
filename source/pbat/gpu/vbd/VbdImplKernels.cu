// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VbdImplKernels.cuh"
#include "pbat/HostDevice.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

namespace pbat {
namespace gpu {
namespace vbd {
namespace kernels {

__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF)
{
    // Get thread info
    extern __shared__ GpuScalar shared[];
    auto tid              = threadIdx.x;
    auto bid              = blockIdx.x;
    auto nThreadsPerBlock = blockDim.x;
    using namespace mini;

    // Vertex index
    GpuIndex i = BDF.partition[bid];
    // Get vertex-tet adjacency information
    GpuIndex GVTbegin          = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - GVTbegin;
    GpuScalar* Hge             = shared + tid * BDF.ExpectedSharedMemoryPerThreadInScalars();
    memset(Hge, 0, BDF.ExpectedSharedMemoryPerThreadInBytes());
    // 1. Compute element elastic energy derivatives w.r.t. i and store them in shared memory
    for (auto elocal = tid; elocal < nAdjacentElements; elocal += nThreadsPerBlock)
    {
        GpuIndex e      = BDF.GVTn[GVTbegin + elocal];
        GpuIndex ilocal = BDF.GVTilocal[GVTbegin + elocal];
        // Each element has a 3x3 hessian + 3x1 gradient = 12 scalars/element in shared memory
        BDF.ComputeStableNeoHookeanDerivatives(e, ilocal, Hge);
    }
    __syncthreads();

    // Remaining execution is synchronous, i.e. only 1 thread is required
    if (tid > 0)
        return;

    // 2. Accumulate results into vertex hessian and gradient
    SVector<GpuScalar, 3> xti     = FromBuffers<3, 1>(BDF.xt, i);
    SVector<GpuScalar, 3> xitilde = FromBuffers<3, 1>(BDF.xtilde, i);
    SVector<GpuScalar, 3> xi      = FromBuffers<3, 1>(BDF.x, i);
    SMatrix<GpuScalar, 3, 3> Hi   = Zeros<GpuScalar, 3, 3>{};
    SVector<GpuScalar, 3> gi      = Zeros<GpuScalar, 3, 1>{};
    // Add elastic energy derivatives
    auto const nActiveThreads = min(nAdjacentElements, nThreadsPerBlock);
    for (auto j = 0; j < nActiveThreads; ++j)
    {
        GpuScalar* HiShared = shared + j * BDF.ExpectedSharedMemoryPerThreadInScalars();
        GpuScalar* giShared = HiShared + BDF.SharedGradientOffset();
        SMatrixView<GpuScalar, 3, 3> Hei(HiShared);
        SMatrixView<GpuScalar, 3, 1> gei(giShared);
        Hi += Hei;
        gi += gei;
    }
    // Add Rayleigh damping terms
    GpuScalar const D = BDF.kD / BDF.dt;
    gi += D * (Hi * (xi - xti));
    Hi *= GpuScalar{1} + D;
    // Add inertial energy derivatives
    GpuScalar const K = BDF.m[i] / BDF.dt2;
    Diag(Hi) += K;
    gi += K * (xi - xitilde);

    // 3. Newton step
    if (abs(Determinant(Hi)) <= BDF.detHZero) // Skip nearly rank-deficient hessian
        return;

    xi = xi - (Inverse(Hi) * gi);

    // 4. Commit vertex descent step
    ToBuffers(xi, BDF.x, i);
};

PBAT_DEVICE void BackwardEulerMinimization::ComputeStableNeoHookeanDerivatives(
    GpuIndex e,
    GpuIndex ilocal,
    GpuScalar* Hge) const
{
    using namespace mini;
    GpuScalar wge                  = wg[e];
    SMatrix<GpuScalar, 2, 1> lamee = FromFlatBuffer<2, 1>(lame, e);
    // Compute (d^k Psi / dF^k)
    SMatrix<GpuIndex, 4, 1> v    = FromBuffers<4, 1>(T, e);
    SMatrix<GpuScalar, 3, 4> xe  = FromBuffers(x, v.Transpose());
    SMatrix<GpuScalar, 4, 3> GPe = FromFlatBuffer<4, 3>(GP, e);
    SMatrix<GpuScalar, 3, 3> Fe  = xe * GPe;
    physics::StableNeoHookeanEnergy<3> Psi{};
    SVector<GpuScalar, 9> gF;
    SMatrix<GpuScalar, 9, 9> HF;
    Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
    // Write vertex-specific derivatives into output memory HGe
    SMatrixView<GpuScalar, 3, 4> HGei(Hge);
    auto Hi = HGei.Slice<3, 3>(0, 0);
    auto gi = HGei.Col(3);
    // Contract (d^k Psi / dF^k) with (d F / dx)^k. See pbat/fem/DeformationGradient.h.
    for (auto kj = 0; kj < 3; ++kj)
        for (auto ki = 0; ki < 3; ++ki)
            Hi += wge * GPe(ilocal, ki) * GPe(ilocal, kj) * HF.Slice<3, 3>(ki * 3, kj * 3);
    for (auto k = 0; k < 3; ++k)
        gi += wge * GPe(ilocal, k) * gF.Slice<3, 1>(k * 3, 0);
}

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat