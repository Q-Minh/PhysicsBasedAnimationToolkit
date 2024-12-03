// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Kernels.cuh"
#include "pbat/HostDevice.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/vbd/Kernels.h"

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
    using namespace pbat::math::linalg::mini;

    // Get vertex-tet adjacency information
    GpuIndex i                 = BDF.partition[bid]; // Vertex index
    GpuIndex GVTbegin          = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - GVTbegin;
    // 1. Compute vertex-element elastic energy derivatives w.r.t. i and store them in shared memory
    auto Hgt = FromFlatBuffer<3, 4>(shared, tid);
    Hgt.SetZero();
    auto Ht = Hgt.Slice<3, 3>(0, 0);
    auto gt = Hgt.Col(3);
    for (auto elocal = tid; elocal < nAdjacentElements; elocal += nThreadsPerBlock)
    {
        GpuIndex e                  = BDF.GVTn[GVTbegin + elocal];
        GpuIndex ilocal             = BDF.GVTilocal[GVTbegin + elocal];
        auto Te                     = FromBuffers<4, 1>(BDF.T, e);
        auto GPe                    = FromFlatBuffer<4, 3>(BDF.GP, e);
        auto xe                     = FromBuffers(BDF.x, Te.Transpose());
        auto lamee                  = FromFlatBuffer<2, 1>(BDF.lame, e);
        auto wg                     = BDF.wg[e];
        SMatrix<GpuScalar, 3, 3> Fe = xe * GPe;
        pbat::physics::StableNeoHookeanEnergy<3> Psi{};
        SVector<GpuScalar, 9> gF;
        SMatrix<GpuScalar, 9, 9> HF;
        Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        using pbat::sim::vbd::kernels::AccumulateElasticGradient;
        using pbat::sim::vbd::kernels::AccumulateElasticHessian;
        AccumulateElasticHessian(ilocal, wg, GPe, HF, Ht);
        AccumulateElasticGradient(ilocal, wg, GPe, gF, gt);
    }
    __syncthreads();

    // 2. Compute total vertex hessian and gradient via parallel reduction
    SMatrix<GpuScalar, 3, 3> Hi = Zeros<GpuScalar, 3, 3>();
    SVector<GpuScalar, 3> gi    = Zeros<GpuScalar, 3, 1>();
    auto nActiveThreads         = min(nAdjacentElements, nThreadsPerBlock);
    if (BDF.bUseParallelReduction)
    {
        do
        {
            ++nActiveThreads >>= 1;
            // When nActiveThreads hits 1, every other thread will have exited
            if (tid >= nActiveThreads)
                return;
            auto rHgt = FromFlatBuffer<3, 4>(shared, tid + nActiveThreads);
            Hgt += rHgt;
            rHgt.SetZero();
            // __syncthreads();
            // If nActiveThreads is not 1, i.e. it is > 1, then we preserve its value,
            // otherwise, nActiveThreads becomes 0 so that we can exit the loop.
            nActiveThreads *= (nActiveThreads > 1);
        } while (nActiveThreads > 0);
        Hi = Ht;
        gi = gt;
    }
    else
    {
        if (tid > 0)
            return;
        for (auto j = 0; j < nActiveThreads; ++j)
        {
            auto Hgj = FromFlatBuffer<3, 4>(shared, j);
            Hi += Hgj.Slice<3, 3>(0, 0);
            gi += Hgj.Col(3);
        }
    }

    // 3. Time integrate vertex position
    GpuScalar mi                  = BDF.m[i];
    SVector<GpuScalar, 3> xti     = FromBuffers<3, 1>(BDF.xt, i);
    SVector<GpuScalar, 3> xitilde = FromBuffers<3, 1>(BDF.xtilde, i);
    SVector<GpuScalar, 3> xi      = FromBuffers<3, 1>(BDF.x, i);
    using pbat::sim::vbd::kernels::AddDamping;
    using pbat::sim::vbd::kernels::AddInertiaDerivatives;
    AddDamping(BDF.dt, xti, xi, BDF.kD, gi, Hi);
    AddInertiaDerivatives(BDF.dt2, mi, xitilde, xi, gi, Hi);
    // 4. Integrate positions
    using pbat::sim::vbd::kernels::IntegratePositions;
    IntegratePositions(gi, Hi, xi, BDF.detHZero);
    ToBuffers(xi, BDF.x, i);
};

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat