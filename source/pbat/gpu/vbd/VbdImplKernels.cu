// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VbdImplKernels.cuh"

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
    using namespace pbat::gpu::math::linalg;
    Matrix<GpuScalar, 3> xti     = BDF.ToLocal(i, BDF.xt);
    Matrix<GpuScalar, 3> xitilde = BDF.ToLocal(i, BDF.xtilde);
    Matrix<GpuScalar, 3> xi      = BDF.ToLocal(i, BDF.x);
    Matrix<GpuScalar, 3, 3> Hi   = Zeros<GpuScalar, 3, 3>{};
    Matrix<GpuScalar, 3, 1> gi   = Zeros<GpuScalar, 3, 1>{};
    // Add elastic energy derivatives
    auto const nActiveThreads = min(nAdjacentElements, nThreadsPerBlock);
    for (auto j = 0; j < nActiveThreads; ++j)
    {
        GpuScalar* HiShared = shared + j * BDF.ExpectedSharedMemoryPerThreadInScalars();
        GpuScalar* giShared = HiShared + BDF.SharedGradientOffset();
        MatrixView<GpuScalar, 3, 3> Hei(HiShared);
        MatrixView<GpuScalar, 3, 1> gei(giShared);
        Hi += Hei;
        gi += gei;
    }
    // Add Rayleigh damping terms
    GpuScalar const D = BDF.kD / BDF.dt;
    gi += D * (Hi * (xi - xti));
    Hi *= GpuScalar{1} + D;
    // Add inertial energy derivatives
    GpuScalar const K = BDF.m[i] / BDF.dt2;
    Hi(0, 0) += K;
    Hi(1, 1) += K;
    Hi(2, 2) += K;
    gi += K * (xi - xitilde);

    // 3. Newton step
    if (abs(Determinant(Hi)) <= BDF.detHZero) // Skip nearly rank-deficient hessian
        return;
    xi = xi - (Inverse(Hi) * gi);

    // 4. Commit vertex descent step
    BDF.ToGlobal(i, xi, BDF.x);
};

__device__ BackwardEulerMinimization::Matrix<GpuScalar, 4, 3>
BackwardEulerMinimization::BasisFunctionGradients(GpuIndex e) const
{
    using namespace pbat::gpu::math::linalg;
    Matrix<GpuScalar, 4, 3> GPlocal = MatrixView<GpuScalar, 4, 3>(GP + e * 12);
    return GPlocal;
}

__device__ BackwardEulerMinimization::Matrix<GpuScalar, 3, 4>
BackwardEulerMinimization::ElementVertexPositions(GpuIndex e) const
{
    Matrix<GpuScalar, 3, 4> xe;
    for (auto i = 0; i < 4; ++i)
    {
        GpuIndex vi = T[i][e];
        xe.Col(i)   = ToLocal(vi, x);
    }
    return xe;
}

__device__ BackwardEulerMinimization::Matrix<GpuScalar, 9, 10>
BackwardEulerMinimization::StableNeoHookeanDerivativesWrtF(
    GpuIndex e,
    Matrix<GpuScalar, 3, 3> const& Fe,
    GpuScalar mu,
    GpuScalar lambda) const
{
    using namespace pbat::gpu::math::linalg;
    Matrix<GpuScalar, 9, 10> HGe;
    auto He = HGe.Slice<9, 9>(0, 0);
    auto ge = HGe.Col(9);

    // Auto-generated from pbat/physics/StableNeoHookeanEnergy.h
    GpuScalar const a0  = Fe(4) * Fe(8);
    GpuScalar const a1  = Fe(5) * Fe(7);
    GpuScalar const a2  = 2 * a0 - 2 * a1;
    GpuScalar const a3  = Fe(3) * Fe(8);
    GpuScalar const a4  = Fe(4) * Fe(6);
    GpuScalar const a5  = lambda * (-a1 * Fe(0) - a3 * Fe(1) - a4 * Fe(2) + Fe(0) * Fe(4) * Fe(8) +
                                   Fe(1) * Fe(5) * Fe(6) + Fe(2) * Fe(3) * Fe(7) - 1 - mu / lambda);
    GpuScalar const a6  = (1.0 / 2.0) * a5;
    GpuScalar const a7  = -2 * a3 + 2 * Fe(5) * Fe(6);
    GpuScalar const a8  = Fe(3) * Fe(7);
    GpuScalar const a9  = -2 * a4 + 2 * a8;
    GpuScalar const a10 = Fe(1) * Fe(8);
    GpuScalar const a11 = -2 * a10 + 2 * Fe(2) * Fe(7);
    GpuScalar const a12 = Fe(0) * Fe(8);
    GpuScalar const a13 = Fe(2) * Fe(6);
    GpuScalar const a14 = 2 * a12 - 2 * a13;
    GpuScalar const a15 = Fe(0) * Fe(7);
    GpuScalar const a16 = -2 * a15 + 2 * Fe(1) * Fe(6);
    GpuScalar const a17 = Fe(1) * Fe(5);
    GpuScalar const a18 = Fe(2) * Fe(4);
    GpuScalar const a19 = 2 * a17 - 2 * a18;
    GpuScalar const a20 = Fe(0) * Fe(5);
    GpuScalar const a21 = -2 * a20 + 2 * Fe(2) * Fe(3);
    GpuScalar const a22 = Fe(0) * Fe(4);
    GpuScalar const a23 = Fe(1) * Fe(3);
    GpuScalar const a24 = 2 * a22 - 2 * a23;
    GpuScalar const a25 = (1.0 / 2.0) * lambda;
    GpuScalar const a26 = a25 * (a0 - a1);
    GpuScalar const a27 = a5 * Fe(8);
    GpuScalar const a28 = a5 * Fe(7);
    GpuScalar const a29 = -a28;
    GpuScalar const a30 = a5 * Fe(5);
    GpuScalar const a31 = -a30;
    GpuScalar const a32 = a5 * Fe(4);
    GpuScalar const a33 = a25 * (-a3 + Fe(5) * Fe(6));
    GpuScalar const a34 = -a27;
    GpuScalar const a35 = a5 * Fe(6);
    GpuScalar const a36 = a5 * Fe(3);
    GpuScalar const a37 = -a36;
    GpuScalar const a38 = a25 * (-a4 + a8);
    GpuScalar const a39 = -a35;
    GpuScalar const a40 = -a32;
    GpuScalar const a41 = a25 * (-a10 + Fe(2) * Fe(7));
    GpuScalar const a42 = a5 * Fe(2);
    GpuScalar const a43 = a5 * Fe(1);
    GpuScalar const a44 = -a43;
    GpuScalar const a45 = a25 * (a12 - a13);
    GpuScalar const a46 = -a42;
    GpuScalar const a47 = a5 * Fe(0);
    GpuScalar const a48 = a25 * (-a15 + Fe(1) * Fe(6));
    GpuScalar const a49 = -a47;
    GpuScalar const a50 = a25 * (a17 - a18);
    GpuScalar const a51 = a25 * (-a20 + Fe(2) * Fe(3));
    GpuScalar const a52 = a25 * (a22 - a23);
    ge(0)               = a2 * a6 + mu * Fe(0);
    ge(1)               = a6 * a7 + mu * Fe(1);
    ge(2)               = a6 * a9 + mu * Fe(2);
    ge(3)               = a11 * a6 + mu * Fe(3);
    ge(4)               = a14 * a6 + mu * Fe(4);
    ge(5)               = a16 * a6 + mu * Fe(5);
    ge(6)               = a19 * a6 + mu * Fe(6);
    ge(7)               = a21 * a6 + mu * Fe(7);
    ge(8)               = a24 * a6 + mu * Fe(8);
    He(0)               = a2 * a26 + mu;
    He(1)               = a26 * a7;
    He(2)               = a26 * a9;
    He(3)               = a11 * a26;
    He(4)               = a14 * a26 + a27;
    He(5)               = a16 * a26 + a29;
    He(6)               = a19 * a26;
    He(7)               = a21 * a26 + a31;
    He(8)               = a24 * a26 + a32;
    He(9)               = a2 * a33;
    He(10)              = a33 * a7 + mu;
    He(11)              = a33 * a9;
    He(12)              = a11 * a33 + a34;
    He(13)              = a14 * a33;
    He(14)              = a16 * a33 + a35;
    He(15)              = a19 * a33 + a30;
    He(16)              = a21 * a33;
    He(17)              = a24 * a33 + a37;
    He(18)              = a2 * a38;
    He(19)              = a38 * a7;
    He(20)              = a38 * a9 + mu;
    He(21)              = a11 * a38 + a28;
    He(22)              = a14 * a38 + a39;
    He(23)              = a16 * a38;
    He(24)              = a19 * a38 + a40;
    He(25)              = a21 * a38 + a36;
    He(26)              = a24 * a38;
    He(27)              = a2 * a41;
    He(28)              = a34 + a41 * a7;
    He(29)              = a28 + a41 * a9;
    He(30)              = a11 * a41 + mu;
    He(31)              = a14 * a41;
    He(32)              = a16 * a41;
    He(33)              = a19 * a41;
    He(34)              = a21 * a41 + a42;
    He(35)              = a24 * a41 + a44;
    He(36)              = a2 * a45 + a27;
    He(37)              = a45 * a7;
    He(38)              = a39 + a45 * a9;
    He(39)              = a11 * a45;
    He(40)              = a14 * a45 + mu;
    He(41)              = a16 * a45;
    He(42)              = a19 * a45 + a46;
    He(43)              = a21 * a45;
    He(44)              = a24 * a45 + a47;
    He(45)              = a2 * a48 + a29;
    He(46)              = a35 + a48 * a7;
    He(47)              = a48 * a9;
    He(48)              = a11 * a48;
    He(49)              = a14 * a48;
    He(50)              = a16 * a48 + mu;
    He(51)              = a19 * a48 + a43;
    He(52)              = a21 * a48 + a49;
    He(53)              = a24 * a48;
    He(54)              = a2 * a50;
    He(55)              = a30 + a50 * a7;
    He(56)              = a40 + a50 * a9;
    He(57)              = a11 * a50;
    He(58)              = a14 * a50 + a46;
    He(59)              = a16 * a50 + a43;
    He(60)              = a19 * a50 + mu;
    He(61)              = a21 * a50;
    He(62)              = a24 * a50;
    He(63)              = a2 * a51 + a31;
    He(64)              = a51 * a7;
    He(65)              = a36 + a51 * a9;
    He(66)              = a11 * a51 + a42;
    He(67)              = a14 * a51;
    He(68)              = a16 * a51 + a49;
    He(69)              = a19 * a51;
    He(70)              = a21 * a51 + mu;
    He(71)              = a24 * a51;
    He(72)              = a2 * a52 + a32;
    He(73)              = a37 + a52 * a7;
    He(74)              = a52 * a9;
    He(75)              = a11 * a52 + a44;
    He(76)              = a14 * a52 + a47;
    He(77)              = a16 * a52;
    He(78)              = a19 * a52;
    He(79)              = a21 * a52;
    He(80)              = a24 * a52 + mu;
    return HGe;
}

__device__ void BackwardEulerMinimization::ComputeStableNeoHookeanDerivatives(
    GpuIndex e,
    GpuIndex ilocal,
    GpuScalar* Hge) const
{
    using namespace pbat::gpu::math::linalg;
    GpuScalar wge                 = wg[e];
    Matrix<GpuScalar, 2, 1> lamee = MatrixView<GpuScalar, 2, 1>{lame + 2 * e};
    // Compute (d^k Psi / dF^k)
    Matrix<GpuScalar, 3, 4> xe   = ElementVertexPositions(e);
    Matrix<GpuScalar, 4, 3> GPe  = BasisFunctionGradients(e);
    Matrix<GpuScalar, 3, 3> Fe   = xe * GPe;
    Matrix<GpuScalar, 9, 10> HGF = StableNeoHookeanDerivativesWrtF(e, Fe, lamee(0), lamee(1));
    auto HF                      = HGF.Slice<9, 9>(0, 0);
    auto gF                      = HGF.Col(9);
    // Write vertex-specific derivatives into output memory HGe
    MatrixView<GpuScalar, 3, 4> HGei(Hge);
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