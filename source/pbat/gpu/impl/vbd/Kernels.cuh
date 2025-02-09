#ifndef PBAT_GPU_IMPL_VBD_KERNELS_H
#define PBAT_GPU_IMPL_VBD_KERNELS_H

#include "pbat/HostDevice.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/vbd/Kernels.h"

#include <array>
#include <cstddef>
#include <cub/block/block_reduce.cuh>
#include <limits>

namespace pbat {
namespace gpu {
namespace impl {
namespace vbd {
namespace kernels {

using namespace pbat::math::linalg::mini;

struct BackwardEulerMinimization
{
    GpuScalar dt;                     ///< Time step
    GpuScalar dt2;                    ///< Squared time step
    GpuScalar* m;                     ///< Lumped mass matrix
    std::array<GpuScalar*, 3> xtilde; ///< Inertial target
    std::array<GpuScalar*, 3> xt;     ///< Previous vertex positions
    std::array<GpuScalar*, 3> x;      ///< Vertex positions
    std::array<GpuScalar*, 3> xb;     ///< Vertex position write buffer

    std::array<GpuIndex*, 4> T; ///< 4x|#elements| array of tetrahedra
    GpuScalar* wg;              ///< |#elements| array of quadrature weights
    GpuScalar* GP;              ///< 4x3x|#elements| array of shape function gradients
    GpuScalar* lame;            ///< 2x|#elements| of 1st and 2nd Lame coefficients
    GpuScalar detHZero;         ///< Numerical zero for hessian determinant check
    // GpuScalar const* kD;                  ///< |#elements| array of damping coefficients

    GpuIndex* GVTp;      ///< Vertex-tetrahedron adjacency list's prefix sum
    GpuIndex* GVTn;      ///< Vertex-tetrahedron adjacency list's neighbour list
    GpuIndex* GVTilocal; ///< Vertex-tetrahedron adjacency list's ilocal property

    GpuScalar kD;   ///< Rayleigh damping coefficient
    GpuScalar muC;  ///< Collision penalty
    GpuScalar muF;  ///< Coefficient of friction
    GpuScalar epsv; ///< IPC smooth friction transition function's relative velocity threshold
    static auto constexpr kMaxCollidingTrianglesPerVertex = 8;
    GpuIndex* fc;               ///< |#vertices|x|kMaxCollidingTrianglesPerVertex| array of
                                ///< colliding triangles
    std::array<GpuIndex*, 3> F; ///< 3x|#collision triangles| array of triangles
    GpuScalar* XVA;             ///< |#vertices| array of vertex areas

    GpuIndex*
        partition; ///< List of vertex indices that can be processed independently, i.e. in parallel

    using ElasticDerivativeStorageType = SMatrix<GpuScalar, 3, 4>;
    template <auto kBlockThreads>
    using BlockReduce = cub::BlockReduce<ElasticDerivativeStorageType, kBlockThreads>;
    template <auto kBlockThreads>
    using BlockStorage = typename BlockReduce<kBlockThreads>::TempStorage;
};

template <auto kBlockThreads>
__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF)
{
    // Get thread info
    using BlockReduce  = typename BackwardEulerMinimization::BlockReduce<kBlockThreads>;
    using BlockStorage = typename BackwardEulerMinimization::BlockStorage<kBlockThreads>;
    extern __shared__ __align__(alignof(BlockStorage)) char shared[];
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    // Get vertex-tet adjacency information
    GpuIndex i                 = BDF.partition[bid]; // Vertex index
    GpuIndex GVTbegin          = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - GVTbegin;
    // 1. Compute vertex-element elastic energy derivatives w.r.t. i and store them in shared memory
    SMatrix<GpuScalar, 3, 4> Hgt = Zeros<GpuScalar, 3, 4>();
    auto Ht                      = Hgt.Slice<3, 3>(0, 0);
    auto gt                      = Hgt.Col(3);
    for (auto elocal = tid; elocal < nAdjacentElements; elocal += kBlockThreads)
    {
        GpuIndex e                   = BDF.GVTn[GVTbegin + elocal];
        GpuIndex ilocal              = BDF.GVTilocal[GVTbegin + elocal];
        SVector<GpuIndex, 4> Te      = FromBuffers<4, 1>(BDF.T, e);
        SMatrix<GpuScalar, 4, 3> GPe = FromFlatBuffer<4, 3>(BDF.GP, e);
        SMatrix<GpuScalar, 3, 4> xe  = FromBuffers(BDF.x, Te.Transpose());
        SVector<GpuScalar, 2> lamee  = FromFlatBuffer<2, 1>(BDF.lame, e);
        GpuScalar wg                 = BDF.wg[e];
        SMatrix<GpuScalar, 3, 3> Fe  = xe * GPe;
        pbat::physics::StableNeoHookeanEnergy<3> Psi{};
        SVector<GpuScalar, 9> gF;
        SMatrix<GpuScalar, 9, 9> HF;
        Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        using pbat::sim::vbd::kernels::AccumulateElasticGradient;
        using pbat::sim::vbd::kernels::AccumulateElasticHessian;
        AccumulateElasticHessian(ilocal, wg, GPe, HF, Ht);
        AccumulateElasticGradient(ilocal, wg, GPe, gF, gt);
    }

    // 2. Compute total vertex hessian and gradient via parallel reduction
    SMatrix<GpuScalar, 3, 4> Hgi = BlockReduce(reinterpret_cast<BlockStorage&>(shared)).Sum(Hgt);
    if (tid > 0)
        return;
    auto Hi = Hgi.Slice<3, 3>(0, 0);
    auto gi = Hgi.Col(3);

    // 3. Add stiffness damping
    GpuScalar mi              = BDF.m[i];
    SVector<GpuScalar, 3> xti = FromBuffers<3, 1>(BDF.xt, i);
    SVector<GpuScalar, 3> xi  = FromBuffers<3, 1>(BDF.x, i);
    using pbat::sim::vbd::kernels::AddDamping;
    AddDamping(BDF.dt, xti, xi, BDF.kD, gi, Hi);

    // 3. Add contact energy
    static auto constexpr kMaxContacts = BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
    SVector<GpuIndex, kMaxContacts> f  = FromFlatBuffer<kMaxContacts, 1>(BDF.fc, i);
    auto nContacts                     = Dot(Ones<GpuIndex, kMaxContacts>(), f >= 0);
    GpuScalar muC                      = (BDF.XVA[i] * BDF.muC) / nContacts;
    for (auto c = 0; c < nContacts; ++c)
    {
        if (f(c) < 0)
            break;

        using pbat::sim::vbd::kernels::AccumulateVertexTriangleContact;
        auto finds = FromBuffers<3, 1>(BDF.F, f(c));
        auto xtf   = FromBuffers(BDF.xt, finds.Transpose());
        auto xf    = FromBuffers(BDF.x, finds.Transpose());
        AccumulateVertexTriangleContact(xti, xi, xtf, xf, BDF.dt, muC, BDF.muF, BDF.epsv, gi, Hi);
    }

    // 4. Add inertial term
    SVector<GpuScalar, 3> xitilde = FromBuffers<3, 1>(BDF.xtilde, i);
    using pbat::sim::vbd::kernels::AddInertiaDerivatives;
    AddInertiaDerivatives(BDF.dt2, mi, xitilde, xi, gi, Hi);

    // 5. Integrate positions
    using pbat::sim::vbd::kernels::IntegratePositions;
    IntegratePositions(gi, Hi, xi, BDF.detHZero);
    ToBuffers(xi, BDF.xb, i);
};

} // namespace kernels
} // namespace vbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_VBD_KERNELS_H
