/**
 * @file Kernels.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief VBD kernels
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */

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
#include <cuda/api/device.hpp>
#include <cuda/api/launch_config_builder.hpp>
#include <limits>

namespace pbat {
namespace gpu {
namespace impl {
namespace vbd {
/**
 * @namespace pbat::gpu::impl::vbd::kernels
 * @brief Device-side VBD kernels
 */
namespace kernels {

using namespace pbat::math::linalg::mini;

/**
 * @brief Device-side BFD1 minimization problem
 */
struct BackwardEulerMinimization
{
    GpuScalar dt;                     ///< Time step
    GpuScalar dt2;                    ///< Squared time step
    GpuScalar* m;                     ///< Lumped mass matrix
    std::array<GpuScalar*, 3> xtilde; ///< Inertial target
    std::array<GpuScalar*, 3> xt;     ///< Previous vertex positions
    std::array<GpuScalar*, 3> x;      ///< Vertex positions
    std::array<GpuScalar*, 3> xb;     ///< Vertex position write buffer

    std::array<GpuIndex*, 4> T; ///< `4x|# elements|` array of tetrahedra
    GpuScalar* wg;              ///< `|# elements|` array of quadrature weights
    GpuScalar* GP;              ///< `4x3x|# elements|` array of shape function gradients
    GpuScalar* lame;            ///< `2x|# elements|` of 1st and 2nd Lame coefficients
    GpuScalar detHZero;         ///< Numerical zero for hessian determinant check
    // GpuScalar const* kD;                  ///< |#elements| array of damping coefficients

    GpuIndex* GVTp;      ///< Vertex-tetrahedron adjacency list's prefix sum
    GpuIndex* GVTn;      ///< Vertex-tetrahedron adjacency list's neighbour list
    GpuIndex* GVTilocal; ///< Vertex-tetrahedron adjacency list's ilocal property

    GpuScalar kD;   ///< Rayleigh damping coefficient
    GpuScalar muC;  ///< Collision penalty
    GpuScalar muF;  ///< Coefficient of friction
    GpuScalar epsv; ///< IPC smooth friction transition function's relative velocity threshold
    static auto constexpr kMaxCollidingTrianglesPerVertex =
        8;                      ///< Maximum number of colliding triangles per vertex
    GpuIndex* fc;               ///< `|# vertices|x|kMaxCollidingTrianglesPerVertex|` array of
                                ///< colliding triangles
    std::array<GpuIndex*, 3> F; ///< `3x|# collision triangles|` array of triangles
    GpuScalar* XVA;             ///< `|# vertices|` array of vertex areas
    GpuScalar* FA;              ///< `|# collision triangles|` array of face areas

    GpuIndex*
        partition; ///< List of vertex indices that can be processed independently, i.e. in parallel

    GpuIndex nVertices; ///< Number of vertices
    GpuScalar* Uetr;    ///< `|# elements|` elastic energies (used for Trust Region acceleration)
    std::array<GpuScalar*, 3> ftr; ///< `3 x |# verts|` per-vertex objective function values (used
                                   ///< for Trust Region acceleration)
    GpuScalar* Rtr;                ///< `|# verts|` Trust Region radius
};

template <auto kBlockThreads>
__global__ void VbdIteration(BackwardEulerMinimization BDF);

/**
 * @brief Traits for VBD iteration kernel
 * @tparam kBlockThreads Number of threads per block
 */
template <auto kBlockThreads>
struct VbdIterationTraits
{
  public:
    using ElasticDerivativeStorageType = SMatrix<GpuScalar, 3, 4>; ///< Type of data to reduce
    using BlockReduce =
        cub::BlockReduce<ElasticDerivativeStorageType, kBlockThreads>; ///< Reduction
    using BlockStorage = typename BlockReduce::TempStorage;            ///< Storage for reduction

    static auto constexpr kDynamicSharedMemorySize =
        sizeof(BlockStorage); ///< Dynamic shared memory size

    /**
     * @brief Get the raw kernel
     * @return Handle to the kernel
     */
    static auto Kernel() { return &VbdIteration<kBlockThreads>; }
};

/**
 * @brief VBD iteration kernel
 *
 * @tparam kBlockThreads Number of threads per block
 * @param BDF BDF1 time-stepping minimization problem
 */
template <auto kBlockThreads>
__global__ void VbdIteration(BackwardEulerMinimization BDF)
{
    // Get thread info
    using Traits       = VbdIterationTraits<kBlockThreads>;
    using BlockReduce  = typename Traits::BlockReduce;
    using BlockStorage = typename Traits::BlockStorage;
    extern __shared__ __align__(alignof(BlockStorage)) char shared[];
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    // Get vertex-tet adjacency information
    GpuIndex i                 = BDF.partition[bid]; // Vertex index
    GpuIndex GVTbegin          = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - GVTbegin;
    // 1. Compute vertex-element elastic energy derivatives w.r.t. i and store them in shared
    // memory
    SMatrix<GpuScalar, 3, 4> Hgi = Zeros<GpuScalar, 3, 4>();
    auto Hi                      = Hgi.Slice<3, 3>(0, 0);
    auto gi                      = Hgi.Col(3);
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
        AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
        AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
    }

    // 2. Compute total vertex hessian and gradient via parallel reduction
    Hgi = BlockReduce(reinterpret_cast<BlockStorage&>(shared)).Sum(Hgi);
    if (tid > 0)
        return;

    // Load vertex data
    GpuScalar mi                  = BDF.m[i];
    SVector<GpuScalar, 3> xti     = FromBuffers<3, 1>(BDF.xt, i);
    SVector<GpuScalar, 3> xitilde = FromBuffers<3, 1>(BDF.xtilde, i);
    SVector<GpuScalar, 3> xi      = FromBuffers<3, 1>(BDF.x, i);

    // 3. Add stiffness damping
    using pbat::sim::vbd::kernels::AddDamping;
    AddDamping(BDF.dt, xti, xi, BDF.kD, gi, Hi);

    // 3. Add contact energy
    static auto constexpr kMaxContacts = BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
    SVector<GpuIndex, kMaxContacts> f  = FromFlatBuffer<kMaxContacts, 1>(BDF.fc, i);
    auto nContacts                     = Dot(Ones<GpuIndex, kMaxContacts>(), f >= 0);
    SVector<GpuScalar, kMaxContacts> fa = Zeros<GpuIndex, kMaxContacts>();
    for (auto c = 0; c < nContacts; ++c)
        fa(c) = BDF.FA[f(c)];
    auto sumfa    = Dot(fa, Ones<GpuScalar, kMaxContacts>());
    GpuScalar muC = (BDF.XVA[i] * BDF.muC) / sumfa;
    for (auto c = 0; c < nContacts; ++c)
    {
        using pbat::sim::vbd::kernels::AccumulateVertexTriangleContact;
        auto finds = FromBuffers<3, 1>(BDF.F, f(c));
        auto xtf   = FromBuffers(BDF.xt, finds.Transpose());
        auto xf    = FromBuffers(BDF.x, finds.Transpose());
        AccumulateVertexTriangleContact(
            xti,
            xi,
            xtf,
            xf,
            BDF.dt,
            fa(c) * muC,
            BDF.muF,
            BDF.epsv,
            &gi,
            &Hi);
    }

    // 4. Add inertial term
    using pbat::sim::vbd::kernels::AddInertiaDerivatives;
    AddInertiaDerivatives(BDF.dt2, mi, xitilde, xi, gi, Hi);

    // 5. Integrate positions
    using pbat::sim::vbd::kernels::IntegratePositions;
    IntegratePositions(gi, Hi, xi, BDF.detHZero);
    ToBuffers(xi, BDF.xb, i);
}

template <auto kBlockThreads>
__global__ static void AccumulateVertexEnergies(BackwardEulerMinimization BDF);

/**
 * @brief Traits for vertex energy accumulation kernel
 *
 * @tparam kBlockThreads Number of threads per block
 */
template <auto kBlockThreads>
struct AccumulateVertexEnergiesTraits
{
  public:
    using BlockReduce  = cub::BlockReduce<GpuScalar, kBlockThreads>; ///< Reduction
    using BlockStorage = typename BlockReduce::TempStorage;          ///< Storage for reduction

    static auto constexpr kDynamicSharedMemorySize =
        sizeof(BlockStorage); ///< Dynamic shared memory size

    /**
     * @brief Get the raw kernel
     * @return Handle to the kernel
     */
    static auto Kernel() { return &AccumulateVertexEnergies<kBlockThreads>; }
};

/**
 * @brief Accumulate vertex energies kernel
 *
 * @tparam kBlockThreads Number of threads per block
 * @param BDF BDF1 time-stepping minimization problem
 */
template <auto kBlockThreads>
__global__ static void AccumulateVertexEnergies(BackwardEulerMinimization BDF)
{
    using Traits       = AccumulateVertexEnergiesTraits<kBlockThreads>;
    using BlockReduce  = typename Traits::BlockReduce;
    using BlockStorage = typename Traits::BlockStorage;
    extern __shared__ __align__(alignof(BlockStorage)) char shared[];

    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto i   = bid;

    // Vertex objective function
    GpuScalar fi{0};

    // 1. Compute total vertex elastic energy
    GpuIndex GVTbegin          = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - GVTbegin;
    for (auto elocal = tid; elocal < nAdjacentElements; elocal += kBlockThreads)
    {
        GpuIndex e = BDF.GVTn[GVTbegin + elocal];
        fi += BDF.Uetr[e];
    }
    fi = BlockReduce(reinterpret_cast<BlockStorage&>(shared)).Sum(fi);
    if (tid > 0)
        return;

    // Load vertex data
    GpuScalar mi                  = BDF.m[i];
    SVector<GpuScalar, 3> xti     = FromBuffers<3, 1>(BDF.xt, i);
    SVector<GpuScalar, 3> xitilde = FromBuffers<3, 1>(BDF.xtilde, i);
    SVector<GpuScalar, 3> xi      = FromBuffers<3, 1>(BDF.x, i);

    // 2. Add contact energy
    static auto constexpr kMaxContacts = BackwardEulerMinimization::kMaxCollidingTrianglesPerVertex;
    SVector<GpuIndex, kMaxContacts> f  = FromFlatBuffer<kMaxContacts, 1>(BDF.fc, i);
    auto nContacts                     = Dot(Ones<GpuIndex, kMaxContacts>(), f >= 0);
    SVector<GpuScalar, kMaxContacts> fa = Zeros<GpuIndex, kMaxContacts>();
    for (auto c = 0; c < nContacts; ++c)
        fa(c) = BDF.FA[f(c)];
    auto sumfa    = Dot(fa, Ones<GpuScalar, kMaxContacts>());
    GpuScalar muC = (BDF.XVA[i] * BDF.muC) / sumfa;
    for (auto c = 0; c < nContacts; ++c)
    {
        using pbat::sim::vbd::kernels::AccumulateVertexTriangleContact;
        auto finds = FromBuffers<3, 1>(BDF.F, f(c));
        auto xtf   = FromBuffers(BDF.xt, finds.Transpose());
        auto xf    = FromBuffers(BDF.x, finds.Transpose());
        fi += AccumulateVertexTriangleContact(
            xti,
            xi,
            xtf,
            xf,
            BDF.dt,
            fa(c) * muC,
            BDF.muF,
            BDF.epsv,
            static_cast<SVector<GpuScalar, 3>*>(nullptr),
            static_cast<SMatrix<GpuScalar, 3, 3>*>(nullptr));
    }

    // 4. Add inertial term
    fi += GpuScalar(0.5) * mi * SquaredNorm(xi - xti);

    // 5. Store vertex objective function
    // BDF.ftr[k % 3] = fi;
}

/**
 * @brief Invokes a VBD kernel on the GPU with the specified number of blocks and threads
 *
 * @tparam TKernelTraits Kernel traits type
 * @tparam TArgs Argument types
 * @param nBlocks Grid size
 * @param nThreads Block size
 * @param args Arguments to pass to the kernel
 */
template <template <auto> class TKernelTraits, class... TArgs>
void Invoke(GpuIndex nBlocks, GpuIndex nThreads, TArgs&&... args)
{
    pbat::common::ForValues<32, 64, 128, 256, 512>([&]<auto kBlockThreads>() {
        if (nThreads > kBlockThreads / 2 and nThreads <= kBlockThreads)
        {
            using KernelTraitsType        = TKernelTraits<kBlockThreads>;
            auto kDynamicSharedMemorySize = static_cast<cuda::memory::shared::size_t>(
                sizeof(KernelTraitsType::kDynamicSharedMemorySize));
            auto kernelLaunchConfiguration =
                cuda::launch_config_builder()
                    .block_size(kBlockThreads)
                    .dynamic_shared_memory_size(kDynamicSharedMemorySize)
                    .grid_size(nBlocks)
                    .build();
            cuda::device::current::get().launch(
                KernelTraitsType::Kernel(),
                kernelLaunchConfiguration,
                std::forward<TArgs>(args)...);
        }
    });
}

} // namespace kernels
} // namespace vbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_VBD_KERNELS_H
