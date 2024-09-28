#ifndef PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH
#define PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>
#include <cstddef>

namespace pbat {
namespace gpu {
namespace vbd {
namespace kernels {

struct FInertialTarget
{
    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            xtilde[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * aext[d][i];
        }
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> vt;
    std::array<GpuScalar*, 3> aext;
    std::array<GpuScalar*, 3> xtilde;
};

struct FAdaptiveInitialization
{
    using Vector3 = pbat::gpu::math::linalg::Matrix<GpuScalar, 3>;

    __device__ Vector3 GetExternalAcceleration(auto i) const
    {
        Vector3 aexti;
        aexti(0) = aext[0][i];
        aexti(1) = aext[1][i];
        aexti(2) = aext[2][i];
        return aexti;
    }

    __device__ Vector3 GetAcceleration(auto i) const
    {
        Vector3 at;
        at(0) = (vt[0][i] - vtm1[0][i]) / dt;
        at(1) = (vt[1][i] - vtm1[1][i]) / dt;
        at(2) = (vt[2][i] - vtm1[2][i]) / dt;
        return at;
    }

    __device__ void operator()(auto i)
    {
        using namespace pbat::gpu::math::linalg;
        Vector3 const aexti    = GetExternalAcceleration(i);
        GpuScalar const atext  = Dot(GetAcceleration(i), aexti) / SquaredNorm(aexti);
        GpuScalar atilde       = min(max(atext, GpuScalar{0}), GpuScalar{1});
        bool const bWasClamped = (atilde == GpuScalar{0}) or (atilde == GpuScalar{1});
        atilde                 = (not bWasClamped) * atext + bWasClamped * atilde;
        for (auto d = 0; d < 3; ++d)
            x[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * atilde * aext[d][i];
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> vtm1;
    std::array<GpuScalar*, 3> vt;
    std::array<GpuScalar*, 3> aext;
    std::array<GpuScalar*, 3> x;
};

struct FChebyshev
{
    FChebyshev(
        GpuScalar rho,
        std::array<GpuScalar*, 3> xkm2,
        std::array<GpuScalar*, 3> xkm1,
        std::array<GpuScalar*, 3> x)
        : rho2(rho * rho), omega(GpuScalar{1}), xkm2(xkm2), xkm1(xkm1), x(x)
    {
    }

    void Update(auto k)
    {
        omega = (k == 0) ? omega = GpuScalar{2} / (GpuScalar{2} - rho2) :
                           GpuScalar{4} / (GpuScalar{4} - rho2 * omega);
    }

    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            x[d][i]    = omega * (x[d][i] - xkm2[d][i]) + xkm2[d][i];
            xkm2[d][i] = xkm1[d][i];
            xkm1[d][i] = x[d][i];
        }
    }

    GpuScalar rho2;
    GpuScalar omega;
    std::array<GpuScalar*, 3> xkm2;
    std::array<GpuScalar*, 3> xkm1;
    std::array<GpuScalar*, 3> x;
};

struct FUpdateVelocity
{
    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            v[d][i] = (x[d][i] - xt[d][i]) / dt;
        }
    }

    GpuScalar dt;
    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
};

struct BackwardEulerMinimization
{
    GpuScalar dt;                     ///< Time step
    GpuScalar dt2;                    ///< Squared time step
    GpuScalar* m;                     ///< Lumped mass matrix
    std::array<GpuScalar*, 3> xtilde; ///< Inertial target
    std::array<GpuScalar*, 3> xt;     ///< Previous vertex positions
    std::array<GpuScalar*, 3> x;      ///< Vertex positions

    std::array<GpuIndex*, 4> T; ///< 4x|#elements| array of tetrahedra
    GpuScalar* wg;              ///< |#elements| array of quadrature weights
    GpuScalar* GP;              ///< 4x3x|#elements| array of shape function gradients
    GpuScalar* lame;            ///< 2x|#elements| of 1st and 2nd Lame coefficients
    // GpuScalar const* kD;                  ///< |#elements| array of damping coefficients

    GpuIndex* GVTp;      ///< Vertex-tetrahedron adjacency list's prefix sum
    GpuIndex* GVTn;      ///< Vertex-tetrahedron adjacency list's neighbour list
    GpuIndex* GVTilocal; ///< Vertex-tetrahedron adjacency list's ilocal property

    GpuScalar kD;                             ///< Rayleigh damping coefficient
    GpuScalar kC;                             ///< Collision penalty
    GpuIndex nMaxCollidingTrianglesPerVertex; ///< Memory capacity for storing vertex triangle
                                              ///< collision constraints
    GpuIndex* FC; ///< |#vertices|x|nMaxCollidingTrianglesPerVertex| array of colliding triangles
    GpuIndex* nCollidingTriangles; ///< |#vertices| array of the number of colliding triangles
                                   ///< for each vertex.
    std::array<GpuIndex*, 4> F;    ///< 3x|#collision triangles| array of triangles

    GpuIndex*
        partition; ///< List of vertex indices that can be processed independently, i.e. in parallel

    template <class ScalarType, auto kRows, auto kCols = 1>
    using Matrix = pbat::gpu::math::linalg::Matrix<ScalarType, kRows, kCols>;

    template <class ScalarType>
    __device__ Matrix<std::remove_const_t<ScalarType>, 3>
    ToLocal(auto vi, std::array<ScalarType*, 3> vData) const
    {
        using UnderlyingScalarType = std::remove_const_t<ScalarType>;
        Matrix<UnderlyingScalarType, 3> vlocal;
        vlocal(0) = vData[0][vi];
        vlocal(1) = vData[1][vi];
        vlocal(2) = vData[2][vi];
        return vlocal;
    }

    template <class ScalarType>
    __device__ void
    ToGlobal(auto vi, Matrix<ScalarType, 3> const& vData, std::array<ScalarType*, 3> vGlobalData)
        const
    {
        vGlobalData[0][vi] = vData(0);
        vGlobalData[1][vi] = vData(1);
        vGlobalData[2][vi] = vData(2);
    }

    __device__ Matrix<GpuScalar, 4, 3> BasisFunctionGradients(GpuIndex e) const;

    __device__ Matrix<GpuScalar, 3, 4> ElementVertexPositions(GpuIndex e) const;

    __device__ Matrix<GpuScalar, 9, 10> StableNeoHookeanDerivativesWrtF(
        GpuIndex e,
        Matrix<GpuScalar, 3, 3> const& Fe,
        GpuScalar mu,
        GpuScalar lambda) const;

    __device__ void
    ComputeStableNeoHookeanDerivatives(GpuIndex e, GpuIndex ilocal, GpuScalar* Hge) const;

    constexpr auto ExpectedSharedMemoryPerThreadInBytes() const { return 12 * sizeof(GpuScalar); }
};

__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF);

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH