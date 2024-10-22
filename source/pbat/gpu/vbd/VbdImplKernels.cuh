#ifndef PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH
#define PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/math/linalg/Matrix.cuh"
#include "pbat/gpu/vbd/InitializationStrategy.h"

#include <array>
#include <cstddef>
#include <limits>

namespace pbat {
namespace gpu {
namespace vbd {
namespace kernels {

struct FKineticEnergyMinimum
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
        // In the original VBD paper, they say that they estimate motion by computing the current
        // acceleration, i.e. a(t) = (v(t) - v(t-1)) / dt. However, acceleration is not a measure of
        // motion, but of total loading. Unfortunately, estimating loading results in unstable
        // ping-pong motion. This is because accelerations opposite the external acceleration result
        // in ignoring external forces in the initialization. Thus, motion that goes against gravity
        // totally ignores gravity. Then, suddenly, when velocity in the direction opposite gravity
        // decreases, acceleration suddenly becomes aligned with external acceleration (i.e.
        // gravity), and the adaptive scheme becomes biased towards it, helping to "push" motion in
        // that direction.
        Vector3 at;
        at(0) = (vt[0][i] - vtm1[0][i]) / dt;
        at(1) = (vt[1][i] - vtm1[1][i]) / dt;
        at(2) = (vt[2][i] - vtm1[2][i]) / dt;
        return at;
    }

    __device__ Vector3 GetLinearizedMotion(auto i) const
    {
        // We replace acceleration by the velocity's direction to actually estimate motion.
        Vector3 v;
        v(0) = vt[0][i];
        v(1) = vt[1][i];
        v(2) = vt[2][i];
        using namespace pbat::gpu::math::linalg;
        return v / (Norm(v) + std::numeric_limits<GpuScalar>::min());
    }

    __device__ void operator()(auto i)
    {
        using namespace pbat::gpu::math::linalg;
        if (strategy == EInitializationStrategy::Position)
        {
            for (auto d = 0; d < 3; ++d)
                x[d][i] = xt[d][i];
        }
        else if (strategy == EInitializationStrategy::Inertia)
        {
            for (auto d = 0; d < 3; ++d)
                x[d][i] = xt[d][i] + dt * vt[d][i];
        }
        else if (strategy == EInitializationStrategy::KineticEnergyMinimum)
        {
            for (auto d = 0; d < 3; ++d)
                x[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * aext[d][i];
        }
        else // (strategy == EInitializationStrategy::AdaptiveVbd)
        {
            Vector3 aexti                           = GetExternalAcceleration(i);
            GpuScalar const aextin2                 = SquaredNorm(aexti);
            bool const bHasZeroExternalAcceleration = (aextin2 == GpuScalar{0});
            GpuScalar atilde{0};
            if (not bHasZeroExternalAcceleration)
            {
                if (strategy == EInitializationStrategy::AdaptiveVbd)
                {
                    Vector3 const ati = GetAcceleration(i);
                    atilde            = Dot(ati, aexti) / aextin2;
                    atilde            = min(max(atilde, GpuScalar{0}), GpuScalar{1});
                }
                if (strategy == EInitializationStrategy::AdaptivePbat)
                {
                    Vector3 const dti = GetLinearizedMotion(i);
                    atilde            = Dot(dti, aexti) / aextin2;
                    // Discard the sign of atilde, because motion that goes against
                    // gravity should "feel" gravity, rather than ignore it (i.e. clamping).
                    atilde = min(abs(atilde), GpuScalar{1});
                }
            }
            for (auto d = 0; d < 3; ++d)
                x[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * atilde * aexti(d);
        }
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> vtm1;
    std::array<GpuScalar*, 3> vt;
    std::array<GpuScalar*, 3> aext;
    std::array<GpuScalar*, 3> x;
    EInitializationStrategy strategy;
};

struct FChebyshev
{
    FChebyshev(
        GpuScalar rho,
        std::array<GpuScalar*, 3> xkm2,
        std::array<GpuScalar*, 3> xkm1,
        std::array<GpuScalar*, 3> x)
        : k(), rho2(rho * rho), omega(), xkm2(xkm2), xkm1(xkm1), x(x)
    {
    }

    void SetIteration(auto kIn)
    {
        k     = kIn;
        omega = (k == 0) ? GpuScalar{1} :
                (k == 1) ? omega = GpuScalar{2} / (GpuScalar{2} - rho2) :
                           GpuScalar{4} / (GpuScalar{4} - rho2 * omega);
    }

    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            if (k > 1)
                x[d][i] = omega * (x[d][i] - xkm2[d][i]) + xkm2[d][i];

            xkm2[d][i] = xkm1[d][i];
            xkm1[d][i] = x[d][i];
        }
    }

    GpuIndex k;
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
    GpuScalar detHZero;         ///< Numerical zero for hessian determinant check
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

    constexpr auto ExpectedSharedMemoryPerThreadInScalars() const { return 12; }
    constexpr auto ExpectedSharedMemoryPerThreadInBytes() const
    {
        return ExpectedSharedMemoryPerThreadInScalars() * sizeof(GpuScalar);
    }
    constexpr auto SharedHessianOffset() const { return 0; }
    constexpr auto SharedGradientOffset() const { return 9; }
};

__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF);

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH