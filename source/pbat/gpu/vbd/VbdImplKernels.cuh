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
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar const*, 3> vt;
    std::array<GpuScalar const*, 3> aext;
    std::array<GpuScalar*, 3> xtilde;
}

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
        GpuScalar const atext  = Dot(GetAcceleration(i), aexti) / SquaredNorm(aexti, aexti);
        GpuScalar atilde       = min(max(atext, GpuScalar{0}), GpuScalar{1});
        bool const bWasClamped = (atilde == GpuScalar{0}) or (atilde == GpuScalar{1});
        atilde                 = (not bWasClamped) * atext + bWasClamped * atilde;
        for (auto d = 0; d < 3; ++d)
            x[d][i] = xt[d][i] + dt * vt[d][i] + dt2 * atilde * aext[d][i];
    }

    GpuScalar dt;
    GpuScalar dt2;
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar const*, 3> vtm1;
    std::array<GpuScalar const*, 3> vt;
    std::array<GpuScalar const*, 3> aext;
    std::array<GpuScalar*, 3> x;
};

struct FChebyshev
{
    FChebyshev(
        GpuScalar rho,
        std::array<GpuScalar const*, 3> xtm1,
        std::array<GpuScalar const*, 3> xt,
        std::array<GpuScalar*, 3> x)
        : rho2(rho * rho), omega(GpuScalar{1}), xtm1(xtm1), xt(xt), x(x)
    {
    }

    __host__ Update(auto k)
    {
        omega = (k == 1) ? omega = GpuScalar{2} / (GpuScalar{2} - rho2) :
                           GpuScalar{4} / (GpuScalar{4} - rho2 * omega);
    }

    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
            x[d][i] = omega * (x[d][i] - xt[d][i]) + xtm1[d][i];
    }

    GpuScalar rho2;
    GpuScalar omega;
    std::array<GpuScalar const*, 3> xtm1;
    std::array<GpuScalar const*, 3> xt;
    std::array<GpuScalar*, 3> x;
};

struct FFinalizeSolution
{
    __device__ void operator()(auto i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            xtm1[d][i] = xt[d][i];
            vtm1[d][i] = v[d][i];
            v[d][i]    = (x[d][i] - xt[d][i]) / dt;
            xt[d][i]   = x[d][i];
        }
    }

    GpuScalar dt;
    std::array<GpuScalar*, 3> xtm1;
    std::array<GpuScalar*, 3> vtm1;
    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
};

struct BackwardEulerMinimization
{
    GpuScalar dt;                           ///< Time step
    GpuScalar dt2;                          ///< Squared time step
    GpuScalar const* m;                     ///< Lumped mass matrix
    std::array<GpuScalar const*, 3> xtilde; ///< Inertial target
    std::array<GpuScalar*, 3> xt;           ///< Previous vertex positions
    std::array<GpuScalar*, 3> x;            ///< Vertex positions

    GpuIndex const* T;                    ///< 4x|#elements| array of tetrahedra
    GpuScalar const* GP;                  ///< 4x3x|#elements| array of shape function gradients
    std::array<GpuScalar const*, 2> lame; ///< 2x|#elements| of 1st and 2nd Lame coefficients
    GpuScalar const* kD;                  ///< |#elements| array of damping coefficients

    GpuIndex const* GVTn;      ///< Vertex-tetrahedron adjacency list's neighbour list
    GpuIndex const* GVTp;      ///< Vertex-tetrahedron adjacency list's prefix sum
    GpuIndex const* GVTilocal; ///< Vertex-tetrahedron adjacency list's ilocal property

    GpuScalar kC;                             ///< Collision penalty
    GpuIndex nMaxCollidingTrianglesPerVertex; ///< Memory capacity for storing vertex triangle
                                              ///< collision constraints
    GpuIndex const*
        FC; ///< |#vertices|x|nMaxCollidingTrianglesPerVertex| array of colliding triangles
    GpuIndex const* nCollidingTriangles; ///< |#vertices| array of the number of colliding triangles
                                         ///< for each vertex.
    GpuIndex const* F;                   ///< 3x|#collision triangles| array of triangles

    GpuIndex const*
        partition; ///< List of vertex indices that can be processed independently, i.e. in parallel

    template <class ScalarType, auto kRows, auto kCols = 1>
    using Matrix = pbat::gpu::math::linalg::Matrix<ScalarType, kRows, kCols>;

    template <class ScalarType>
    __device__ Matrix<ScalarType, 3> ToLocal(auto vi, std::array<ScalarType*, 3> vData) const
    {
        Matrix<ScalarType, 3> vlocal;
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
        vGlobal[0][vi] = vData(0);
        vGlobal[1][vi] = vData(1);
        vGlobal[2][vi] = vData(2);
    }

    __device__ Matrix<GpuScalar, 4, 3> BasisFunctionGradients(auto e) const
    {
        using namespace pbat::gpu::math::linalg;
        Matrix<GpuScalar, 4, 3> GP = MatrixView<GpuScalar, 4, 3>(BDF.GP + e * 12);
        return GP;
    }

    __device__ Matrix<GpuScalar, 3, 4> ElementVertexPositions(auto e) const
    {
        Matrix<GpuScalar, 3, 4> xe;
        for (auto i = 0; i < 4; ++i)
        {
            GpuIndex vi = T[e * 4 + i];
            xe.Col(i)   = ToLocal(vi, x);
        }
        return xe;
    }

    __device__ Matrix<GpuScalar, 9, 10>
    StableNeoHookeanDerivativesWrtF(auto e, Matrix<GpuScalar, 4, 3> const& GP) const
    {
        Matrix<GpuScalar, 3, 3> F = BDF.ElementVertexPositions(e) * GP;
        GpuScalar mu              = lame[0][e];
        GpuScalar lambda          = lame[1][e];

        Matrix<GpuScalar, 9, 10> HGe;
        auto He = HGe.Slice<9, 9>(0, 0);
        auto ge = HGe.Col(9);

        // Auto-generated from pbat/physics/StableNeoHookeanEnergy.h
        GpuScalar const a0  = F(4) * F(8);
        GpuScalar const a1  = F(5) * F(7);
        GpuScalar const a2  = 2 * a0 - 2 * a1;
        GpuScalar const a3  = F(3) * F(8);
        GpuScalar const a4  = F(4) * F(6);
        GpuScalar const a5  = lambda * (-a1 * F(0) - a3 * F(1) - a4 * F(2) + F(0) * F(4) * F(8) +
                                       F(1) * F(5) * F(6) + F(2) * F(3) * F(7) - 1 - mu / lambda);
        GpuScalar const a6  = (1.0 / 2.0) * a5;
        GpuScalar const a7  = -2 * a3 + 2 * F(5) * F(6);
        GpuScalar const a8  = F(3) * F(7);
        GpuScalar const a9  = -2 * a4 + 2 * a8;
        GpuScalar const a10 = F(1) * F(8);
        GpuScalar const a11 = -2 * a10 + 2 * F(2) * F(7);
        GpuScalar const a12 = F(0) * F(8);
        GpuScalar const a13 = F(2) * F(6);
        GpuScalar const a14 = 2 * a12 - 2 * a13;
        GpuScalar const a15 = F(0) * F(7);
        GpuScalar const a16 = -2 * a15 + 2 * F(1) * F(6);
        GpuScalar const a17 = F(1) * F(5);
        GpuScalar const a18 = F(2) * F(4);
        GpuScalar const a19 = 2 * a17 - 2 * a18;
        GpuScalar const a20 = F(0) * F(5);
        GpuScalar const a21 = -2 * a20 + 2 * F(2) * F(3);
        GpuScalar const a22 = F(0) * F(4);
        GpuScalar const a23 = F(1) * F(3);
        GpuScalar const a24 = 2 * a22 - 2 * a23;
        GpuScalar const a25 = (1.0 / 2.0) * lambda;
        GpuScalar const a26 = a25 * (a0 - a1);
        GpuScalar const a27 = a5 * F(8);
        GpuScalar const a28 = a5 * F(7);
        GpuScalar const a29 = -a28;
        GpuScalar const a30 = a5 * F(5);
        GpuScalar const a31 = -a30;
        GpuScalar const a32 = a5 * F(4);
        GpuScalar const a33 = a25 * (-a3 + F(5) * F(6));
        GpuScalar const a34 = -a27;
        GpuScalar const a35 = a5 * F(6);
        GpuScalar const a36 = a5 * F(3);
        GpuScalar const a37 = -a36;
        GpuScalar const a38 = a25 * (-a4 + a8);
        GpuScalar const a39 = -a35;
        GpuScalar const a40 = -a32;
        GpuScalar const a41 = a25 * (-a10 + F(2) * F(7));
        GpuScalar const a42 = a5 * F(2);
        GpuScalar const a43 = a5 * F(1);
        GpuScalar const a44 = -a43;
        GpuScalar const a45 = a25 * (a12 - a13);
        GpuScalar const a46 = -a42;
        GpuScalar const a47 = a5 * F(0);
        GpuScalar const a48 = a25 * (-a15 + F(1) * F(6));
        GpuScalar const a49 = -a47;
        GpuScalar const a50 = a25 * (a17 - a18);
        GpuScalar const a51 = a25 * (-a20 + F(2) * F(3));
        GpuScalar const a52 = a25 * (a22 - a23);
        ge(0)               = a2 * a6 + mu * F(0);
        ge(1)               = a6 * a7 + mu * F(1);
        ge(2)               = a6 * a9 + mu * F(2);
        ge(3)               = a11 * a6 + mu * F(3);
        ge(4)               = a14 * a6 + mu * F(4);
        ge(5)               = a16 * a6 + mu * F(5);
        ge(6)               = a19 * a6 + mu * F(6);
        ge(7)               = a21 * a6 + mu * F(7);
        ge(8)               = a24 * a6 + mu * F(8);
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

    __device__ void StableNeoHookeanGradAndHessian(auto e, auto ilocal, GpuScalar* Hge) const
    {
        // Compute (d^k Psi / dF^k)
        Matrix<GpuScalar, 4, 3> GP   = BDF.BasisFunctionGradients(e);
        Matrix<GpuScalar, 9, 10> HGF = StableNeoHookeanDerivativesWrtF(e, GP);
        auto HF                      = HGF.Slice<9, 9>(0, 0);
        auto gF                      = HGF.Col(9);
        // Write vertex-specific derivatives into output memory HGe
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 4> HGei(Hge);
        HGei.SetZero();
        auto Hi = HGei.Slice<3, 3>(0, 0);
        auto gi = HGei.Col(3);
        // Contract (d^k Psi / dF^k) with (d F / dx)^k. See pbat/fem/DeformationGradient.h.
        for (auto kj = 0; kj < 3; ++kj)
            for (auto ki = 0; ki < 3; ++ki)
                Hi += GP(ilocal, ki) * GP(ilocal, kj) * HF.Slice<3, 3>(ki * 3, kj * 3);
        for (auto k = 0; k < 3; ++k)
            gi += GP(ilocal, k) * gF.Slice<3, 1>(k * 3, 0);
        return HGei;
    }
};

__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF)
{
    extern __shared__ GpuScalar shared[];
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    // Vertex index
    GpuIndex i = BDF.partition[bid];
    // Get vertex-tet adjacency information
    GpuIndex ke                = BDF.GVTp[i];
    GpuIndex nAdjacentElements = BDF.GVTp[i + 1] - ke;
    assert(nAdjacentElements <= blockDim.x);
    if (tid >= nAdjacentElements) // Thread not associated with any adjacent element
        return;
    GpuIndex e      = BDF.GVTn[ke + tid];
    GpuIndex ilocal = BDF.GVTilocal[ke + tid];

    // Each element/thread has a 3x3 hessian + 3x1 gradient = 12 scalars/element
    GpuScalar* Hge = shared + tid * 12;

    using namespace pbat::gpu::math::linalg;

    // 1. Compute element elastic energy derivatives w.r.t. i
    Matrix<GpuScalar, 3, 4> HGei = BDF.StableNeoHookeanGradAndHessian(e, ilocal, Hge);
    __syncthreads();

    // Remaining execution is synchronous, i.e. only 1 thread is required
    if (tid > 0)
        return;

    // 2. Accumulate results into vertex hessian and gradient
    Matrix<GpuScalar, 3> xti     = BDF.ToLocal(i, BDF.xt);
    Matrix<GpuScalar, 3> xitilde = BDF.ToLocal(i, BDF.xtilde);
    Matrix<GpuScalar, 3> xi      = BDF.ToLocal(i, BDF.x);
    Matrix<GpuScalar, 3, 3> Hi;
    Hi.SetZero();
    Matrix<GpuScalar, 3, 1> gi;
    gi.SetZero();
    for (auto k = 0; k < nAdjacentElements; ++k)
    {
        GpuScalar* HiShared = Hge + k * 12;
        GpuScalar* giShared = HiShared + 9;
        MatrixView<GpuScalar, 3, 3> Hei(HiShared);
        MatrixView<GpuScalar, 3, 1> gei(giShared);
        GpuScalar const D = BDF.kD / BDF.dt; // Rayleigh damping term
        Hi += (GpuScalar{1} + D) * Hei;
        gi += gei + Hei * (xi - xti) * D;
    }
    Identity<GpuScalar, 3, 3> I{};
    GpuScalar mi = BDF.m[i];
    Hi += (mi / BDF.dt2) * I;
    gi += (mi / BDF.dt2) * (xi - xitilde);

    // 3. Newton step
    if (abs(Determinant(Hi)) <= GpuScalar{1e-6}) // Skip nearly rank-deficient hessian
        return;
    xi = xi - (Inverse(Hi) * gi);

    // 4. Commit vertex descent step
    BDF.ToGlobal(i, xi, BDF.x);
}

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_IMPL_KERNELS_CUH